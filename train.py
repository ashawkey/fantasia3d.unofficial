# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import glob
import time
import argparse
import json
import itertools

import numpy as np
import torch
import nvdiffrast.torch as dr
import xatlas

# Import data readers / generators
from dataset.dataset_mesh import DatasetDream

# Import topology / geometry trainers
from geometry.dmtet import DMTetGeometry
from geometry.dlmesh import DLMesh

import render.renderutils as ru
from render import obj
from render import material
from render import util
from render import mesh
from render import texture
from render import mlptexture
from render import light
from render import render

RADIUS = 3.0

from sd_utils import StableDiffusion

# Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)

###############################################################################
# Loss setup
###############################################################################

@torch.no_grad()
def createLoss(FLAGS):
    if FLAGS.loss == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
    elif FLAGS.loss == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
    elif FLAGS.loss == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb')
    elif FLAGS.loss == "logl2":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb')
    elif FLAGS.loss == "relmse":
        return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none')
    else:
        assert False

###############################################################################
# Mix background into a dataset image
###############################################################################

@torch.no_grad()
def prepare_batch(target, bg_type='black'):

    target['mv'] = target['mv'].cuda()
    target['mvp'] = target['mvp'].cuda()
    target['campos'] = target['campos'].cuda()

    return target

###############################################################################
# UV - map geometry & convert to a mesh
###############################################################################

@torch.no_grad()
def xatlas_uvmap(glctx, geometry, mat, FLAGS):
    eval_mesh, _ = geometry.getMesh(mat)
    
    # Create uvs with xatlas
    v_pos = eval_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = eval_mesh.t_pos_idx.detach().cpu().numpy()
    vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    
    uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
    faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

    new_mesh = mesh.Mesh(v_tex=uvs, t_tex_idx=faces, base=eval_mesh)

    mask, kd, ks, normal = render.render_uv(glctx, new_mesh, FLAGS.texture_res, eval_mesh.material['kd_ks_normal'])
    
    if FLAGS.layers > 1:
        kd = torch.cat((kd, torch.rand_like(kd[...,0:1])), dim=-1)

    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

    new_mesh.material = material.Material({
        'bsdf'   : mat['bsdf'],
        'kd'     : texture.Texture2D(kd, min_max=[kd_min, kd_max]),
        'ks'     : texture.Texture2D(ks, min_max=[ks_min, ks_max]),
        'normal' : texture.Texture2D(normal, min_max=[nrm_min, nrm_max])
    })

    return new_mesh

###############################################################################
# Utility functions for material
###############################################################################

def initial_guess_material(geometry, mlp, FLAGS, init_mat=None, bsdf='pbr'):
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')
    
    if mlp:
        mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0) # [9,]
        mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0) # [9,]
        # input 3d coord [..., 3], output 9-channel texture in mlp_min~mlp_max [..., 9]
        mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=9, min_max=[mlp_min, mlp_max])
        mat =  material.Material({'kd_ks_normal' : mlp_map_opt})
    else:
        # Setup Kd (albedo) and Ks (x, roughness, metalness) textures
        if FLAGS.random_textures or init_mat is None:
            num_channels = 4 if FLAGS.layers > 1 else 3
            kd_init = torch.rand(size=FLAGS.texture_res + [num_channels], device='cuda') * (kd_max - kd_min)[None, None, 0:num_channels] + kd_min[None, None, 0:num_channels]
            kd_map_opt = texture.create_trainable(kd_init , FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])

            ksR = np.random.uniform(size=FLAGS.texture_res + [1], low=0.0, high=0.01)
            ksG = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[1].cpu(), high=ks_max[1].cpu())
            ksB = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[2].cpu(), high=ks_max[2].cpu())

            ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])
        else:
            kd_map_opt = texture.create_trainable(init_mat['kd'], FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])
            ks_map_opt = texture.create_trainable(init_mat['ks'], FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])

        # Setup normal map
        if FLAGS.random_textures or init_mat is None or 'normal' not in init_mat:
            normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])
        else:
            normal_map_opt = texture.create_trainable(init_mat['normal'], FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])

        mat = material.Material({
            'kd'     : kd_map_opt,
            'ks'     : ks_map_opt,
            'normal' : normal_map_opt
        })

    if init_mat is not None:
        mat['bsdf'] = init_mat['bsdf']
    else:
        mat['bsdf'] = bsdf

    return mat

###############################################################################
# Validation & testing
###############################################################################

def validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS, use_normal=False):
    result_dict = {}
    with torch.no_grad():
        lgt.build_mips()
        if FLAGS.camera_space_light:
            lgt.xfm(target['mv'])

        buffers = geometry.render(glctx, target, lgt, opt_material)

        if use_normal:
            result_dict['opt'] = util.rgb_to_srgb(buffers['normal'][...,0:3])[0]
        else:
            result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][...,0:3])[0]
        result_image = result_dict['opt']

        if FLAGS.display is not None:
            for layer in FLAGS.display:
                if 'latlong' in layer and layer['latlong']:
                    if isinstance(lgt, light.EnvironmentLight):
                        result_dict['light_image'] = util.cubemap_to_latlong(lgt.base, FLAGS.display_res)
                    result_image = torch.cat([result_image, result_dict['light_image']], axis=1)
                elif 'relight' in layer:
                    if not isinstance(layer['relight'], light.EnvironmentLight):
                        layer['relight'] = light.load_env(layer['relight'])
                    img = geometry.render(glctx, target, layer['relight'], opt_material)
                    result_dict['relight'] = util.rgb_to_srgb(img[..., 0:3])[0]
                    result_image = torch.cat([result_image, result_dict['relight']], axis=1)
                elif 'bsdf' in layer:
                    buffers = geometry.render(glctx, target, lgt, opt_material, bsdf=layer['bsdf'])
                    if layer['bsdf'] == 'kd':
                        result_dict[layer['bsdf']] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])  
                    elif layer['bsdf'] == 'normal':
                        result_dict[layer['bsdf']] = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                    else:
                        result_dict[layer['bsdf']] = buffers['shaded'][0, ..., 0:3]
                    result_image = torch.cat([result_image, result_dict[layer['bsdf']]], axis=1)
   
        return result_image, result_dict

def validate(glctx, geometry, opt_material, lgt, dataset_validate, guidance_model, text_z, out_dir, FLAGS):

    # ==============================================================================================
    #  Validation loop
    # ==============================================================================================
    loss_values = []

    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_validate.collate)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as fout:
        fout.write('ID, MSE, PSNR\n')

        print("Running validation")
        for it, target in enumerate(dataloader_validate):

            # Mix validation background
            target = prepare_batch(target, FLAGS.background)

            result_image, result_dict = validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS)
           
            # Compute metrics
            opt = torch.clamp(result_dict['opt'], 0.0, 1.0) # [H, W, 3]

            pred_rgb = opt.permute(2, 0, 1).contiguous().unsqueeze(0) # [1, 3, H, W]
            
            loss = 0

            loss_values.append(loss)

            line = "%d, %1.8f\n" % (it, loss)
            fout.write(str(line))

            for k in result_dict.keys():
                np_img = result_dict[k].detach().cpu().numpy()
                util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)

        avg_loss = np.mean(np.array(loss_values))
        
        line = "AVERAGES: %1.4f\n" % (avg_loss,)
        fout.write(str(line))

    return avg_loss

###############################################################################
# Main shape fitter function / optimization loop
###############################################################################

class Trainer(torch.nn.Module):
    def __init__(self, glctx, geometry, lgt, mat, optimize_geometry, optimize_light, image_loss_fn, guidance_model, text_z, FLAGS):
        super(Trainer, self).__init__()

        self.glctx = glctx
        self.geometry = geometry
        self.light = lgt
        self.material = mat
        self.optimize_geometry = optimize_geometry # assert true
        self.optimize_light = optimize_light
        self.image_loss_fn = image_loss_fn
        self.guidance_model = guidance_model
        self.text_z = text_z
        self.FLAGS = FLAGS

        if not self.optimize_light:
            with torch.no_grad():
                self.light.build_mips()

        self.params = list(self.material.parameters())
        self.params += list(self.light.parameters()) if optimize_light else []
        self.geo_params = list(self.geometry.parameters())

    def forward(self, target, it):
        if self.optimize_light:
            self.light.build_mips()
            if self.FLAGS.camera_space_light:
                self.light.xfm(target['mv'])

        return self.geometry.tick(glctx, target, self.light, self.material, self.image_loss_fn, self.guidance_model, self.text_z, it)
    
    # save and load checkpoints
    def save(self, it):
        state = {
            'geometry': self.geometry.state_dict(),
            'material': self.material.state_dict(),
            'light': self.light.state_dict() if self.optimize_light else None,
        }
        os.makedirs(os.path.join(self.FLAGS.out_dir, "checkpoints"), exist_ok=True)
        print(f'[INFO] saving checkpoints to {it}.pth')
        torch.save(state, os.path.join(self.FLAGS.out_dir, "checkpoints", f"{it}.pth"))
    
    def load(self, it=None):
        if it is None:
            candidates = [int(os.path.basename(x).split(".")[0]) for x in glob.glob(os.path.join(self.FLAGS.out_dir, "checkpoints", "*.pth"))]
            if len(candidates) == 0: 
                print(f'[INFO] cannot find checkpoints to load')
                return 0
            it = np.max(candidates)
        print(f'[INFO] loading checkpoints from {it}.pth')
        state = torch.load(os.path.join(self.FLAGS.out_dir, "checkpoints", f"{it}.pth"))
        self.material.load_state_dict(state['material'])
        self.geometry.load_state_dict(state['geometry'])
        if state['light'] is not None:
            self.light.load_state_dict(state['light'])

        return it

def optimize_mesh(
    glctx,
    geometry,
    opt_material,
    lgt,
    dataset_train,
    dataset_validate,
    guidance_model, text_z,
    FLAGS,
    warmup_iter=0,
    log_interval=10,
    pass_idx=0,
    pass_name="",
    optimize_light=True,
    optimize_geometry=True
    ):

    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================

    learning_rate = FLAGS.learning_rate[pass_idx] if isinstance(FLAGS.learning_rate, list) or isinstance(FLAGS.learning_rate, tuple) else FLAGS.learning_rate
    learning_rate_pos = learning_rate[0] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate
    learning_rate_mat = learning_rate[1] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate

    def lr_schedule(iter, fraction):
        if iter < int(FLAGS.iter * 0.2):
            return 1 # 1e-3 geom init
        elif iter < int(FLAGS.iter * 0.6):
            return 0.1 # 1e-4 geom fine
        else:
            return 10  # 1e-2 material
    
        # if iter < warmup_iter:
        #     return iter / warmup_iter 
        # return max(0.0, 10**(-(iter - warmup_iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.    

    # ==============================================================================================
    #  Image loss
    # ==============================================================================================
    image_loss_fn = createLoss(FLAGS)

    trainer_noddp = Trainer(glctx, geometry, lgt, opt_material, optimize_geometry, optimize_light, image_loss_fn, guidance_model, text_z, FLAGS)

    # load latest model
    load_it = trainer_noddp.load()

    if FLAGS.multi_gpu: 
        # Multi GPU training mode
        from torch.nn.parallel import DistributedDataParallel as DDP

        trainer = DDP(trainer_noddp, device_ids=[FLAGS.local_rank], output_device=FLAGS.local_rank, find_unused_parameters=True)
        trainer.train()
    
        optimizer_mesh = torch.optim.AdamW(trainer_noddp.geo_params, lr=learning_rate_pos)
        scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

        optimizer = torch.optim.AdamW(trainer_noddp.params, lr=learning_rate_mat)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9)) 
    else:
        # Single GPU training mode
        trainer = trainer_noddp
    
        optimizer_mesh = torch.optim.AdamW(trainer_noddp.geo_params, lr=learning_rate_pos)
        scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

        optimizer = torch.optim.AdamW(trainer_noddp.params, lr=learning_rate_mat)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    img_loss_vec = []
    reg_loss_vec = []
    iter_dur_vec = []
    v_it = 0

    dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=FLAGS.batch, collate_fn=dataset_train.collate, shuffle=True)
    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_train.collate)
    v_it = itertools.cycle(dataloader_validate)

    for it, target in enumerate(dataloader_train):

        # load checkpoints
        it += load_it
        if it > FLAGS.iter:
            break

        # Mix randomized background into dataset image
        target = prepare_batch(target, 'random')

        # ==============================================================================================
        #  Display / save outputs. Do it before training so we get initial meshes
        # ==============================================================================================

        # Show/save image before training step (want to get correct rendering of input)
        if FLAGS.local_rank == 0:
            display_image = FLAGS.display_interval and (it % FLAGS.display_interval == 0)
            save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)
            if display_image or save_image:
                result_image, result_dict = validate_itr(glctx, prepare_batch(next(v_it), FLAGS.background), geometry, opt_material, lgt, FLAGS, use_normal=(it < int(FLAGS.iter * 0.6)))
                np_result_image = result_image.detach().cpu().numpy()
                if display_image:
                    util.display_image(np_result_image, title='%d / %d' % (it, FLAGS.iter))
                if save_image:
                    img_cnt = it // FLAGS.save_interval
                    util.save_image(FLAGS.out_dir + '/' + ('img_%s_%06d.png' % (pass_name, img_cnt)), np_result_image)

        iter_start_time = time.time()

        # ==============================================================================================
        #  Zero gradients
        # ==============================================================================================
        optimizer.zero_grad()
        optimizer_mesh.zero_grad()

        # ==============================================================================================
        #  Training
        # ==============================================================================================
        img_loss, reg_loss = trainer(target, it)

        # ==============================================================================================
        #  Final loss
        # ==============================================================================================
        total_loss = img_loss + reg_loss

        img_loss_vec.append(img_loss.item())
        reg_loss_vec.append(reg_loss.item())

        # ==============================================================================================
        #  Backpropagate
        # ==============================================================================================
        total_loss.backward()
        if hasattr(lgt, 'base') and lgt.base.grad is not None and optimize_light:
            lgt.base.grad *= 64
        
        # if 'kd_ks_normal' in opt_material:
        #     opt_material['kd_ks_normal'].encoder.embeddings.grad /= 8.0
            # opt_material['kd_ks_normal'].encoder.params.grad /= 8.0

        # optimize materials only after finishing geometry
        if it < int(FLAGS.iter * 0.6):
            optimizer_mesh.step()
            scheduler_mesh.step()
        else:
            optimizer.step()
            scheduler.step()


        # ==============================================================================================
        #  Clamp trainables to reasonable range
        # ==============================================================================================
        with torch.no_grad():
            if 'kd' in opt_material:
                opt_material['kd'].clamp_()
            if 'ks' in opt_material:
                opt_material['ks'].clamp_()
            if 'normal' in opt_material:
                opt_material['normal'].clamp_()
                opt_material['normal'].normalize_()
            if lgt is not None:
                lgt.clamp_(min=0.0)

        torch.cuda.current_stream().synchronize()
        iter_dur_vec.append(time.time() - iter_start_time)

        # ==============================================================================================
        #  Logging
        # ==============================================================================================
        if it % log_interval == 0 and FLAGS.local_rank == 0:
            img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
            reg_loss_avg = np.mean(np.asarray(reg_loss_vec[-log_interval:]))
            iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))
            
            remaining_time = (FLAGS.iter-it)*iter_dur_avg
            print("iter=%5d, img_loss=%.6f, reg_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s" % 
                (it, img_loss_avg, reg_loss_avg, optimizer.param_groups[0]['lr'], iter_dur_avg*1000, util.time_to_text(remaining_time)))
        
        if it > 0 and it % 500 == 0 and FLAGS.local_rank == 0:
            # save model
            trainer_noddp.save(it)
    

    return geometry, opt_material

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=100)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001)
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-dt', '--directional-text', action='store_true', default=False)
    parser.add_argument('-bg', '--background', default='white', choices=['black', 'white', 'checker', 'reference'])
    parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relmse'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('-rm', '--ref_mesh', type=str)
    parser.add_argument('-bm', '--base-mesh', type=str, default=None)
    parser.add_argument('--validate', type=bool, default=True)
    
    FLAGS = parser.parse_args()

    FLAGS.mtl_override        = None                     # Override material of model
    FLAGS.dmtet_grid          = 128                      # Resolution of initial tet grid. We provide 64 and 128 resolution grids. Other resolutions can be generated with https://github.com/crawforddoran/quartet
    FLAGS.mesh_scale          = 2.1                      # Scale of tet grid box. Adjust to cover the model
    FLAGS.env_scale           = 1.0                      # Env map intensity multiplier
    FLAGS.envmap              = None                     # HDR environment probe
    FLAGS.display             = None                     # Conf validation window/display. E.g. [{"relight" : <path to envlight>}]
    FLAGS.camera_space_light  = False                    # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
    FLAGS.lock_light          = False                    # Disable light optimization in the second pass
    FLAGS.lock_pos            = False                    # Disable vertex position optimization in the second pass
    FLAGS.sdf_regularizer     = 0.2                      # Weight for sdf regularizer (see paper for details)
    FLAGS.laplace             = "absolute"               # Mesh Laplacian ["absolute", "relative", "large_steps"]
    FLAGS.laplace_scale       = 10000                  # Weight for sdf regularizer. Default is relative with large weight
    # FLAGS.normal_scale        = 0.02                  # Weight for sdf regularizer. Default is relative with large weight
    FLAGS.pre_load            = True                     # Pre-load entire dataset into memory for faster training
    FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
    FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    FLAGS.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    FLAGS.ks_max              = [ 1.0,  1.0,  1.0]
    FLAGS.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
    FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
    FLAGS.cam_near_far        = [0.1, 1000.0]
    FLAGS.learn_light         = True

    FLAGS.local_rank = 0
    FLAGS.multi_gpu  = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

    if FLAGS.multi_gpu:

        # adjust total iters
        FLAGS.iter = int(FLAGS.iter / int(os.environ["WORLD_SIZE"]))

        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = 'localhost'
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = '23456'

        FLAGS.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(FLAGS.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]

    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res
    if FLAGS.out_dir is None:
        FLAGS.out_dir = 'out/cube_%d' % (FLAGS.train_res)
    else:
        FLAGS.out_dir = 'out/' + FLAGS.out_dir

    if FLAGS.local_rank == 0:
        print("Config / Flags:")
        print("---------")
        for key in FLAGS.__dict__.keys():
            print(key, FLAGS.__dict__[key])
        print("---------")

    os.makedirs(FLAGS.out_dir, exist_ok=True)

    glctx = dr.RasterizeGLContext()

    # ==============================================================================================
    #  Create CLIP
    # ==============================================================================================

    guidance_model = StableDiffusion('cuda', fp16=True, vram_O=False)

    guidance_model.eval()
    for p in guidance_model.parameters():
        p.requires_grad = False

    if FLAGS.directional_text:    
        text_z = []
        for d in ['front', 'side', 'back', 'side']:
            # construct dir-encoded text
            text_z.append(guidance_model.get_text_embeds([f"{FLAGS.text}, {d} view"], [''], 1))
        text_z = torch.stack(text_z, dim=0)
    else:
        text_z = guidance_model.get_text_embeds([f"{FLAGS.text}"], [''], FLAGS.batch)

    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    
    dataset_train    = DatasetDream(glctx, RADIUS, FLAGS, validate=False)
    dataset_validate = DatasetDream(glctx, RADIUS, FLAGS, validate=True)

    # ==============================================================================================
    #  Create env light with trainable parameters
    # ==============================================================================================
    
    if FLAGS.learn_light:
        lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=0.5)
    else:
        lgt = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)

    # ==============================================================================================
    #  always use DMtets to create geometry
    # ==============================================================================================

    # Setup geometry for optimization
    geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)

    # Setup textures, make initial guess from reference if possible
    mat = initial_guess_material(geometry, True, FLAGS)

    # Run optimization
    geometry, mat = optimize_mesh(glctx, geometry, mat, lgt, dataset_train, dataset_validate, guidance_model, text_z,
                    FLAGS, pass_idx=0, pass_name="dmtet_pass1", optimize_light=FLAGS.learn_light)

    if FLAGS.local_rank == 0 and FLAGS.validate:
        validate(glctx, geometry, mat, lgt, dataset_validate, guidance_model, text_z, os.path.join(FLAGS.out_dir, "dmtet_validate"), FLAGS)

    # Create textured mesh from result
    base_mesh = xatlas_uvmap(glctx, geometry, mat, FLAGS)

    # Free temporaries / cached memory 
    torch.cuda.empty_cache()
    mat['kd_ks_normal'].cleanup()
    del mat['kd_ks_normal']

    lgt = lgt.clone()
    geometry = DLMesh(base_mesh, FLAGS)

    if FLAGS.local_rank == 0:
        # Dump mesh for debugging.
        os.makedirs(os.path.join(FLAGS.out_dir, "dmtet_mesh"), exist_ok=True)
        obj.write_obj(os.path.join(FLAGS.out_dir, "dmtet_mesh/"), base_mesh)
        light.save_env_map(os.path.join(FLAGS.out_dir, "dmtet_mesh/probe.hdr"), lgt)

    # # ==============================================================================================
    # #  Pass 2: Train with fixed topology (mesh)
    # # ==============================================================================================
    # geometry, mat = optimize_mesh(glctx, geometry, base_mesh.material, lgt, dataset_train, dataset_validate, guidance_model, text_z, 
    #             FLAGS, pass_idx=1, pass_name="mesh_pass", warmup_iter=100, optimize_light=FLAGS.learn_light and not FLAGS.lock_light, 
    #             optimize_geometry=not FLAGS.lock_pos)

    # # ==============================================================================================
    # #  Validate
    # # ==============================================================================================
    # if FLAGS.validate and FLAGS.local_rank == 0:
    #     validate(glctx, geometry, mat, lgt, dataset_validate, guidance_model, text_z, os.path.join(FLAGS.out_dir, "validate"), FLAGS)

    # # ==============================================================================================
    # #  Dump output
    # # ==============================================================================================
    # if FLAGS.local_rank == 0:
    #     final_mesh = geometry.getMesh(mat)
    #     os.makedirs(os.path.join(FLAGS.out_dir, "mesh"), exist_ok=True)
    #     obj.write_obj(os.path.join(FLAGS.out_dir, "mesh/"), final_mesh)
    #     light.save_env_map(os.path.join(FLAGS.out_dir, "mesh/probe.hdr"), lgt)

#----------------------------------------------------------------------------
