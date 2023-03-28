# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import torch

from render import mesh
from render import render
from render import regularizer

from encoding import get_encoder

import tqdm
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x

def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1,2,0).squeeze()
        x = x.detach().cpu().numpy()
        
    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()

###############################################################################
# Marching tetrahedrons implementation (differentiable), adapted from
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
###############################################################################

class DMTet:
    def __init__(self):
        self.triangle_table = torch.tensor([
                [-1, -1, -1, -1, -1, -1],
                [ 1,  0,  2, -1, -1, -1],
                [ 4,  0,  3, -1, -1, -1],
                [ 1,  4,  2,  1,  3,  4],
                [ 3,  1,  5, -1, -1, -1],
                [ 2,  3,  0,  2,  5,  3],
                [ 1,  4,  0,  1,  5,  4],
                [ 4,  2,  5, -1, -1, -1],
                [ 4,  5,  2, -1, -1, -1],
                [ 4,  1,  0,  4,  5,  1],
                [ 3,  2,  0,  3,  5,  2],
                [ 1,  3,  5, -1, -1, -1],
                [ 4,  1,  2,  4,  3,  1],
                [ 3,  0,  4, -1, -1, -1],
                [ 2,  0,  1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1]
                ], dtype=torch.long, device='cuda')

        self.num_triangles_table = torch.tensor([0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0], dtype=torch.long, device='cuda')
        self.base_tet_edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype=torch.long, device='cuda')

    ###############################################################################
    # Utility functions
    ###############################################################################

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:,0] > edges_ex2[:,1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)      
            b = torch.gather(input=edges_ex2, index=1-order, dim=1)  

        return torch.stack([a, b],-1)

    def map_uv(self, faces, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx+1)//2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            ) # indexing='ij')

        pad = 0.9 / N

        uvs = torch.stack([
            tex_x      , tex_y,
            tex_x + pad, tex_y,
            tex_x + pad, tex_y + pad,
            tex_x      , tex_y + pad
        ], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='trunc')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((
            tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
        ), dim = -1). view(-1, 3)

        return uvs, uv_idx

    ###############################################################################
    # Marching tets implementation
    ###############################################################################

    def __call__(self, pos_nx3, sdf_n, tet_fx4):
        # pos_nx3: [N, 3]
        # sdf_n:   [N]
        # tet_fx4: [F, 4]

        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4)
            occ_sum = torch.sum(occ_fx4, -1) # [F,]
            valid_tets = (occ_sum>0) & (occ_sum<4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:,self.base_tet_edges].reshape(-1,2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges,dim=0, return_inverse=True)  
            
            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long,device="cuda")
            idx_map = mapping[idx_map] # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_sdf[:,-1] *= -1

        denominator = edges_to_interp_sdf.sum(1,keepdim = True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1,6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda"))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
        ), dim=0)

        # Get global face index (static, does not depend on topology)
        num_tets = tet_fx4.shape[0]
        tet_gidx = torch.arange(num_tets, dtype=torch.long, device="cuda")[valid_tets]
        face_gidx = torch.cat((
            tet_gidx[num_triangles == 1]*2,
            torch.stack((tet_gidx[num_triangles == 2]*2, tet_gidx[num_triangles == 2]*2 + 1), dim=-1).view(-1)
        ), dim=0)

        uvs, uv_idx = self.map_uv(faces, face_gidx, num_tets*2)

        return verts, faces, uvs, uv_idx

###############################################################################
# Regularizer
###############################################################################

def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff

###############################################################################
#  Geometry interface
###############################################################################


class DMTetGeometry(torch.nn.Module):
    def __init__(self, grid_res, scale, FLAGS):
        super(DMTetGeometry, self).__init__()

        self.FLAGS         = FLAGS
        self.grid_res      = grid_res
        self.marching_tets = DMTet()


        tets = np.load('data/tets/{}_tets.npz'.format(self.grid_res))
        self.verts    = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * scale # for 64/128, [N=36562/277410, 3], in [-0.5, 0.5]^3
        self.indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda') # for 64/128, [M=192492/1524684, 4], vert indices for each tetrahetron?
        self.generate_edges()

        self.encoder, self.in_dim = get_encoder('hashgrid', input_dim=3)
        self.mlp = MLP(self.in_dim, 4, 32, 3, False)
        self.bg_mlp = MLP(3, 3, 32, 2, False)

        self.encoder.cuda()
        self.mlp.cuda()
        self.bg_mlp.cuda()

        # init sdf from base mesh
        if FLAGS.base_mesh is not None:

            print(f'[INFO] init sdf from base mesh: {FLAGS.base_mesh}')

            import cubvh, trimesh
            mesh = trimesh.load(FLAGS.base_mesh, force='mesh')

            scale = 1 / np.array(mesh.bounds[1] - mesh.bounds[0]).max()
            center = np.array(mesh.bounds[1] + mesh.bounds[0]) / 2
            mesh.vertices = (mesh.vertices - center) * scale
            
            BVH = cubvh.cuBVH(mesh.vertices, mesh.faces) # build with numpy.ndarray/torch.Tensor
            sdf, face_id, _ = BVH.signed_distance(self.verts, return_uvw=False, mode='watertight')
            sdf *= -1 # INNER is POSITIVE
            
            # pretraining 
            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-3)
            
            pretrain_iters = 10000
            batch_size = 10240
            print(f"[INFO] start SDF pre-training ")
            for i in tqdm.tqdm(range(pretrain_iters)):
                rand_idx = torch.randint(0, self.verts.shape[0], (batch_size,))
                p = self.verts[rand_idx]
                ref_value = sdf[rand_idx]
                output = self.mlp(self.encoder(p))
                loss = loss_fn(output[...,0], ref_value)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if i % 100 == 0:
                #     print(f"[INFO] SDF pre-train: {loss.item()}")
                
            print(f"[INFO] SDF pre-train final loss: {loss.item()}")

            # visualize 
            # sdf_np_gt = sdf.cpu().numpy()
            # sdf_np = self.mlp(self.encoder(self.verts)).detach().cpu().numpy()[..., 0]
            # verts_np = self.verts.cpu().numpy()
            # color = np.zeros_like(verts_np)
            # color[sdf_np < 0] = [1, 0, 0]
            # color[sdf_np > 0] = [0, 0, 1]
            # color = (color * 255).astype(np.uint8)
            # pc = trimesh.PointCloud(verts_np, color)
            # axes = trimesh.creation.axis(axis_length=4)
            # box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
            # trimesh.Scene([mesh, pc, axes, box]).show()

            del mesh, BVH

       
    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0,1, 0,2, 0,3, 1,2, 1,3, 2,3], dtype = torch.long, device = "cuda") # six edges for each tetrahedron.
            all_edges = self.indices[:,edges].reshape(-1,2) # [M * 6, 2]
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    def getMesh(self, material):

        pred = self.mlp(self.encoder(self.verts)) # predict SDF and per-vertex deformation
        sdf, deform = pred[:, 0], pred[:, 1:]

        v_deformed = self.verts + 1 / (self.grid_res) * torch.tanh(deform)

        verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, sdf, self.indices)

        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)

        return imesh, sdf

    def render(self, glctx, target, lgt, opt_material, bsdf=None):
        
        # return rendered buffers, keys: ['shaded', 'kd_grad', 'occlusion'].
        opt_mesh, sdf = self.getMesh(opt_material)
        buffers = render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
                                        msaa=True, background=None, bsdf=bsdf)
        buffers['mesh'] = opt_mesh
        buffers['sdf'] = sdf

        # background layer
        bg_color = torch.sigmoid(self.bg_mlp(target['rays_d']))
        B, H, W = buffers['shaded'].shape[:3]
        buffers['bg_color'] = bg_color.view(B, H, W, -1)

        return buffers


    def tick(self, glctx, target, lgt, opt_material, loss_fn, guidance_model, text_z, iteration):

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers = self.render(glctx, target, lgt, opt_material)

        mesh = buffers['mesh']

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter

        if iteration < int(self.FLAGS.iter * 0.2):
            # mode = 'normal_latent'
            pred_rgb = buffers['normal'][..., 0:4].permute(0, 3, 1, 2).contiguous()
            as_latent = True
        elif iteration < int(self.FLAGS.iter * 0.6):
            # mode = 'normal'
            pred_rgb = buffers['normal'][..., 0:3].permute(0, 3, 1, 2).contiguous()
            as_latent = False
        else:
            # mode = 'rgb'
            pred_rgb = buffers['shaded'][..., 0:3].permute(0, 3, 1, 2).contiguous()
            pred_ws = buffers['shaded'][..., 3].unsqueeze(1) # [B, 1, H, W]
            bg_color = buffers['bg_color'].permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            pred_rgb = pred_rgb * pred_ws + (1 - pred_ws) * bg_color
            as_latent = False

        # torch_vis_2d(bg_color[0])
        # torch_vis_2d(pred_rgb[0])
        # torch_vis_2d(pred_normal[0])
        # torch_vis_2d(pred_ws[0])
        
        img_loss = guidance_model.train_step(text_z, pred_rgb.half(), as_latent=as_latent)

        # img_loss = torch.tensor(0.0, device = "cuda")

        # below are lots of regularizations...
        reg_loss = torch.tensor(0.0, device = "cuda")

        # SDF regularizer
        sdf_weight = self.FLAGS.sdf_regularizer - (self.FLAGS.sdf_regularizer - 0.01) * min(1.0, 4.0 * t_iter)
        sdf_loss = sdf_reg_loss(buffers['sdf'], self.all_edges).mean() * sdf_weight # Dropoff to 0.01
        reg_loss = reg_loss + sdf_loss

        # directly regularize mesh smoothness
        lap_loss = regularizer.laplace_regularizer_const(mesh.v_pos, mesh.t_pos_idx) * self.FLAGS.laplace_scale * (1 - t_iter)
        reg_loss = reg_loss + lap_loss

        # print(lap_loss, sdf_loss)
        # reg_loss += regularizer.normal_consistency(mesh.v_pos, mesh.t_pos_idx) * self.FLAGS.laplace_scale * (1 - t_iter)
        
        # # Albedo (k_d) smoothnesss regularizer
        # reg_loss += torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, iteration / 500)

        # # Visibility regularizer
        # reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 0.001 * min(1.0, iteration / 500)

        # # Light white balance regularizer
        # reg_loss = reg_loss + lgt.regularizer() * 0.005

        return img_loss, reg_loss