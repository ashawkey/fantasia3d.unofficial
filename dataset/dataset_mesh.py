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

from render import util
from render import mesh
from render import render
from render import light

from .dataset import Dataset

###############################################################################
# Reference dataset using mesh & rendering
###############################################################################

class DatasetDream(Dataset):

    def __init__(self, glctx, cam_radius, FLAGS, validate=False):
        # Init 
        self.glctx              = glctx
        self.cam_radius         = cam_radius
        self.FLAGS              = FLAGS
        self.validate           = validate
        self.fovy               = np.deg2rad(45)
        self.aspect             = FLAGS.train_res[1] / FLAGS.train_res[0]

    
    def get_rays(self, mv):
        i, j = torch.meshgrid(torch.arange(self.FLAGS.train_res[1], device=mv.device), torch.arange(self.FLAGS.train_res[0], device=mv.device)) # float
        i = i.t().contiguous().view(-1) + 0.5
        j = j.t().contiguous().view(-1) + 0.5
        zs = -torch.ones_like(i) # z is flipped
        focal = 0.5 * self.FLAGS.train_res[1] / np.tan(0.5 * self.fovy)
        xs = (i - self.FLAGS.train_res[1] / 2) / focal
        ys = -(j - self.FLAGS.train_res[0] / 2) / focal # y is flipped
        directions = torch.stack((xs, ys, zs), dim=-1) # [N, 3]
        rays_d = (directions.unsqueeze(1) @ mv[:, :3, :3]).squeeze(1) 
        return rays_d


    def _rotate_scene(self, itr):
        self.fov = np.deg2rad(45)
        proj_mtx = util.perspective(self.fovy, self.FLAGS.display_res[1] / self.FLAGS.display_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Smooth rotation for display.
        ang    = (itr / 50) * np.pi * 2
        mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(-0.4) @ util.rotate_y(ang))
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), self.FLAGS.display_res, self.FLAGS.spp

    def _random_scene(self):
        # ==============================================================================================
        #  Setup projection matrix
        # ==============================================================================================
        iter_res = self.FLAGS.train_res
        self.fov = np.random.uniform(np.pi/7, np.pi/4)
        proj_mtx = util.perspective(self.fovy, iter_res[1] / iter_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # ==============================================================================================
        #  Random camera & light position
        # ==============================================================================================

        # Random rotation/translation matrix for optimization.

        # mv     = util.translate(0, 0, -self.cam_radius) @ util.random_rotation()
        angle_x = np.random.uniform(-np.pi/4, np.pi/18)
        angle_y = np.random.uniform(0, 2 * np.pi)
        mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(angle_x) @ util.rotate_y(angle_y))
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), iter_res, self.FLAGS.spp # Add batch dimension

    def __len__(self):
        return 50 if self.validate else (self.FLAGS.iter + 1) * self.FLAGS.batch

    def __getitem__(self, itr):
        # ==============================================================================================
        #  Randomize scene parameters
        # ==============================================================================================

        if self.validate:
            mv, mvp, campos, iter_res, iter_spp = self._rotate_scene(itr)
        else:
            mv, mvp, campos, iter_res, iter_spp = self._random_scene()

        rays_d = self.get_rays(mv).unsqueeze(0) # [1, N, 3]

        return {
            'mv' : mv, # [1, 4, 4], world2cam
            'mvp' : mvp, # [1, 4, 4], world2cam + projection
            'campos' : campos, # [1, 3], camera's position in world coord
            'resolution' : iter_res, # [2], training res, e.g., [512, 512]
            'spp' : iter_spp, # [1], mostly == 1
            'rays_d': rays_d,
        }
