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

    
    def _rotate_scene(self, itr):
        self.fov = np.deg2rad(45)
        proj_mtx = util.perspective(self.fovy, self.FLAGS.display_res[1] / self.FLAGS.display_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Smooth rotation for display.
        angle_y    = (itr / 50) * np.pi * 2

        # direction
        # 0 = front, 1 = side, 2 = back, 3 = side
        if angle_y >= 0 and angle_y <= np.pi/8 or angle_y > 15*np.pi/8:
            direction = 0
        elif angle_y > np.pi/8 and angle_y <= 7*np.pi/8:
            direction = 1
        elif angle_y > 7*np.pi/8 and angle_y <= 9*np.pi/8:
            direction = 2
        elif angle_y > 9*np.pi/8 and angle_y <= 15*np.pi/8:
            direction = 3

        mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(-0.4) @ util.rotate_y(angle_y))
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), self.FLAGS.display_res, self.FLAGS.spp, direction

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

        # direction
        # 0 = front, 1 = side, 2 = back, 3 = side
        if angle_y >= 0 and angle_y <= np.pi/8 or angle_y > 15*np.pi/8:
            direction = 0
        elif angle_y > np.pi/8 and angle_y <= 7*np.pi/8:
            direction = 1
        elif angle_y > 7*np.pi/8 and angle_y <= 9*np.pi/8:
            direction = 2
        elif angle_y > 9*np.pi/8 and angle_y <= 15*np.pi/8:
            direction = 3

        mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(angle_x) @ util.rotate_y(angle_y))
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), iter_res, self.FLAGS.spp, direction # Add batch dimension

    def __len__(self):
        return 50 if self.validate else (self.FLAGS.iter + 1) * self.FLAGS.batch

    def __getitem__(self, itr):
        # ==============================================================================================
        #  Randomize scene parameters
        # ==============================================================================================

        if self.validate:
            mv, mvp, campos, iter_res, iter_spp, direction = self._rotate_scene(itr)
        else:
            mv, mvp, campos, iter_res, iter_spp, direction = self._random_scene()

 
        return {
            'mv' : mv, # [1, 4, 4], world2cam
            'mvp' : mvp, # [1, 4, 4], world2cam + projection
            'campos' : campos, # [1, 3], camera's position in world coord
            'resolution' : iter_res, # [2], training res, e.g., [512, 512]
            'spp' : iter_spp, # [1], mostly == 1
            'direction' : direction,
        }
