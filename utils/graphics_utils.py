#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import jittor as jt
import math
import numpy as np
from typing import NamedTuple
from jittor.einops import rearrange




class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = jt.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = jt.cat([points, ones], dim=1)
    points_out = jt.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = jt.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def get_view_dir(view):
    cx = view.image_height/2
    cy = view.image_width/2
    focal_length_x = fov2focal(view.FoVx, view.image_height)
    focal_length_y = fov2focal(view.FoVy, view.image_width)
    
    w2c = np.eye(4).astype(float)
    w2c[:3, :3] = view.R.transpose()
    w2c[:3, 3] = view.T
    pose = np.linalg.inv(w2c)
    pose = jt.array(pose).float()
    
    # u, v = jt.meshgrid(jt.arange(view.image_height), jt.arange(view.image_width), indexing='ij')
    u, v = jt.meshgrid(jt.arange(view.image_height), jt.arange(view.image_width))
    u = u.float()
    v = v.float()

    uvd = jt.stack([v, u], dim=-1)

    x = (uvd[..., 0] - cx) / focal_length_x
    y = (uvd[..., 1] - cy) / focal_length_y
    z = jt.ones_like(v)
    pc_c = jt.stack([x, y, z], dim=-1)
    pc_c = pc_c.reshape((-1, 3))
    # pc_w = (pose[:3, :3] @ pc_c.T).T + pose[:3, 3].reshape(-1, 3)
    pc_w = (pose[:3, :3] @ pc_c.transpose(-1,-2)).transpose(-1,-2) + pose[:3, 3].reshape(-1, 3)
    return pc_w


# def get_world_space_normal_from_depth_cuda(depth, view, is_distance_map=True, return_pc = False):
#     # save normal
#     depth = depth.squeeze()
    
#     cx = view.image_height/2
#     cy = view.image_width/2
#     focal_length_x = fov2focal(view.FoVx, view.image_height)
#     focal_length_y = fov2focal(view.FoVy, view.image_width)
    
#     w2c = np.eye(4).astype(float)
#     w2c[:3, :3] = view.R.transpose()
#     w2c[:3, 3] = view.T
#     pose = np.linalg.inv(w2c)
#     pose = jt.from_numpy(pose).float().cuda()
    
#     u, v = jt.meshgrid(jt.arange(depth.shape[0]), jt.arange(depth.shape[1]), indexing='ij')
#     u = u.float().cuda()
#     v = v.float().cuda()
#     uvd = jt.stack([v, u, depth], dim=-1)

#     x_over_z = (uvd[..., 0] - cx) / focal_length_x
#     y_over_z = (uvd[..., 1] - cy) / focal_length_y
#     if is_distance_map:
#         z = depth / jt.sqrt(1. + x_over_z**2 + y_over_z**2)
#     else:
#         z = depth
#     x = x_over_z * z
#     y = y_over_z * z
#     pc_c = jt.stack([x, y, z], dim=-1)
    
#     # pc_c = jt.tensor([convert_from_uvd(w, h, depth[h, w], cx=cx, cy=cy, focalx=focal_length_x, focaly=focal_length_y, is_distance_map=is_distance_map) for h in range(depth.shape[0]) for w in range(depth.shape[1])]).float().cuda()
#     pc_c = pc_c.reshape((-1, 3))
#     # pc_c = transform_matrix[:3, :3].T @ (pc_w - transform_matrix[:, 3])
#     # this is neus world coordinate
#     pc_w = (pose[:3, :3] @ pc_c.T).T + pose[:3, 3].reshape(-1, 3)
#     if return_pc:
#         return pc_w
    
#     # save_obj('depth_pc.obj',pc_w,jt.empty(0))
#     # save_ply('depth_pc.ply',pc_w)
    
#     new_depth = pc_c.reshape((depth.shape[0], depth.shape[1], 3))
#     normal_image = get_normal_map_by_point_cloud(new_depth)

#     normal_image[..., [0]] = -1 * normal_image[..., [0]]
#     normal_image[..., [1]] = -1 * normal_image[..., [1]]
#     normal_image[..., [2]] = -1 * normal_image[..., [2]]

#     normal_image_w = (pose[:3, :3] @ normal_image.reshape(-1, 3).T).T 
#     normal_image_w = normal_image_w / (jt.linalg.norm(normal_image_w, axis=-1, keepdims=True) + 1e-10)
#     normal_image_w[jt.isnan(normal_image_w)] = 0
#     normal_image_w = normal_image_w.reshape(normal_image.shape)
#     return normal_image_w





# def depth_to_normals(depth_map,view, return_pc=False):
#     normal = get_world_space_normal_from_depth_cuda(depth_map,view,return_pc=return_pc)
#     return normal



def flip_align_view(normal, viewdir):
    # normal: (N, 3), viewdir: (N, 3)
    dotprod = jt.sum(
        normal * -viewdir, dim=-1, keepdims=True) # (N, 1)
    non_flip = dotprod>=0 # (N, 1)
    normal_flipped = normal*jt.where(non_flip, 1, -1) # (N, 3)
    return normal_flipped, non_flip


# depth_map = jt.rand((1,1, 256, 256)) 
# K = jt.eye(3)[None]
# normal = depth_to_normals(depth_map,K)
# jtvision.utils.save_image(depth_map, 'depth.png')
# jtvision.utils.save_image(normal, 'normal.png')

