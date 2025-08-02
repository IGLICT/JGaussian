import os
import math
import numpy as np
from typing import NamedTuple
from plyfile import PlyData, PlyElement

import jittor as jt
from jittor import nn

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))  
sys.path.append(project_root)

from ops.diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def nan_to_num(input, nan=0.0, posinf=None, neginf=None):
    # 处理 NaN
    output = jt.where(jt.isnan(input), jt.array(nan, dtype=input.dtype), input)
    # 处理正无穷 (如果没有指定 posinf，则用最大的有限值代替)
    if posinf is None:
        posinf = jt.finfo(input.dtype).max
    output = jt.where(jt.isposinf(output), jt.array(posinf, dtype=input.dtype), output)
    
    # 处理负无穷 (如果没有指定 neginf，则用最小的有限值代替)
    if neginf is None:
        neginf = jt.finfo(input.dtype).min
    output = jt.where(jt.isneginf(output), jt.array(neginf, dtype=input.dtype), output)
    return output


def inverse_sigmoid(x):
    return jt.log(x/(1-x))

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    
    def helper(step):
        if lr_init == lr_final:
            # constant lr, ignore other params
            return lr_init
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    # uncertainty = jt.zeros((L.shape[0], 6), dtype=jt.float, device="cuda")
    uncertainty = jt.zeros((L.shape[0], 6), dtype=jt.float, device=L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # eps must be small enough !!!
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

    power[power > 0] = -1e10 # abnormal values... make weights 0
        
    return jt.exp(power)

def build_rotation(r):
    norm = jt.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    # R = jt.zeros((q.size(0), 3, 3), device='cuda')
    R = jt.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    # L = jt.zeros((s.shape[0], 3, 3), dtype=jt.float, device="cuda")
    L = jt.zeros((s.shape[0], 3, 3), dtype=jt.float, device=r.device)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm

def build_covariance_from_scaling_rotation_cov(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = jt.zeros(4, 4)

    z_sign = 1.0
    
    if tanHalfFovX == 0 or tanHalfFovY == 0:
        # dummy inputs are given
        P[0, 0] = 1
        P[1, 1] = 1
    else:
        P[0, 0] = 1 / tanHalfFovX
        P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, device=None):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        try:
            w2c = jt.linalg.inv(c2w)
        except:
            w2c = c2w.clone() # when dummy c2w are given
        
        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = w2c.clone().transpose(0, 1)
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = - c2w[:3, 3].clone()

        if device is not None:
            self.world_view_transform = self.world_view_transform
            self.projection_matrix = self.projection_matrix
            self.full_proj_transform = self.full_proj_transform
            self.camera_center = self.camera_center
    def logg(self):
        print("camera")      
        print(self.world_view_transform) 
        print(self.projection_matrix) 
        print(self.full_proj_transform)
        print(self.camera_center)


class Renderer:
    def __init__(self, sh_degree=3, white_background=False, radius=1):
        
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius

        # self.gaussians = GaussianModel(sh_degree)

        self.bg_color = jt.array(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=jt.float32,
        )
                
        self.scaling_activation = jt.exp
        self.scaling_inverse_activation = jt.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = jt.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = jt.normalize
        
        # init param?
        self.active_sh_degree = 0
        self.max_sh_degree = 0
        
    def get_scaling(self, _scaling):
        return self.scaling_activation(_scaling)
    
    def get_rotation(self, _rotation):
        return self.rotation_activation(_rotation)
    
    def get_xyz(self):
        return self._xyz
    
    def get_features(self, features_dc, features_rest):
        return jt.cat((features_dc, features_rest), dim=1)
    
    def get_opacity(self, _opacity):
        return self.opacity_activation(_opacity)
    
    
    
    def render(
        self,
        gaussian_params,
        viewpoint_camera,
        scaling_modifier=1.0,
        invert_bg_color=False,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        random_background=True,
    ):
        
        _xyz = gaussian_params["_xyz"]
        _features_dc = gaussian_params["_features_dc"]
        _features_rest = gaussian_params["_features_rest"]
        _scaling = gaussian_params["_scaling"]
        _rotation = gaussian_params["_rotation"]
        _opacity = gaussian_params["_opacity"]

        # Create zero tensor. We will use it to make pyjt return gradients of the 2D (screen-space) means
        screenspace_points = (
            jt.zeros_like(
                # self.gaussians.get_xyz,
                # dtype=self.gaussians.get_xyz.dtype,
                _xyz,
                dtype=_xyz.dtype,
            )
            + 0
        )
        # try:
        #     screenspace_points.retain_grad()
        # except:
        #     pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        bg = self.bg_color if not invert_bg_color else 1 - self.bg_color
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=0.5*bg,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
            # debug=True,
        )
        
        # print("check")
        # print("tanfovx",tanfovx)
        # print("tanfovy",tanfovy)
        # print("bg",bg)
        # print("self.active_sh_degree",self.active_sh_degree)
        # print("viewpoint_camera.world_view_transform",viewpoint_camera.world_view_transform)
        # print("viewpoint_camera.full_proj_transform",viewpoint_camera.full_proj_transform)
        # print("scaling_modifier",scaling_modifier)
        # print("int(viewpoint_camera.image_height)",int(viewpoint_camera.image_height))
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        rasterizer = rasterizer
        

        # means3D = self.gaussians.get_xyz
        means3D = _xyz
        means2D = screenspace_points
        # opacity = self.gaussians.get_opacity
        opacity = self.get_opacity(_opacity)
        

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            # cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
            cov3D_precomp = self.get_covariance(scaling_modifier)
        else:
            # scales = self.gaussians.get_scaling
            # rotations = self.gaussians.get_rotation
            scales = self.get_scaling(_scaling)
            rotations = self.get_rotation(_rotation)
            
        # check nan
        # print(scales)
        # if jt.isnan(scales).sum() > 0.:
        #     # print(jt.isnan(_scaling).sum(), _scaling[jt.where(jt.isnan(_scaling))])
        #     print("# nans in {}".format(scales.device), jt.isnan(scales).sum().item())
        # # scales[jt.where(jt.isnan(scales))] = 0.0
        scales = nan_to_num(scales)
        
        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                pass
                # shs_view = self.gaussians.get_features.transpose(1, 2).view(
                #     -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                # )
                # dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                #     self.gaussians.get_features.shape[0], 1
                # )
                # dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                # sh2rgb = eval_sh(
                #     self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                # )
                # colors_precomp = jt.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                # shs = self.gaussians.get_features
                shs = self.get_features(_features_dc, _features_rest)
        else:
            colors_precomp = override_color
        # print("check")
        # print("means3D",means3D)
        # print("means3D",means2D)
        # print("shs",shs)
        # print("colors",colors_precomp)
        # print("opacity",opacity)
        # print("scales",scales)
        # print("rot",rotations)
        # print("cov3D",cov3D_precomp)

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        # with jt.autocast(device_type=_xyz.device.type, dtype=jt.float32):
            # rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                means3D=means3D.float(),
                means2D=means2D.float(),
                shs=shs.float(),
                colors_precomp=colors_precomp,
                opacities=opacity.float(),
                scales=scales.float(),
                rotations=rotations.float(),
                cov3D_precomp=cov3D_precomp,
            )

        

        rendered_image = rendered_image / 0.5 - 1.
        zero_idx = jt.where(rendered_depth == 0, 1., 0.)
        rendered_depth = rendered_depth * (1. - zero_idx) + jt.ones_like(rendered_depth) * jt.max(rendered_depth) * zero_idx

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
