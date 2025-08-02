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
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from ops.diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
# import roma
from utils.graphics_utils import flip_align_view,get_view_dir
# from brdf import *
# from nvdiffrec.render import util
# import jtvision, os


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : jt.Var, scaling_modifier = 1.0, override_color = None, test = False, color_diff = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pyjt return gradients of the 2D (screen-space) means
    screenspace_points = pc.screenspace_points

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python or pc.deform_flag:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

        # rotmat = roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(rotations))
        # if pc.flatten:
        #     normals = rotmat[:,:,-1]
        #     # normals = rotmat
        # else:
        #     index = jt.argmin(scales,1)
        #     normals = jt.gather(rotmat,2,index.unsqueeze(1).unsqueeze(1).expand(-1, 3, -1))

        # dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_xyz.shape[0], 1))
        # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        # normals,_ = flip_align_view(normals.squeeze(),dir_pp_normalized.detach())

    # sio.savemat('normal.mat',{'normal':normals.detach().cpu().numpy(),'point':means3D.detach().cpu().numpy()})
    # cc()

    # depths = (means3D - viewpoint_camera.camera_center).pow(2).sum(-1).sqrt()

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    # colors_precomp = None

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python or pc.deform_flag:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            if pc.deform_flag:
                 dir_pp_normalized = jt.matmul(pc.deform_rot.transpose(1,2),dir_pp_normalized.unsqueeze(2)).squeeze(2)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = jt.clamp(sh2rgb + 0.5, min_v = 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # view_dir = get_view_dir(viewpoint_camera).view(800,800,3) - viewpoint_camera.camera_center
    # view_dir = viewpoint_camera.camera_center.repeat(640000, 1).reshape(800,800,3)
    # view_dir = view_dir/view_dir.norm(dim=2, keepdim=True)

    # color, radii, is_visible,out_depth, out_normal,out_depth_index,out_alpha = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp,normals = normals,depth = depths,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     u = rotmat[:,:,0].squeeze(),v= rotmat[:,:,1].squeeze(),
    #     delta_normal = pc.get_delta_normal,
    #     gt_image = color_diff,
    #     ray_dirs = None
    # )
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )



    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
            "render": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
    # return {"render": color,
    #         "alpha": out_alpha,
    #         "viewspace_points": screenspace_points,
    #         "visibility_filter" : radii > 0,
    #         "is_visible": is_visible,
    #         "radii": radii,"depth": out_depth, "normal":out_normal,'depth_index':out_depth_index}
