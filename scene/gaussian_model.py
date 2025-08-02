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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from jittor import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from ops.simple_knn import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.gsmesh_utils import split_mesh_and_gaussian, split_mesh_and_gaussian_pro, get_barycentric_coordinate
import trimesh
import math
# from gaussian_renderer.diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import pyACAP

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, is_sym = True):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            if is_sym:
                symm = strip_symmetric(actual_covariance)
            else:
                symm = actual_covariance
            return symm
        
        self.scaling_activation = jt.exp
        self.scaling_inverse_activation = jt.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = jt.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = jt.normalize

        if self.is_gsmesh:
            self.bc_activation = jt.nn.softmax
            self.distance_activation = jt.sigmoid

        # self.diffuse_activation = jt.sigmoid
        # self.specular_activation = jt.sigmoid
        # self.roughness_activation = jt.sigmoid

        # self.envmap_activation = jt.sigmoid

        # self.delta_normal_activation = jt.sigmoid

        # self.rotation_activation = jt.nn.functional.normalize


    def __init__(self, sh_degree : int, mesh_path = ""):
        self.mesh_path = mesh_path
        if os.path.exists(mesh_path):
            self.is_gsmesh = True
        else:
            self.is_gsmesh = False
        
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = jt.empty(0)
        self._features_dc = jt.empty(0)
        self._features_rest = jt.empty(0)
        self._scaling = jt.empty(0)
        self._rotation = jt.empty(0)
        self._opacity = jt.empty(0)
        self.max_radii2D = jt.empty(0)
        self.xyz_gradient_accum = jt.empty(0)
        self.denom = jt.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        if self.is_gsmesh:
            self.alpha_distance = 4     #hyperparam
            self._bc = jt.empty(0) #N,3
            self._distance = jt.empty(0) # N,1
            self.vertex_index = jt.empty(0) # N,3
            self.vertex1 = jt.empty(0) #N,3 coordinate
            self.vertex2 = jt.empty(0)
            self.vertex3 = jt.empty(0)
            self.fid = jt.empty(0) #N,1 index of origin mesh(not split)
            self.normal = jt.empty(0)  # N,3 
            self.r = jt.empty(0) #N,1 restrict gaussian not too far offset the face
        self.deform_flag = False

        self.setup_functions()


    def capture(self):
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)   

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        if self.is_gsmesh:
            if not self.deform_flag:
                bc = self.bc_activation(self._bc,dim=1)
                proj_xyz = jt.unsqueeze(bc[:,0],1)*self.vertex1 + jt.unsqueeze(bc[:,1],1)*self.vertex2 + jt.unsqueeze(bc[:,2],1)*self.vertex3
                offset = self.alpha_distance*self.r*(self.distance_activation(self._distance)-0.5)*self.normal
                xyz = proj_xyz + offset
                return xyz
            else:
                return self.deform_pos
        else:
            return self._xyz

    @property
    def get_proj_xyz(self):
        if self.is_gsmesh:
            bc = self.bc_activation(self._bc,dim=1)
            n = bc.shape[0]
            proj_xyz = jt.unsqueeze(bc[:,0],1)*self.vertex1 + jt.unsqueeze(bc[:,1],1)*self.vertex2 + jt.unsqueeze(bc[:,2],1)*self.vertex3
            return proj_xyz
        else:
            return self._xyz
        
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return jt.concat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1, is_sym = True):
        if not self.deform_flag:
            return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation, is_sym)
        else:
            return strip_symmetric(self.deform_cov)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def apply_ct(self, color_ct: np.ndarray):
        sh_data = self.get_features.numpy() # (641723, 16, 3)
        cnt = sh_data.shape[0]
        sh_data = sh_data.reshape((-1, 3))
        color_ct = color_ct.astype(np.float32)
        sh_data = (sh_data @ color_ct[:3, :3].T + color_ct[:3, 3][np.newaxis, :]).astype(np.float32) # (10267568, 3) 
        sh_data = jt.array(sh_data.reshape(cnt, -1, 3))
        self._features_dc = sh_data[:, 0:1, :]
        self._features_rest = sh_data[:, 1:, :]

    def create_from_mesh(self,file = None):
        if(file is None):
            print("load gaussian from mesh:",self.mesh_path)
            mesh = trimesh.load(self.mesh_path)
        else:
            mesh = trimesh.load(file)
        vertex = np.array(mesh.vertices)  
        triangles = np.array(mesh.faces)
        num_pts = triangles.shape[0]    
        face_normals = np.array(mesh.face_normals)    
        fused_point_cloud = (jt.ones((num_pts,3), dtype = jt.float))/3
        distance = (jt.zeros((num_pts,1), dtype = jt.float))
        fused_color = RGB2SH(jt.array(np.random.random((num_pts, 3)), dtype = jt.float))
        features = jt.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype = jt.float)
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        self.vertex1 = jt.array(vertex[triangles[:,0]], dtype = jt.float).stop_grad()
        self.vertex2 = jt.array(vertex[triangles[:,1]], dtype = jt.float).stop_grad()
        self.vertex3 = jt.array(vertex[triangles[:,2]], dtype = jt.float).stop_grad()
        a = jt.unsqueeze(jt.norm((self.vertex1 - self.vertex2),dim=1),1)
        b = jt.unsqueeze(jt.norm((self.vertex2 - self.vertex3),dim=1),1)
        c = jt.unsqueeze(jt.norm((self.vertex3 - self.vertex1),dim=1),1)
        
        self.r = (a+b+c)/3
        self.fid = jt.unsqueeze(jt.arange(num_pts),1).stop_grad()
        
        self.normal = jt.array(face_normals, dtype = jt.float).stop_grad()
        self.vertex_index = jt.array(triangles).stop_grad()
        self.v = jt.array(vertex).stop_grad()

        p_tmp = (self.vertex1+self.vertex2+self.vertex3)/3
        dist2 = jt.clamp(distCUDA2(p_tmp), min_v=0.0000001)
        scales = jt.log(jt.sqrt(dist2))[...,None].repeat(1, 3)
        rots = jt.zeros((fused_point_cloud.shape[0], 4))
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * jt.ones((fused_point_cloud.shape[0], 1), dtype=jt.float32))

        ## jittor init
        self._bc = fused_point_cloud.clone()
        self._distance = distance.clone()
        self._features_dc = features[:,:,0:1].transpose(1, 2).clone()
        self._features_rest = features[:,:,1:].transpose(1, 2).clone()
        self._scaling = scales.clone()
        self._rotation = rots.clone()
        self._opacity = opacities.clone()
        self.max_radii2D = jt.zeros((self.get_opacity.shape[0]))

       

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = jt.array(np.asarray(pcd.points)).float()
        fused_color = RGB2SH(jt.array(np.asarray(pcd.colors)).float())
        features = jt.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # dist2 = jt.clamp_min(distCUDA2(jt.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # scales = jt.log(jt.sqrt(dist2))[...,None].repeat(1, 3)
        # rots = jt.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        # rots[:, 0] = 1
        # opacities = inverse_sigmoid(0.1 * jt.ones((fused_point_cloud.shape[0], 1), dtype=jt.float, device="cuda"))
        dist2 = jt.clamp(distCUDA2(jt.array(np.asarray(pcd.points)).float()), min_v=0.0000001)
        scales = jt.log(jt.sqrt(dist2))[...,None].repeat(1, 3)
        rots = jt.zeros((fused_point_cloud.shape[0], 4))
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * jt.ones((fused_point_cloud.shape[0], 1), dtype=jt.float32))
        self._xyz = fused_point_cloud.clone()
        self._features_dc = features[:,:,0:1].transpose(1, 2).clone()
        self._features_rest = features[:,:,1:].transpose(1, 2).clone()
        self._scaling = scales.clone()
        self._rotation = rots.clone()
        self._opacity = opacities.clone()
        self.max_radii2D = jt.zeros((self.get_opacity.shape[0]))

    def training_setup(self, training_args, init_max_gaussian = 50000):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = jt.zeros((self.get_opacity.shape[0], 1))
        self.xyz_gradient_accum_abs = jt.zeros((self.get_opacity.shape[0], 1))
        self.denom = jt.zeros((self.get_opacity.shape[0], 1))
        if self.is_gsmesh:
            l = [
            {'params': [self._bc], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "bc"},
            {'params': [self._distance], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "distance"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        else:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]
        # for dict in l:
        #     print(dict["name"])
        #     print(dict['params'][0].shape)
        self.optimizer = jt.nn.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        #init for GSMESH
        if self.is_gsmesh:
            while(self.get_opacity.shape[0] <= init_max_gaussian):
                print("split init!!,less gaussian than",init_max_gaussian)
                self.densify_and_split_for_init()
    

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in ["xyz", "bc", "distance"]:
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # print(lr)
                return lr

    def reset_viewspace_point(self):
        # self.screenspace_points = jt.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype) + 0
        self.screenspace_points = jt.zeros_like(self.get_scaling, dtype=self.get_xyz.dtype) + 0
        pg = self.optimizer.param_groups[-1]
        if pg["name"] == "screenspace_points":
            self.optimizer.param_groups.pop()
        self.optimizer.add_param_group(
            {'params': [self.screenspace_points], 'lr':0., "name": "screenspace_points"}
        )
    def get_viewspace_point_grad(self):
        pg = self.optimizer.param_groups[-1]
        if pg["name"] == "screenspace_points":
            # breakpoint()
            return pg["grads"][0]
        else:
            assert False, "No viewspace_point_grad found"

    def construct_list_of_attributes(self):
        if self.is_gsmesh:
            l = ['x', 'y', 'z', 'nx', 'ny', 'nz','ca', 'cb', 'cc','v1x','v1y','v1z','v2x','v2y','v2z','v3x','v3y','v3z','dis','v_index1','v_index2','v_index3','radius','face_id']
        else:
            l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    # ToDo
    def save_ply_color(self,path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().numpy()
        f_dc = jt.clamp(0.28209479177387814 * self._features_dc+ 0.5, 0, 1).detach().transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
        color = (f_dc * 255).astype(np.uint8)
        dtype_full = [('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, color), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_mesh(self, dst):
        import shutil
        shutil.copy(self.mesh_path, dst)
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.get_xyz.detach().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).numpy()
        opacities = self._opacity.detach().numpy()
        scale = self._scaling.detach().numpy()
        rotation = self._rotation.detach().numpy()
        normals = np.zeros_like(xyz)

        if self.is_gsmesh:
            bc = self._bc.detach().numpy()
            v1 = self.vertex1.detach().numpy()
            v2 = self.vertex2.detach().numpy()
            v3 = self.vertex3.detach().numpy()
            radius = self.r.detach().numpy()
            fid = self.fid.detach().numpy()
            distance = self._distance.detach().numpy()
            normals = self.normal.detach().numpy()
            v_index = self.vertex_index.detach().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if self.is_gsmesh:
            attributes = np.concatenate((xyz, normals, bc, v1, v2, v3, distance, v_index, radius, fid, f_dc, f_rest, opacities, scale, rotation), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        self.load_xyz = self.get_xyz.detach().numpy()

    def reset_opacity(self):
        def min(a,b):
            return jt.where(a<b,a,b)
        opacities_new = inverse_sigmoid(min(self.get_opacity, jt.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, reset_basis_dim=0):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = jt.array(xyz, dtype=jt.float)
        self._features_dc = jt.array(features_dc, dtype=jt.float).transpose(1, 2)
        self._features_rest = jt.array(features_extra, dtype=jt.float).transpose(1, 2)
        self._opacity = jt.array(opacities, dtype=jt.float)
        self._scaling = jt.array(scales, dtype=jt.float)
        self._rotation = jt.array(rots, dtype=jt.float)


        if self.is_gsmesh:
            bc = np.stack((np.asarray(plydata.elements[0]["ca"]),
                        np.asarray(plydata.elements[0]["cb"]),
                        np.asarray(plydata.elements[0]["cc"])),  axis=1)
            v1 = np.stack((np.asarray(plydata.elements[0]["v1x"]),
                            np.asarray(plydata.elements[0]["v1y"]),
                            np.asarray(plydata.elements[0]["v1z"])),  axis=1)
            v2 = np.stack((np.asarray(plydata.elements[0]["v2x"]),
                            np.asarray(plydata.elements[0]["v2y"]),
                            np.asarray(plydata.elements[0]["v2z"])),  axis=1)
            v3 = np.stack((np.asarray(plydata.elements[0]["v3x"]),
                            np.asarray(plydata.elements[0]["v3y"]),
                            np.asarray(plydata.elements[0]["v3z"])),  axis=1)
            normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
                            np.asarray(plydata.elements[0]["ny"]),
                            np.asarray(plydata.elements[0]["nz"])),  axis=1)
            fid = np.asarray(plydata.elements[0]["face_id"])[..., np.newaxis]

            vertex_index = np.stack((np.asarray(plydata.elements[0]["v_index1"]),
                            np.asarray(plydata.elements[0]["v_index2"]),
                            np.asarray(plydata.elements[0]["v_index3"])),  axis=1)
            
            distance = np.asarray(plydata.elements[0]["dis"])[..., np.newaxis]
            radius = np.asarray(plydata.elements[0]["radius"])[..., np.newaxis]
            self._bc = jt.array(xyz, dtype=jt.float)
            self._distance = jt.array(distance, dtype=jt.float)
            self.vertex1 = jt.array(v1, dtype=jt.float).stop_grad()
            self.vertex2 = jt.array(v2, dtype=jt.float).stop_grad()
            self.vertex3 = jt.array(v3, dtype=jt.float).stop_grad()
            self.normal = jt.array(normal, dtype=jt.float).stop_grad()
            self.r = jt.array(radius, dtype=jt.float).stop_grad()
            self.fid = jt.array(fid, dtype=jt.int).stop_grad()
            self.vertex_index = jt.array(vertex_index, dtype=jt.float).stop_grad()

            #for deform
            self.origin_pos = self.get_xyz
            self.origin_cov = self.get_covariance(is_sym = False)
            self.origin_sh = self.get_features
            self.deform_rot = jt.init.eye(3).unsqueeze(0).expand(self.get_xyz.shape[0], -1, -1).float()
            self.deform_cov = self.origin_cov.clone()
            self.deform_pos = self.origin_pos.clone()
        
            #binding mesh and gaussian
            mesh = trimesh.load(self.mesh_path)
            vertex = np.array(mesh.vertices)  
            triangles = np.array(mesh.faces)
            self.gaussian_triangles = triangles[self.fid.squeeze(1)]
            gaussian_intersection = self.get_proj_xyz
            gaussian_triangles_p1 = vertex[self.gaussian_triangles[:,0]]
            gaussian_triangles_p2 = vertex[self.gaussian_triangles[:,1]]
            gaussian_triangles_p3 = vertex[self.gaussian_triangles[:,2]]
            self.coord = get_barycentric_coordinate(gaussian_intersection,gaussian_triangles_p1,
                                                gaussian_triangles_p2,gaussian_triangles_p3)
            self.weight_g_pos = jt.array(np.expand_dims(self.coord, axis=2))
            self.weight_g_rs = jt.array(np.expand_dims(self.coord, axis=(2, 3)))
            self.vertex = jt.array(vertex)
            self.ACAPtool = pyACAP.pyACAP(self.mesh_path) 

        self.screenspace_points = jt.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype) + 0
        self.active_sh_degree = self.max_sh_degree

        return xyz, opacities, scales

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == 'screenspace_points': 
                continue
            if group["name"] == name:
                with jt.enable_grad():
                    group["params"][0] = tensor.copy()
                group["m"][0] = jt.zeros_like(tensor)
                group["values"][0] = jt.zeros_like(tensor)
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == 'screenspace_points': 
                continue
            if group['params'][0] is not None:

                group['m'][0].update(group['m'][0][mask])
                group['values'][0].update(group['values'][0][mask])
                with jt.enable_grad():
                    old = group["params"].pop()
                    group["params"].append(old[mask])
                    del old
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = mask.logical_not()
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        if self.is_gsmesh:
            self._bc = optimizable_tensors["bc"]
            self._distance = optimizable_tensors["distance"]
            self.vertex1 = self.vertex1[valid_points_mask]
            self.vertex2 = self.vertex2[valid_points_mask]
            self.vertex3 = self.vertex3[valid_points_mask]
            self.r = self.r[valid_points_mask]
            self.fid = self.fid[valid_points_mask]
            self.normal = self.normal[valid_points_mask]
            self.vertex_index = self.vertex_index[valid_points_mask]
        else:
            self._xyz = optimizable_tensors["xyz"]

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] == 'screenspace_points': 
                continue
            extension_tensor = tensors_dict[group["name"]]
            group["m"][0] = jt.concat((group["m"][0], jt.zeros_like(extension_tensor)), dim=0)
            
            group["values"][0] = jt.concat((group["values"][0], jt.zeros_like(extension_tensor)), dim=0)
            old_tensor = group["params"].pop()
            with jt.enable_grad():
                group["params"].append(jt.concat((old_tensor, extension_tensor), dim=0))
                del old_tensor
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_distance = None):
        if self.is_gsmesh:
            d = {"bc": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling" : new_scaling,
            "rotation" : new_rotation,
            "distance" : new_distance}
        else:
            d = {"xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling" : new_scaling,
            "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        if self.is_gsmesh:
            self._bc = optimizable_tensors["bc"]
            self._distance = optimizable_tensors["distance"]
        else:
            self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = jt.zeros((self.get_opacity.shape[0], 1))
        self.denom = jt.zeros((self.get_opacity.shape[0], 1))
        self.max_radii2D = jt.zeros((self.get_opacity.shape[0]))

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, selected_pts_mask = None):
        if self.is_gsmesh:
            raise Exception("GSMESH CANNOT USE THIS WAY TO SPLIT")
        n_init_points = self.get_opacity.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = jt.zeros((n_init_points))
        padded_grad[:grads.shape[0]] = grads.squeeze()
        # print("check",grads.squeeze())
        if selected_pts_mask is None:
            selected_pts_mask = jt.where(padded_grad >= grad_threshold, True, False)
            # selected_pts_mask = jt.logical_or(selected_pts_mask, selected_pts_mask_abs)
            selected_pts_mask = jt.logical_and(selected_pts_mask,
                                                jt.max(self.get_scaling, dim=1) > self.percent_dense*scene_extent)
        else:
            padded_mask = jt.zeros((n_init_points))
            padded_mask[:selected_pts_mask.shape[0]] = selected_pts_mask.squeeze()
            selected_pts_mask = jt.logical_and(padded_mask,
                                                jt.max(self.get_scaling, dim=1) > self.percent_dense*scene_extent)
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =jt.zeros((stds.size(0), 3))
        samples = jt.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = jt.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        if(selected_pts_mask.sum()==0):
            prune_filter = selected_pts_mask
        else:
            prune_filter = jt.cat((selected_pts_mask.bool(), jt.zeros((N * selected_pts_mask.sum()[0].item()))))
        self._prune_points(prune_filter)

    def densify_and_split_gsmesh(self, grads, grad_threshold, scene_extent, N=2, selected_pts_mask = None):
        if not self.is_gsmesh:
            raise Exception("ONLY GSMESH USE THIS WAY TO SPLIT")
        n_init_points = self.get_opacity.shape[0]
        # print("Number of points : ", n_init_points)
        # Extract points that satisfy the gradient condition
        padded_grad = jt.zeros((n_init_points))
        padded_grad[:grads.shape[0]] = grads.squeeze()
        if selected_pts_mask is None:
            selected_pts_mask = jt.where(padded_grad >= grad_threshold, True, False)
        else:
            padded_mask = jt.zeros((n_init_points))
            padded_mask[:selected_pts_mask.shape[0]] = selected_pts_mask.squeeze()
        padded_grad = jt.zeros((n_init_points))
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = jt.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = jt.logical_and(selected_pts_mask,
                                              jt.max(self.get_scaling, dim=1) > self.percent_dense*scene_extent)
        
        if(selected_pts_mask.sum().item()==0):
            return

        bc = self._bc[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(-1,3)   
        new_bc = jt.ones_like(bc)/3
        distance = self._distance[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(-1,1)
        new_distance = jt.zeros_like(distance) 
        gaussian_num = new_bc.shape[0]
        split_num = self._bc[selected_pts_mask].shape[0]
        #new index
        new_v_index = self.vertex_index[selected_pts_mask].unsqueeze(1).repeat(1,N,1)
        new_v = jt.zeros(split_num,3).unsqueeze(1).repeat(1,3,1)
        new_vertex1 = self.vertex1[selected_pts_mask].unsqueeze(1).repeat(1,N,1)
        new_vertex2 = self.vertex2[selected_pts_mask].unsqueeze(1).repeat(1,N,1)
        new_vertex3 = self.vertex3[selected_pts_mask].unsqueeze(1).repeat(1,N,1)
        new_r = self.r[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num,-1)
        new_fid = self.fid[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num,-1)
        if(N==4):
            new_vertex1,new_vertex2,new_vertex3,new_v,new_v_index = \
            split_mesh_and_gaussian(new_vertex1,new_vertex2,new_vertex3,new_v,new_v_index,self.v.shape[0])
        elif(N==5):
            new_vertex1,new_vertex2,new_vertex3,new_v,new_v_index = \
            split_mesh_and_gaussian_pro(new_vertex1,new_vertex2,new_vertex3,new_v,new_v_index,self.v.shape[0])

        new_vertex1 = new_vertex1.view(gaussian_num, 3)
        new_vertex2 = new_vertex2.view(gaussian_num, 3)
        new_vertex3 = new_vertex3.view(gaussian_num, 3)
        new_v = new_v.view(-1, 3)
        new_v_index = new_v_index.view(gaussian_num, 3)
            
        new_normal = self.normal[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num, 3)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num,3) / (4*0.8))
        new_rotation = self._rotation[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num,4)
            
        new_features_dc = self._features_dc[selected_pts_mask].unsqueeze(1).repeat(1,N,1,1).view(gaussian_num,-1,3)
        new_features_rest = self._features_rest[selected_pts_mask].unsqueeze(1).repeat(1,N,1,1).view(gaussian_num,-1,3)
        new_opacity = self._opacity[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num,-1)

        self.densification_postfix(new_bc, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_distance)

        self.vertex1 = jt.concat((self.vertex1, new_vertex1), dim=0)
        self.vertex2 = jt.concat((self.vertex2, new_vertex2), dim=0)
        self.vertex3 = jt.concat((self.vertex3, new_vertex3), dim=0)
        self.vertex_index = jt.concat((self.vertex_index, new_v_index), dim=0)
        self.r = jt.concat((self.r, new_r), dim=0)
        self.v = jt.concat((self.v, new_v), dim=0)
        self.normal = jt.concat((self.normal, new_normal),dim=0)
        self.fid = jt.concat((self.fid, new_fid),dim=0)
        
        prune_filter = jt.concat((selected_pts_mask, jt.zeros(N * selected_pts_mask.sum().item(), dtype=bool)))
        self.prune_points(prune_filter)


    def densify_and_clone(self, grads, grad_threshold, scene_extent, selected_pts_mask = None):
        # Extract points that satisfy the gradient condition
        if selected_pts_mask is None:
            selected_pts_mask = jt.where(jt.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = jt.logical_and(selected_pts_mask,
                                              jt.max(self.get_scaling, dim=1) <= self.percent_dense*scene_extent) #self.percent_dense*scene_extent

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        

    def densify_and_prune_gsmesh(self, max_grad, min_opacity, extent, max_screen_size, selected_pts_mask = None, N = 4):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_split_gsmesh(grads, max_grad, extent, N)
        jt.gc()       

    def densify_and_prune_basic(self, max_grad, min_opacity, extent, max_screen_size, selected_pts_mask = None,color_mask = None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent,selected_pts_mask = selected_pts_mask)
        self.densify_and_split(grads, max_grad, extent,selected_pts_mask = selected_pts_mask)

        prune_mask = self.prune_points_new(min_opacity, extent, None,color_mask = color_mask)
    
    def prune_points_new(self,min_opacity, extent, max_screen_size,min_size = 1e-3, color_mask = None):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        small_points = (self.get_scaling.sum(dim=1) < min_size).squeeze()
        prune_mask = jt.logical_or(prune_mask, small_points)

        if color_mask is not None:
            prune_mask = jt.logical_or(prune_mask, color_mask)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1) > 0.1 * extent
            prune_mask = jt.logical_or(jt.logical_or(prune_mask, big_points_vs), big_points_ws)
        self._prune_points(prune_mask)
        jt.gc()
        return prune_mask
    
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, selected_pts_mask = None,color_mask = None, N = 4):
        if self.is_gsmesh:
            self.densify_and_prune_gsmesh(max_grad, min_opacity, extent, max_screen_size, selected_pts_mask, N)
        else:
            self.densify_and_prune_basic(max_grad, min_opacity, extent, max_screen_size, selected_pts_mask, color_mask)

    def add_densification_stats(self,  viewspace_point_tensor_grad, update_filter):
        self.xyz_gradient_accum[update_filter] += jt.norm(viewspace_point_tensor_grad[update_filter,:2], dim=-1, keepdim=True)
        # self.xyz_gradient_accum_abs[update_filter] += jt.norm(viewspace_point_tensor_grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def apply_weights(self, camera, image_weights):
        rasterizer = self.camera2rasterizer(
            camera, jt.array([0.0, 0.0, 0.0], dtype=jt.float32)
        )
        weights,weights_cnt=rasterizer.apply_weights(
            self.get_xyz,
            None,
            self.get_opacity,
            self.get_features,
            None,
            self.get_scaling,
            self.get_rotation,
            None,
            image_weights,
        )
        return weights,weights_cnt
    
    def camera2rasterizer(self, viewpoint_camera, bg_color: jt.Var, sh_degree: int = 0):
        from ops.diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        return rasterizer

    def _prune_points(self, mask):
        valid_points_mask = mask.logical_not()
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def prune_over_sized_points(self, size_limit, all_dir:bool=True):
        if all_dir:
            # all direction > size_limit, then prune
            print('scale shape',self.get_scaling.shape, 'scale max',self.get_scaling.abs().max(dim=1).max(), 'scale min',self.get_scaling.abs().min(dim=1).min()) # mean 0.0794
            prune_mask = self.get_scaling.abs().max(dim=1) > size_limit
        else:
            # one direction > size_limit, then prune
            prune_mask = self.get_scaling.abs(dim=1).min(dim=1) > size_limit
        print("before scale prune", self.get_scaling.shape)
        # self.xyz_gradient_accum = jt.zeros((self.get_opacity.shape[0], 1))
        # self.denom = jt.zeros((self.get_opacity.shape[0], 1))
        # self.max_radii2D = jt.zeros((self.get_opacity.shape[0]))
        self.prune_points(prune_mask)
        print("after scale prune", self.get_opacity.shape)
        jt.gc()


    def prune_over_opacity_points(self, max_opacity, min_opacity, all_dir:bool=True):
        if all_dir:
            # all direction > size_limit, then prune
            print('opacity shape',self.get_scaling.shape, 'opacity max',self.get_opacity.abs().max(dim=1).max(),'opacity min',self.get_opacity.abs().min(dim=1).min()) # mean 0.0794
            selected_pts_mask = jt.logical_and(self.get_opacity.abs().max(dim=1) < max_opacity, self.get_opacity.abs().max(dim=1) > min_opacity)
        else:
            # one direction > size_limit, then prune
            selected_pts_mask = jt.logical_or(self.get_opacity.abs(dim=1).max(dim=1) < max_opacity & self.get_opacity.abs(dim=1).min(dim=1) > min_opacity)
        print("before opacity prune", self.get_opacity.shape)
        # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        # self.xyz_gradient_accum = jt.zeros((self.get_opacity.shape[0], 1))
        # self.denom = jt.zeros((self.get_opacity.shape[0], 1))
        # self.max_radii2D = jt.zeros((self.get_opacity.shape[0]))

        self.prune_points(selected_pts_mask.logical_not())
        print("after opacity prune", self.get_opacity.shape)
        jt.gc()
       
    def densify_and_split_for_init(self, N=4):
        n_init_points = self.get_opacity.shape[0]
        print("after init the face number is:", n_init_points*N)

        # split all points
        selected_pts_mask = jt.ones((n_init_points),dtype=jt.bool)

        bc = self._bc[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(-1,3)   
        new_bc = jt.ones_like(bc)/3
        distance = self._distance[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(-1,1)
        new_distance = jt.zeros_like(distance) 
        gaussian_num = new_bc.shape[0]
        split_num = self._bc[selected_pts_mask].shape[0]
        #new index
        new_v_index = self.vertex_index[selected_pts_mask].unsqueeze(1).repeat(1,N,1)
        new_v = jt.zeros(split_num,3).unsqueeze(1).repeat(1,3,1)
        new_vertex1 = self.vertex1[selected_pts_mask].unsqueeze(1).repeat(1,N,1)
        new_vertex2 = self.vertex2[selected_pts_mask].unsqueeze(1).repeat(1,N,1)
        new_vertex3 = self.vertex3[selected_pts_mask].unsqueeze(1).repeat(1,N,1)
        new_fid = self.fid[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num,-1)
        new_r = self.r[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num,-1)
        new_vertex1,new_vertex2,new_vertex3,new_v,new_v_index = \
            split_mesh_and_gaussian(new_vertex1,new_vertex2,new_vertex3,new_v,new_v_index,self.v.shape[0])
       
        new_vertex1 = new_vertex1.view(gaussian_num, 3)
        new_vertex2 = new_vertex2.view(gaussian_num, 3)
        new_vertex3 = new_vertex3.view(gaussian_num, 3)
        new_v = new_v.view(-1, 3)
        new_v_index = new_v_index.view(gaussian_num, 3)
            
        new_normal = self.normal[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num, 3)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num,3) / (4*0.8))
        new_rotation = self._rotation[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num,4)
            
        new_features_dc = self._features_dc[selected_pts_mask].unsqueeze(1).repeat(1,N,1,1).view(gaussian_num,-1,3)
        new_features_rest = self._features_rest[selected_pts_mask].unsqueeze(1).repeat(1,N,1,1).view(gaussian_num,-1,3)
        new_opacity = self._opacity[selected_pts_mask].unsqueeze(1).repeat(1,N,1).view(gaussian_num,-1)

        self.densification_postfix(new_bc, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_distance)

        self.vertex1 = jt.concat((self.vertex1, new_vertex1), dim=0)
        self.vertex2 = jt.concat((self.vertex2, new_vertex2), dim=0)
        self.vertex3 = jt.concat((self.vertex3, new_vertex3), dim=0)
        self.vertex_index = jt.concat((self.vertex_index, new_v_index), dim=0)
        self.r = jt.concat((self.r, new_r), dim=0)
        self.v = jt.concat((self.v, new_v), dim=0)
        self.normal = jt.concat((self.normal, new_normal),dim=0)
        self.fid = jt.concat((self.fid, new_fid),dim=0)

        prune_filter = jt.concat((selected_pts_mask, jt.zeros(N * n_init_points, dtype=bool)))
        self.prune_points(prune_filter)
        jt.gc()

    def deform_gaussian_mesh(self, deform_mesh_path):
        print("use this function after func gaussians.load_ply")
            # self.origin_pos = self.get_xyz
            # self.origin_cov = self.get_covariance(is_sym = False)
            # self.origin_sh = self.get_features
            # self.deform_rot = jt.init.eye(3).unsqueeze(0).expand(self.get_xyz.shape[0], -1, -1).float()
            # self.deform_cov = self.origin_cov.clone()
            # self.deform_pos = self.origin_pos.clone()
        mesh = trimesh.load(deform_mesh_path)
        deform_vertex = np.array(mesh.vertices)  
        R1, S1 = self.ACAPtool.GetRS(self.vertex, deform_vertex, 1,  os.cpu_count()//2)
        cur_pos = jt.array(deform_vertex)
        cur_rot = jt.array(R1.reshape((-1, 3, 3)))
        cur_shear = jt.array(S1.reshape((-1, 3, 3)))

        # get deform pos
        delta_pos_ = (cur_pos - self.vertex)[self.gaussian_triangles] 
        g_delta_pos = jt.sum(self.weight_g_pos * delta_pos_, dim=1)
        
        # get deform cov
        R_ = cur_rot[self.gaussian_triangles]
        g_delta_r = jt.sum(self.weight_g_rs * R_, dim=1)
        self.deform_rot = g_delta_r.transpose(1,2)

        S_ = cur_shear[self.gaussian_triangles]
        g_delta_s = jt.sum(self.weight_g_rs * S_, dim=1)
        g_delta_rs = jt.matmul(self.deform_rot, g_delta_s)

        self.deform_cov = jt.matmul(jt.matmul(g_delta_rs, self.origin_cov), g_delta_rs.transpose(1,2))
        self.deform_pos = self.origin_pos + g_delta_pos  
        #setting flag
        self.deform_flag = True
        
        
