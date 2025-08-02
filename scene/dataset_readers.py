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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import jittor as jt
import pickle
# import open3d as o3d
# import cubvh
# from utils.mesh_utils import load_mesh
# import pickle
# import torch
# from torchvision import transforms
# from scipy.spatial.transform import Rotation as R

def compute_oriented_bounding_box(points):
    """
    计算点云的 Oriented Bounding Box (OBB)

    :param points: (N, 3) 点云数据
    :return: (obb_center, obb_axes, obb_extents)
        obb_center: OBB 的中心 (3,)
        obb_axes: OBB 的局部坐标轴 (3, 3)，每一列是一个轴的方向
        obb_extents: OBB 的尺寸 (3,) 对应长、宽、高
    """
    # 1. 计算点云的中心
    centroid = np.mean(points, axis=0)

    # 2. 计算协方差矩阵
    cov_matrix = np.cov(points - centroid, rowvar=False)

    # 3. 计算 PCA 的特征向量（OBB 轴）
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 4. 变换点云到 OBB 坐标系
    transformed_points = (points - centroid) @ eigenvectors

    # 5. 计算 AABB（轴对齐包围盒）
    min_bound = np.min(transformed_points, axis=0)
    max_bound = np.max(transformed_points, axis=0)
    obb_extents = max_bound - min_bound  # OBB 的尺寸

    # 6. 计算 OBB 的中心
    obb_center_local = (min_bound + max_bound) / 2
    obb_center = centroid + obb_center_local @ eigenvectors.T

    # 7. OBB 轴方向即为 PCA 特征向量
    obb_axes = eigenvectors

    return obb_center, obb_axes, obb_extents
def sample_points_in_obb(obb_center, obb_axes, obb_extents, k):
    """
    在 OBB 内均匀采样 k 个点
    :param obb_center: OBB 中心 (3,)
    :param obb_axes: OBB 坐标轴 (3, 3)
    :param obb_extents: OBB 尺寸 (3,)
    :param k: 采样点数
    :return: (k, 3) 均匀采样点
    """
    # 1. 在 OBB 的局部坐标系中均匀采样
    local_samples = np.random.uniform(-obb_extents / 2, obb_extents / 2, size=(k, 3))

    # 2. 变换到全局坐标系
    global_samples = obb_center + local_samples @ obb_axes.T

    return global_samples
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    mask: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    segment: list

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    # mesh_BVH: cubvh.cuBVH
    # mesh: o3d.geometry.TriangleMesh
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    # mesh_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder,segment_method = 'ours',is_exist_bg = False):
    cam_infos = []
    is_exist_mask = False
    masks_folder = os.path.join(os.path.dirname(images_folder),"masks")
    if(os.path.exists(masks_folder)):
        is_exist_mask = True
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]  # 焦距 f
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            k = intr.params[3]  # 径向畸变参数
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0].split('_')[-1]
        image = Image.open(image_path)

        #for GSMESH training
        if(is_exist_mask):
            # mask_name = image_name + ".png"
            
            mask_name = str(idx) + ".jpg"
            print(image_name)
            print(mask_name)
            mask_path = os.path.join(masks_folder,mask_name)
            mask = Image.open(mask_path)
        elif(is_exist_bg and not is_exist_mask):
            assert False, "You need mask to deform the scene!"
        else:
            mask = None



        if segment_method =='ours':
            segment_dir = 'segment'
        else:
            segment_dir = 'segment_' + segment_method
        # segment = None
        # print(os.path.join(images_folder, "../segment/SEG_"+image_name+'.pkl'))

        segments = None
        if os.path.exists(os.path.join(images_folder, "../"+segment_dir+"/SEG_"+image_name+'.pkl')):
            with open(os.path.join(images_folder, "../"+segment_dir+"/SEG_"+image_name+'.pkl'),'rb') as F:
                segments = jt.array(pickle.load(F))
        # print(masks.shape)
        # cc()
        # print(os.path.join(images_folder, "../"+segment_dir+"/SEG_"+image_name+'.pkl'))
        # cc()

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,segment = segments,mask = mask,
                              image_path=images_folder, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8,segment_method = 'ours', random_initial = False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir),segment_method = segment_method)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    if random_initial:
        points =pcd.points
        num_pts = points.shape[0] * 2


        Q1 = np.percentile(points, 25, axis=0)
        Q3 = np.percentile(points, 75, axis=0)
        IQR = Q3 - Q1

        # 定义离群点范围
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 过滤掉离群点
        mask = np.all((points >= lower_bound) & (points <= upper_bound), axis=1)
        points = points[mask]
        obb_center, obb_axes, obb_extents = compute_oriented_bounding_box(points)
        xyz = sample_points_in_obb(obb_center, obb_axes, obb_extents, num_pts)


        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        print('random initial:True !!!!!!!!!!!!!!!!!!!!!')

    # mesh = None
    # mesh_BVH = None
    # mesh_path = os.path.join(path, "mesh.obj")
    # if os.path.exists(mesh_path):

    #     mesh = load_mesh(mesh_path)

    #     print(f"Load BVH from mesh ({mesh_path})...")
    #     mesh_BVH = cubvh.cuBVH(mesh.verts_list()[0].detach().cpu().numpy(), mesh.faces_list()[0].detach().cpu().numpy())


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png",segment_method = 'ours'):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)


            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            mask = Image.fromarray(np.array(norm_data[:, :, 3:4].squeeze()*255.0, dtype=np.byte), "L")
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            # segment = []
            if segment_method =='ours':
                segment_dir = 'segment'
            elif segment_method =='rctree':
                segment_dir = 'segment_rctree'
            
            masks = None
            if os.path.exists(os.path.join(path, segment_dir + "/SEG_"+frame["file_path"].split('_')[-1]+'.pkl')):
                with open(os.path.join(path, segment_dir + "/SEG_"+frame["file_path"].split('_')[-1]+'.pkl'),'rb') as F:
                    masks = jt.array(pickle.load(F).numpy())
                # for i in range(1,masks.max().long().item()+1):
                #     seg = (masks==i).float()
                #     segment.append(seg)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], mask = mask,segment = masks))

    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png",segment_method = 'ours'):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, segment_method)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, segment_method)

    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None


    # mesh = None
    # mesh_BVH = None
    # mesh_path = os.path.join(path, "mesh.obj")
    # if os.path.exists(mesh_path):

    #     mesh = load_mesh(mesh_path)

    #     print(f"Load BVH from mesh ({mesh_path})...")
    #     mesh_BVH = cubvh.cuBVH(mesh.verts_list()[0].detach().cpu().numpy(), mesh.faces_list()[0].detach().cpu().numpy())
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
