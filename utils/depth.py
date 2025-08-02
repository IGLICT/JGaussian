from einops import rearrange
import torch
import math
import numpy as np
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))
def get_normal_map_by_point_cloud(pts_3d):
    if isinstance(pts_3d, np.ndarray):
        is_cuda = False
    else:
        is_cuda = True
    H, W, C = pts_3d.shape
    
    # print(f.shape)
    # print(t.shape)
    if is_cuda:
        f = pts_3d[0:H-1, 1:W, :]  - pts_3d[0:H-1, 0:W-1, :]
        t = pts_3d[1:H, 0:W-1, :]  - pts_3d[0:H-1, 0:W-1, :]
        output_normal = torch.zeros((H, W, C)).float().cuda()
        normal_cross = torch.cross(f, t)
        output_normal[0:H-1, 0:W-1, :] = normal_cross
        normal_unit = output_normal / (torch.linalg.norm(output_normal, axis=-1, keepdims=True) + 1e-10)
    else:
        pts_3d = rearrange(pts_3d, 'H W C -> C H W')
        f = (
            pts_3d[:, 0:H-1, 1:W] - pts_3d[:, 0:H-1, 0:W-1]
        )
        t = (
            pts_3d[:, 1:H, 0:W-1] - pts_3d[:, 0:H-1, 0:W-1]
        )
        output_normal = np.zeros((H, W, C), dtype=np.float32)
        normal_cross = np.cross(f, t, axisa=0, axisb=0)
        output_normal[0:H-1, 0:W-1, :] = normal_cross
        normal_unit = output_normal / (np.linalg.norm(output_normal, axis=-1, keepdims=True) + 1e-10)
    return normal_unit



def get_world_space_normal_from_depth_cuda(depth, view, is_distance_map=False):
    # save normal
    depth = depth.squeeze()
    
    # intrinsics = view.projection_matrix
    cx = view.image_height/2
    cy = view.image_width/2
    focal_length_x = fov2focal(view.FoVx, view.image_height)
    focal_length_y = fov2focal(view.FoVy, view.image_width)
    
    w2c = np.eye(4).astype(float)
    w2c[:3, :3] = view.R.transpose()
    w2c[:3, 3] = view.T
    pose = np.linalg.inv(w2c)
    pose = torch.from_numpy(pose).float().cuda()
    
    u, v = torch.meshgrid(torch.arange(depth.shape[0]), torch.arange(depth.shape[1]), indexing='ij')
    u = u.float().cuda()
    v = v.float().cuda()
    uvd = torch.stack([v, u, depth], dim=-1).float().cuda()

    x_over_z = (uvd[..., 0] - cx) / focal_length_x
    y_over_z = (uvd[..., 1] - cy) / focal_length_y
    if is_distance_map:
        z = depth / torch.sqrt(1. + x_over_z**2 + y_over_z**2)
    else:
        z = depth
    x = x_over_z * z
    y = y_over_z * z
    pc_c = torch.stack([x, y, z], dim=-1).float().cuda()
    # pc_c = torch.tensor([convert_from_uvd(w, h, depth[h, w], cx=cx, cy=cy, focalx=focal_length_x, focaly=focal_length_y, is_distance_map=is_distance_map) for h in range(depth.shape[0]) for w in range(depth.shape[1])]).float().cuda()
    pc_c = pc_c.reshape((-1, 3))
    # pc_c = transform_matrix[:3, :3].T @ (pc_w - transform_matrix[:, 3])
    # this is neus world coordinate
    pc_w = (pose[:3, :3] @ pc_c.T).T + pose[:3, 3].reshape(-1, 3)
    
    new_depth = pc_c.reshape((depth.shape[0], depth.shape[1], 3)).float().cuda()
    normal_image = get_normal_map_by_point_cloud(new_depth)

    normal_image[..., [0]] = -1 * normal_image[..., [0]]
    normal_image[..., [1]] = -1 * normal_image[..., [1]]
    normal_image[..., [2]] = -1 * normal_image[..., [2]]

    normal_image_w = (pose[:3, :3] @ normal_image.reshape(-1, 3).T).T 
    normal_image_w = normal_image_w / (torch.linalg.norm(normal_image_w, axis=-1, keepdims=True) + 1e-10)
    normal_image_w[torch.isnan(normal_image_w)] = 0
    normal_image_w = normal_image_w.reshape(normal_image.shape)
    return normal_image_w