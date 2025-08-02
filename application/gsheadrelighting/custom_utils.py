import os
from plyfile import PlyData, PlyElement
import numpy as np
import jittor as jt

def save_ply(xyz, features_dc, features_rest, scale, opacity, rotation, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(features_dc.shape[1]*features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(features_rest.shape[1]*features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scale.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation.shape[1]):
        l.append('rot_{}'.format(i))

    xyz = xyz.detach().numpy()
    normals = np.zeros_like(xyz)
    f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
    f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
    opacities = opacity.detach().numpy()
    scale = scale.detach().numpy()
    rotation = rotation.detach().numpy()

    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def project_batched_3d_to_2d(mean_3d_batched, covariance_3d_batched, extrinsic_matrix_batched):
    """
    Example of inputs
    # Camera extrinsic matrix (batched)
    extrinsic_matrix_batched = jt.randn((batch_size, 3, 4))

    # 3D Gaussian (mean and covariance, batched)
    mean_3d_batched = jt.randn((batch_size, 3))
    covariance_3d_batched = jt.randn((batch_size, 3, 3))
    """
    batch_size, num_gaussians = mean_3d_batched.shape[0:2]
    
    if mean_3d_batched is not None:
        # Apply extrinsic matrix to batched 3D mean
        mean_3d_homogeneous_batched = jt.cat([mean_3d_batched, jt.ones((batch_size, num_gaussians, 1))], dim=2)
        mean_2d_homogeneous_batched = jt.matmul(extrinsic_matrix_batched, mean_3d_homogeneous_batched.unsqueeze(-1))

        # Normalize homogeneous coordinates
        mean_2d_normalized_batched = mean_2d_homogeneous_batched[:, :, :, :-1] / mean_2d_homogeneous_batched[:, :, :, -1]
    else:
        mean_2d_normalized_batched = None

    # Project batched 3D covariance to 2D using extrinsic matrix
    covariance_2d_batched = jt.matmul(
        extrinsic_matrix_batched[:, :, :3],
        jt.matmul(covariance_3d_batched, extrinsic_matrix_batched[:, :, :3].transpose(1, 2))
    )

    return mean_2d_normalized_batched, covariance_2d_batched
