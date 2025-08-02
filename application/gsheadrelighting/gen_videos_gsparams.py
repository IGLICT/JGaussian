# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""
import warnings
warnings.filterwarnings("ignore")  # 忽略所有 Python 警告

import sys
import os

# get path to parent directory
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

sys.path.append(parent_directory)

import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import jittor as jt
from tqdm import tqdm
import legacy

from camera_utils import LookAtPoseSampler
import jittorutils

from training.gs_generator import GSGenerator
import time
from custom_utils import save_ply

import json
import cv2
import pyshtools
from PIL import Image
from typing import List, Optional, Tuple, Union
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
import warnings
warnings.filterwarnings("ignore")
from scipy import ndimage # gaussian blur
jt.flags.use_cuda = 1
jt.flags.log_silent = 1  # 1=WARNING, 2=ERROR, 3=SILENT
#----------------------------------------------------------------------------
def save_image(mat,path):
    mat = mat.transpose(1,2,0)
    mat = mat[:,:,[2,1,0]].clamp(0,1) * 255
    cv2.imwrite(path,mat.numpy().astype(np.uint8))

def SH_basis(normal):
    '''
        get SH basis based on normal
        normal is a Nx3 matrix
        return a Nx9 matrix
        The order of SH here is:
        1, Y, Z, X, YX, YZ, 3Z^2-1, XZ, X^2-y^2
    '''
    numElem = normal.shape[0]

    norm_X = normal[:,0]
    norm_Y = normal[:,1]
    norm_Z = normal[:,2]

    sh_basis = np.zeros((numElem, 9))
    att= np.pi*np.array([1, 2.0/3.0, 1/4.0])
    sh_basis[:,0] = 0.5/np.sqrt(np.pi)*att[0]

    sh_basis[:,1] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Y*att[1]
    sh_basis[:,2] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Z*att[1]
    sh_basis[:,3] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_X*att[1]

    sh_basis[:,4] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_X*att[2]
    sh_basis[:,5] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_Z*att[2]
    sh_basis[:,6] = np.sqrt(5)/4/np.sqrt(np.pi)*(3*norm_Z**2-1)*att[2]
    sh_basis[:,7] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_X*norm_Z*att[2]
    sh_basis[:,8] = np.sqrt(15)/4/np.sqrt(np.pi)*(norm_X**2-norm_Y**2)*att[2]
    return sh_basis

def get_shading(normal, SH):
    '''
        get shading based on normals and SH
        normal is Nx3 matrix
        SH: 9 x m vector
        return Nxm vector, where m is the number of returned images
    '''
    sh_basis = SH_basis(normal)
    shading = np.matmul(sh_basis, SH)
    #shading = np.matmul(np.reshape(sh_basis, (-1, 9)), SH)
    #shading = np.reshape(shading, normal.shape[0:2])
    return shading

def render_ball(sh):
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x**2 + z**2)
    valid = mag <=1
    y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
    normal = np.reshape(normal, (-1, 3))

    if sh.ndim == 1:
        sh = sh[None, ...]
        sh = sh.repeat(3,axis=0)
    assert sh.shape[0] == 3
    shadings = []
    for i in range(3):
        sh_mono = sh[i]
        shading = get_shading(normal, sh_mono)
        value = np.percentile(shading, 95)
        ind = shading > value
        shading[ind] = value
        # shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
        shading = shading.clip(0,1)
        shading = (shading *255.0).astype(np.uint8)
        shading = np.reshape(shading, (256, 256))
        shading = shading * valid
        shadings.append(shading)
    shading_rgb = np.stack(shadings, axis=2)
    return shading_rgb

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(jt.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = jt.arange(0, N ** 3, 1, out=jt.int64)
    samples = jt.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

def gen_interp_video(G, mp4: str, seeds, shuffle_seed=None, w_frames=60*4, kind='cubic', grid_dims=(1,1), num_keyframes=None, wraps=2, psi=1, truncation_cutoff=14, cfg='FFHQ', image_mode='image', gen_shapes=False, device="cuda", gs_params=None, shs=None, image_names=None, **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]
    # print(grid_w*grid_h)
    print(seeds)
    if num_keyframes is None:
        if len(seeds) % (grid_w*grid_h) != 0:
            raise ValueError('Number of input seeds must be divisible by grid W*H')
        num_keyframes = len(seeds) // (grid_w*grid_h)
    all_seeds = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
    for idx in range(num_keyframes*grid_h*grid_w):
        all_seeds[idx] = seeds[idx % len(seeds)]

    if shuffle_seed is not None:
        rng = np.random.RandomState(seed=shuffle_seed)
        rng.shuffle(all_seeds)

    camera_lookat_point = jt.array(G.rendering_kwargs['avg_camera_pivot'])
    zs = jt.array(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds]))
    
    # forward facing camera
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
    focal_length = 4.2647 if cfg != 'Shapenet' else 1.7074 # shapenet has higher FOV
    intrinsics = jt.array([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]])
    c = jt.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(len(zs), 1)
    ws = G.mapping(z=zs, c=c, truncation_psi=psi, truncation_cutoff=truncation_cutoff)

    shs = jt.array(shs).float()
    # print(ws[:1].shape, c[:1].shape, shs[:1].shape)
    # quit()
    ret_dict_warmup = G.synthesis_test(ws[:1], c[:1], shs[:1], gs_params=gs_params) # warm up
    dec_out = ret_dict_warmup['dec_out']
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)
    # interp_lit = scipy.interpolate.interp1d(np.arange(shs.shape[0]), shs.cpu().numpy(), kind='cubic', axis=0)

    # Render video.
    max_batch = 10000000
    voxel_resolution = 512
    # video_out = imageio.get_writer(mp4, mode='I', fps=1, codec='libx264', **video_kwargs)
    video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', **video_kwargs)

    if gen_shapes:
        outdir = 'interpolation_{}_{}/'.format(all_seeds[0], all_seeds[1])
        os.makedirs(outdir, exist_ok=True, mode=0o777)
        os.chmod(outdir, 0o777)
    all_poses = []
    
    # w_frames=3

    gaussian_params = None
    for sh, image_mode in zip(jt.cat([shs[0:1], shs], dim=0), ['image_albedo']+['image']*len(shs)):
        sh = sh.unsqueeze(0)
        for frame_idx in tqdm(range(num_keyframes * w_frames)):
            imgs = []
            for yi in range(grid_h):
                for xi in range(grid_w):
                    pitch_range = 0.25
                    # pitch_range = 0
                    yaw_range = 0.35
                    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                            3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                            camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
                    all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                    focal_length = 4.2647 if cfg != 'Shapenet' else 1.7074 # shapenet has higher FOV
                    intrinsics = jt.array([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]])
                    c = jt.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                    interp = grid[yi][xi]
                    w = jt.array(interp(frame_idx / w_frames))

                    entangle = 'camera'
                    if entangle == 'conditioning':
                        c_forward = jt.cat([LookAtPoseSampler.sample(3.14/2,
                                                                        3.14/2,
                                                                        camera_lookat_point,
                                                                        radius=G.rendering_kwargs['avg_camera_radius']).reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                        w_c = G.mapping(z=zs[0:1], c=c[0:1], truncation_psi=psi, truncation_cutoff=truncation_cutoff)
                        img = G.synthesis(ws=w_c, c=c_forward, l=sh, noise_mode='const', gs_params=gs_params)[image_mode][0]
                    elif entangle == 'camera':
                        ret_dict = G.synthesis_test(ws=w.unsqueeze(0).to(jt.float32), c=c[0:1], l=sh, noise_mode='const', gs_params=gs_params, image_mode=image_mode, dec_in=dec_out)
                        img = ret_dict['image'][0]
                        if gaussian_params is None:
                            gaussian_params = ret_dict['gaussian_params']
                        
                    elif entangle == 'both':
                        w_c = G.mapping(z=zs[0:1], c=c[0:1], truncation_psi=psi, truncation_cutoff=truncation_cutoff)
                        img = G.synthesis(ws=w_c, c=c[0:1], l=sh, noise_mode='const', gs_params=gs_params)[image_mode][0]

                    if image_mode == 'image_depth':
                        img = -img
                        img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

                    imgs.append(img)

            video_out.append_data(layout_grid(jt.stack(imgs), grid_w=grid_w, grid_h=grid_h))
    video_out.close()
    print(f"save video at {mp4}")
    all_poses = np.stack(all_poses)

    # save ply, which can be visualized in visualizing tools for Gaussian Splatting
    # ply_save_fn = mp4.replace('.mp4', '.ply')
    # save_ply(gaussian_params["_xyz"], gaussian_params["_features_dc"], gaussian_params["_features_rest"], gaussian_params["_scaling"], \
    #     gaussian_params["_opacity"], gaussian_params["_rotation"], ply_save_fn)
    # print("save ply at", ply_save_fn)


#----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

def to_tensor(img: Union[Image.Image, np.ndarray], normalize=True, device='cpu') -> jt.Var:
    if isinstance(img, Image.Image):
        img = np.array(img)
        if len(img.shape) > 2:
            img = img.transpose(2, 0, 1)
        else:
            img = img[None, ...]
    else:
        if img.shape[0] == img.shape[1]:
            img = img.transpose(2, 0, 1)
    if normalize:
        img = jt.array(img).to(jt.float32) / 127.5 - 1
    else:
        img = jt.array(img).to(jt.float32) / 255.
    return img[None, ...]

def shtools_matrix2vec(SH_matrix):
    numOrder = SH_matrix.shape[1]
    vec_SH = np.zeros(numOrder**2)
    count = 0
    for i in range(numOrder):
        for j in range(i,0,-1):
            vec_SH[count] = SH_matrix[1,i,j]
            count = count + 1
        for j in range(0,i+1):
            vec_SH[count]= SH_matrix[0, i,j]
            count = count + 1
    return vec_SH

def shtools_vec2matrix(coefficients, degree):
    '''
        convert vector of sh to matrix
    '''
    coeffs_matrix = np.zeros((2, degree + 1, degree + 1))
    current_zero_index = 0
    for l in range(0, degree + 1):
        coeffs_matrix[0, l, 0] = coefficients[current_zero_index]
        for m in range(1, l + 1):
            coeffs_matrix[0, l, m] = coefficients[current_zero_index + m]
            coeffs_matrix[1, l, m] = coefficients[current_zero_index - m]
        current_zero_index += 2*(l+1)
    return coeffs_matrix 

def shtools_getSH(envMap, order=5):
    SH_r =  pyshtools.expand.SHExpandDH(envMap[...,0], sampling=2, lmax_calc=order, norm=4)
    SH_g =  pyshtools.expand.SHExpandDH(envMap[...,1], sampling=2, lmax_calc=order, norm=4)
    SH_b =  pyshtools.expand.SHExpandDH(envMap[...,2], sampling=2, lmax_calc=order, norm=4)
    return SH_r, SH_g, SH_b

def convert_env_to_img(env):
    im_gamma_correct = np.clip(np.power(env, 0.45), 0, 1)
    return (im_gamma_correct*255).astype(np.uint8)
    # return to_tensor(Image.fromarray((im_gamma_correct*255).astype(np.uint8)))

def rotate_SH(SH, angles):
    """
    Rotate the SH coefficients.
    :param SH: SH coefficients matrix.
    :param angles: Rotation angles (alpha, beta, gamma) in degrees.
    :return: Rotated SH coefficients matrix.
    """
    alpha, beta, gamma = np.radians(angles)
    x = np.array([alpha, beta, gamma])
    dj = pyshtools.rotate.djpi2(SH.shape[-1])
    rotated_SH = pyshtools.rotate.SHRotateRealCoef(SH, x, dj)
    return rotated_SH

def get_SH_from_env(path_to_envMap: str, rotation_angles=(0, 0, 0), ldr=False, device='cpu'):
    if path_to_envMap.endswith('.exr'):
        env = imageio.imread(path_to_envMap, format='EXR-FI')[...,:3]
    else:
        env = imageio.imread(path_to_envMap)[...,:3]
    # env = cv2.resize(env, (1024, 512))
    # env = cv2.resize(env, (20, 10), interpolation=cv2.INTER_CUBIC)
    radius = int(env.shape[0] * 0.1)
    sigma = radius / 2
    blurred_env = cv2.GaussianBlur(env, (2 * radius + 1, 2 * radius + 1), sigma).astype(np.float32)
    # blurred_env_r = ndimage.gaussian_filter(env[:,:,0], sigma=50)
    # blurred_env_g = ndimage.gaussian_filter(env[:,:,1], sigma=50)
    # blurred_env_b = ndimage.gaussian_filter(env[:,:,2], sigma=50)
    # blurred_env = np.stack([blurred_env_r,blurred_env_g,blurred_env_b], -1)
    env = cv2.resize(blurred_env, (20, 10))
    # ndimage.gaussian_filter(env[:,:,0], sigma=5)
    if ldr:
        env = convert_env_to_img(env)
    imageio.imwrite(path_to_envMap.split('.')[0] + '_resize.hdr', env)
    SH_r, SH_g, SH_b = shtools_getSH(env, 2)
    SH_r_rotated = rotate_SH(SH_r, rotation_angles)
    SH_g_rotated = rotate_SH(SH_g, rotation_angles)
    SH_b_rotated = rotate_SH(SH_b, rotation_angles)
    SH = np.vstack([shtools_matrix2vec(SH_r_rotated)[None, ...],
                   shtools_matrix2vec(SH_g_rotated)[None, ...],
                   shtools_matrix2vec(SH_b_rotated)[None, ...]])
    # factor = (np.random.rand()*0.2 + 0.7)/SH.max()
    factor = 0.7 / max(0.7, SH.max())
    SH *= factor
    # return jt.array(SH).to(device), convert_env_to_img(env)
    return SH

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--grid', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1,1))
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--cfg', help='Config', type=click.Choice(['FFHQ', 'AFHQ', 'Shapenet']), required=False, metavar='STR', default='FFHQ', show_default=True)
@click.option('--image_mode', help='Image mode', type=click.Choice(['image', 'image_depth', 'image_raw']), required=False, metavar='STR', default='image', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float, help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
@click.option('--shapes', type=bool, help='Gen shapes for shape interpolation', default=False, show_default=True)
@click.option('--interpolate', type=bool, help='Interpolate between seeds', default=False, show_default=True)

@click.option('--g_type', type=str, help='G or G_ema', default="G_ema", show_default=True)
@click.option('--load_architecture', type=bool, help='load_architecture', default=False, show_default=True)

@click.option('--res_visualize', type=parse_range, help='gaussian resolution for visualization', default=[])
@click.option('--rendering_scale', type=float, help='multiplicative scaling factors for visualization', default=0.)
@click.option('--opacity_ones', type=bool, help='multiplicative scaling factors for visualization', default=False)
@click.option('--visualize_anchor', type=bool, help='multiplicative scaling factors for visualization', default=False)
@click.option('--point_index', type=parse_range, help='', default=[])
@click.option('--num_init_near_point', type=int, help='', default=-1)
@click.option('--postfix', type=str, help='name postfix', default='')

@click.option('--lighting_pattern', type=str, help='lighting pattern of video', default='envmap')
@click.option('--lighting_transfer_ids', type=parse_range, help='List of random lighting transfer id in the dataset', default=[])
@click.option('--ffhq', type=str, help='dataset dir', default='/home/jovyan/data7/lvhenglei/datasets/FFHQ')
@click.option('--sh_file_dir', type=str, help='dataset dir', default='/home/jovyan/data7/lvhenglei/projects/example_light/')
@click.option('--rgb_sh', type=bool, help='if using rgb sh', default=False, show_default=True)
@click.option('--with_bg', type=bool, help='if using background', default=True, show_default=True)

def generate_images(
    network_pkl: str,
    seeds: List[int],
    shuffle_seed: Optional[int],
    truncation_psi: float,
    truncation_cutoff: int,
    grid: Tuple[int,int],
    num_keyframes: Optional[int],
    w_frames: int,
    outdir: str,
    reload_modules: bool,
    cfg: str,
    image_mode: str,
    sampling_multiplier: float,
    nrr: Optional[int],
    shapes: bool,
    interpolate: bool,
    g_type: str,
    load_architecture: bool,
    res_visualize: List[int],
    rendering_scale: float,
    postfix: str,
    opacity_ones: bool,
    visualize_anchor: bool,
    point_index: List[int],
    num_init_near_point: int,
    lighting_pattern: str,
    lighting_transfer_ids: List[int],
    ffhq: str,
    sh_file_dir: str,
    rgb_sh: bool,
    with_bg: bool
):
    """Render a latent vector interpolation video.

    Examples:

    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    Animation length and seed keyframes:

    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.

    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.

    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """
    image_names=None
    if lighting_pattern == 'transfer':
        image_names = [f'img{id:08d}.png' for id in lighting_transfer_ids]
        json_path = os.path.join(ffhq, 'dataset.json' if not rgb_sh else 'dataset_rgb_sh_2.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        shs = data['sh']
        shs = [shs[name] for name in image_names]
        shs = np.array(shs)
        shs = shs.reshape(shs.shape[0], -1)
        shs = shs[0][None]
        # shs[1] *= 0.2
        # shs *= 0.
        # shs = [np.array(sh) for sh in shs]
        # shs = shs.astype({1: np.int64, 2: np.float32}[shs.ndim])
    elif lighting_pattern == 'circle':
        if rgb_sh:
            sh_file_dir = '/home/jovyan/data7/lvhenglei/projects/example_light_rgb'
            factor = 2.0
        else:
            factor = 0.7

        lighting_pattern = 'FFHQ_rgb'
        sh_file_dir = '/home/jovyan/data7/lvhenglei/projects/example_FFHQ_sh_rgb'
        factor = 1

        sh_names = os.listdir(sh_file_dir)
        sh_names.sort()
        sh_paths = [os.path.join(sh_file_dir, sh_name) for sh_name in sh_names]
        shs = [np.loadtxt(sh_path)[0:9] * factor for sh_path in sh_paths]
        # shs.append(shs[-1]*0)
        lighting_pattern += sh_names[-1]
        shs = [shs[-1]]
        shs = np.array(shs)
        # shs = shs[0][None]
        # shs[0] *= 0.8
        # shs[4] *= 0.4
        # shs[5] *= 0.3
        # shs[6] *= 0.3
    elif lighting_pattern == 'envmap':
        envmap_dir = "/mnt/155_16T/zhangbotao/jgaussian/data/env/resized_32x16"
        # envmap_dir = "/home/jovyan/data7/lvhenglei/projects/example_hdr"
        # envmap_dir = "/home/jovyan/data7/lvhenglei/projects/example_env_online"
        lighting_pattern = 'envmap_new_exr'
        # lighting_pattern = 'envmap_env_online'
        ldr=False
        envmap_names = os.listdir(envmap_dir)
        envmap_names.sort()
        envmap_paths = [os.path.join(envmap_dir, envmap_name) for envmap_name in envmap_names if not envmap_name.endswith('resize.hdr')]
        print(envmap_paths)
        shs = [get_SH_from_env(envmap_path, rotation_angles=(0,0,0), ldr=ldr) for envmap_path in envmap_paths]
        # up = np.loadtxt('/home/jovyan/data7/lvhenglei/projects/example_light/rotate_light_02.txt')
        # new_sh = np.stack([up*0.7, up*0.8, up*0.9], 0)
        # shs.append(new_sh*0.7)
        shs = np.array(shs)
            



    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    import pickle
    # print('Loading networks from "%s"...' % network_pkl)
    # import sys
    # sys.modules["torch_utils"] = jittorutils
    # sys.modules["torch_utils.ops"] = jittorutils

    # with open(network_pkl, "rb") as f:  # 'rb' 表示二进制读取
    #     data = pickle.load(f)
    # if g_type == "G_ema":
    #         G_ema = data["G_ema"]
    #         G = G_ema
    # elif g_type == "G":
    #         G = data["G"]
    # else:
    #         print("Wrong G_type: {}".format(g_type))
    #         exit(-1)
    # G_new = GSGenerator(**G.init_kwargs)
    # w_pytorch= []
    # w_jittor = []
    # for name, param in G_new.named_parameters():
    #     w_jittor.append(name)
    #     # print(f"Layer: {name}")
    #     # print(f"Shape: {param.shape}")
    # state_dict = G.state_dict()
    # for key, value in state_dict.items():
    #     w_pytorch.append(key)
    #     # print(f"Layer: {key}")
    #     # print(f"Shape: {value.shape}")
    # for name, param in state_dict.items():
    #     if name in ['_xyz','_xyz_bg']:
    #         name = 'jt' + name
    #     if name in G_new.state_dict():
    #         # 获取 Jittor 中对应的参数
    #         jt_param = G_new.state_dict()[name]
    #         # 检查形状是否匹配
    #         if tuple(param.shape) == tuple(jt_param.shape):
    #             # 将 PyTorch Tensor 转为 Jittor Tensor 并赋值
    #             jt_param.assign(jt.array(param.cpu().detach().numpy()))
    #         else:
    #             print(f"[ERR] 形状不匹配: {name} (PyTorch: {param.shape} vs Jittor: {jt_param.shape})")
    #     else:
    #         print(f"[ERR] 未找到对应层: {name}")
    # G = G_new        
    # # G.save("/mnt/155_16T/zhangbotao/jgaussian/checkpoints/gshead.pkl")
    # # 打包权重和参数
    # save_data = {
    #     "state_dict": G.state_dict(),  # 模型权重
    #     "init_kwargs": G.init_kwargs,  # 初始化参数
    # }

    # # 保存到单个 .pkl 文件
    # save_path = "/mnt/155_16T/zhangbotao/jgaussian/checkpoints/gshead.pkl"
    # with open(save_path, "wb") as f:
    #     pickle.dump(save_data, f)

    with open(network_pkl, "rb") as f:
        loaded_data = pickle.load(f)
    # 重新初始化模型
    G = GSGenerator(**loaded_data["init_kwargs"])
    # 加载权重
    G.load_state_dict(loaded_data["state_dict"])

    # TODO visualization parameters
    if len(res_visualize) == 0:
        res_visualize = None # if length is 0, use all resolutions 
    
    # Configuration for visualization
    gs_params = {
        "rendering_scale": 0,                           # reduce the scale of Gaussians at rendering
        "res_visualize": res_visualize,                 # visualizing specific blocks only (e.g. [0, 1] for two coarsest blocks)
        "disable_background": not with_bg,                    # True, for disabling background generator
        "opacity_ones": opacity_ones,                   # True, for setting opacity of Gaussians to 1
        "point_index": point_index,                     # index of the initial point and its children to visualize
        "num_init_near_point": num_init_near_point,     # number of points to visualize near the initial point (use with "point_index")
        "visualize_anchor": visualize_anchor,           # visualinzg anchors, not actual Gaussians
        'visualize_index': False,                       
        'camera_cond': [[-4.7208e-08, 4.7208e-08,  - 1.0000e+00]], # camera direction condition of color layer for visualization (fix for consistent representation)
    }
    
    postfix += f'_{lighting_pattern}'
    if lighting_pattern == 'transfer':
        postfix += f"_{'_'.join([str(id) for id in lighting_transfer_ids])}"
    if gs_params['opacity_ones']:
        postfix = postfix + '_ones'
    
    print("visualization parameters: {}".format(gs_params))        


    if nrr is not None: G.neural_rendering_resolution = nrr

    if truncation_cutoff == 0:
        truncation_psi = 1.0 # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14 # no truncation so doesn't matter where we cutoff

    if interpolate:
        output = os.path.join(outdir, f'interpolation{postfix}.mp4')
        gen_interp_video(G=G, mp4=output, bitrate='10M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds, shuffle_seed=shuffle_seed, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, gen_shapes=shapes, gs_params=gs_params)
    else:
        for seed in seeds:
            output = os.path.join(outdir, f'{seed}{postfix}.mp4')
            seeds_ = [seed]
            gen_interp_video(G=G, mp4=output, bitrate='10M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds_, shuffle_seed=shuffle_seed, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, gs_params=gs_params, shs=shs, image_names=image_names)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    # with jt.no_grad():
        generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
