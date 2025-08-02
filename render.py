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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render

from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
# from scene import Scene, GaussianModel
from gaussian_renderer import GaussianModel
import imageio
import cv2
import numpy as np

from scene.colmap_loader import qvec2rotmat, rotmat2qvec
from scene.cameras import Camera

jt.flags.use_cuda = 1


def save_image(mat,path):
    mat = mat.transpose(1,2,0)
    mat = mat[:,:,[2,1,0]].clamp(0,1) * 255
    cv2.imwrite(path,mat.numpy().astype(np.uint8))

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    frames = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        rendering = (rendering * 255).clamp(0,255).permute(1,2,0).numpy().astype(np.uint8)
        frames.append(rendering)

    vid_path = render_path + '.mp4'
    imageio.mimwrite(vid_path, frames, fps=30)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, deform_mesh_path : str):
     with jt.no_grad():
        mesh_path = ""
        if(os.path.exists(os.path.join(dataset.model_path,"mesh.obj"))):
            print("guess gaussian mesh format,if not,please check")
            mesh_path = os.path.join(dataset.model_path,"mesh.obj")
        gaussians = GaussianModel(dataset.sh_degree,mesh_path)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        train_name = "train"
        test_name = "test"
        if os.path.exists(deform_mesh_path):
            print("find deform mesh")
            new_name = os.path.basename(deform_mesh_path)
            train_name = train_name + new_name
            test_name = test_name + new_name
            gaussians.deform_gaussian_mesh(deform_mesh_path)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = jt.array(bg_color, dtype=jt.float32)
        # gaussians.prune_over_opacity_points(1, 0.005)
        # print(gaussians.get_xyz)
        if not skip_train:
             render_set(dataset.model_path, train_name, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        if not skip_test:
             render_set(dataset.model_path, test_name, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

def render_ring_cameras(dataset : ModelParams, iteration : int, pipeline : PipelineParams, frame_number: int, ring_radius: float, fps: int, deform_mesh_path : str):
    with jt.no_grad():
        mesh_path = ""
        if(os.path.exists(os.path.join(dataset.model_path,"mesh.obj"))):
            print("guess gaussian mesh format,if not,please check")
            mesh_path = os.path.join(dataset.model_path,"mesh.obj")
        gaussians = GaussianModel(dataset.sh_degree, mesh_path)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = jt.array(bg_color, dtype=jt.float32)
        model_path = dataset.model_path
        name = "render_res"
        iteration = scene.loaded_iter
        views = scene.getTrainCameras()
        newnewView = []
        for i in range(frame_number):
            theta = 2 * np.pi * i / frame_number
            newnewView.append(Camera(colmap_id=i, R=qvec2rotmat(np.array([1,0,0,0])), T=np.array([ring_radius*np.cos(theta),ring_radius*np.sin(theta),0]), 
                  FoVx=views[0].FoVx, FoVy=views[0].FoVy, 
                  image=views[0].original_image, gt_alpha_mask=None,
                  image_name=views[0].image_name, uid=i,mask=None,segment=None,
                  image_path="",width = scene.img_width,height = scene.img_height,resolution=1))

        # makedirs(os.path.join(model_path, name, "ours_{}".format(iteration), "video_gen"), exist_ok=True)
        allpics = []
        # toPIL = torchvision.transforms.ToPILImage()
        for idx, view in enumerate(tqdm(newnewView, desc="Rendering progress")):
            rendering = render(view, gaussians, pipeline, background)["render"]
            # pic = toPIL(rendering)
            pic = (rendering * 255).clamp(0,255).permute(1,2,0).numpy().astype(np.uint8)
            allpics.append(pic)
        # Save video
        with imageio.get_writer(os.path.join(model_path, f"render_{iteration}.mp4"), fps=fps) as video:
            for image in allpics:
                frame = imageio.core.asarray(image)
                video.append_data(frame)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--deform_mesh", default="", type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gen_video", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.gen_video:
        render_ring_cameras(model.extract(args), args.iteration, pipeline.extract(args), 30, 1.0, 30, args.deform_mesh)
    else:
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.deform_mesh)