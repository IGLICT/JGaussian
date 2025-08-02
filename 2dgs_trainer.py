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
import jittor as jt
from random import randint
from utils.loss_utils import l1_loss, ssim, total_variation_loss
from gaussian_renderer_2dgs import render
import sys
from scene_pbr import Scene
from scene_pbr.gaussian_pbr_model import GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import cv2
import imageio
    
jt.flags.use_cuda = 1
jt.gc()
def save_image(mat,path):
    mat = mat.transpose(1,2,0)
    mat = mat[:,:,[2,1,0]].clamp(0,1) * 255
    cv2.imwrite(path,mat.numpy().astype(np.uint8))
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    case = os.path.basename(dataset.source_path)
    shiny_cases =  ['car', 'helmet', 'teapot']
    default_roughness = 0
    if case in shiny_cases:
        default_roughness = -100

    gaussians = GaussianModel(dataset.sh_degree, dataset.brdf_dim, dataset.brdf_mode, dataset.brdf_envmap_res, default_roughness=default_roughness)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = jt.load(checkpoint)
        gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = jt.array(bg_color, dtype=jt.float32)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_mask_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    loss_print_bar = tqdm(total=1)
    first_iter += 1
    # opt.densify_from_iter = 0
    # opt.densification_interval=20
    # opt.iterations = 10
    for iteration in range(first_iter, opt.iterations + 1):        
        gaussians.update_learning_rate(iteration)
        if iteration < opt.densify_until_iter:
            gaussians.reset_viewspace_point()
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack= scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        if gaussians.brdf_mode=="envmap":
            gaussians.brdf_mlp.build_mips()
                
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image
        gt_alpha_mask_cuda = viewpoint_cam.gt_alpha_mask
        Ll1 = l1_loss(image*gt_alpha_mask_cuda, gt_image*gt_alpha_mask_cuda)
        loss_dict = { }
        loss_dict["ssim_loss"] = opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss_dict["mse_loss"] = (1.0 - opt.lambda_dssim) * Ll1

        lambda_normal = opt.lambda_normal if iteration > 3000 else 0.0
        lambda_dist = opt.lambda_dist
        
        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        # surf_normal = viewpoint_cam.normal_image
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        loss_dict["normal_loss"] = lambda_normal * (normal_error).mean()
        # loss_dict["dist_loss"] = lambda_dist * (rend_dist).mean()
        loss_dict["tv_loss"] = 0.001 * (total_variation_loss(render_pkg["diffuse"])+total_variation_loss(render_pkg["specular"])+total_variation_loss(render_pkg["roughness"]))
        loss_dict["mask_loss"] = opt.lambda_mask * l1_loss(render_pkg["rend_alpha"][0:1, :, :], gt_alpha_mask_cuda)
        
        # loss
        # total_loss = mse_loss + ssim_loss + mask_loss + normal_loss+ tv_loss
        total_loss = sum(list(loss_dict.values()))
        gaussians.optimizer.backward(total_loss)
        
        if iteration < opt.densify_until_iter:
            viewspace_point_tensor_grad = gaussians.get_viewspace_point_grad()
        update_flag = False
        
        with jt.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{3}f}"})
                progress_bar.update(10)
                radio = (gaussians.get_opacity ==1).sum() / gaussians.get_opacity.shape[0]
                loss_print_bar.set_postfix(loss_dict)
                loss_print_bar.update(0)
            if iteration == opt.iterations:
                progress_bar.close()
                loss_print_bar.close()


            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            # [Reason]: [f 0721 19:41:16.914143 60 helper_cuda.h:128] CUDA error at /home/zhangbotao/.cache/jittor/jt1.3.9/g++11.4.0/py3.10.16/Linux-6.8.0-60xb3/IntelRXeonRSilx06/2237/default/cu11.8.89_sm_80_86/jit/where__Ti_bool__To_int32__NDIM_1__JIT_1__JIT_cuda_1__index_t_int32_hash_f9e62f6ea235fc9a_op.cc:199  code=700( cudaErrorIllegalAddress ) cudaMemcpy(&n, np, 4, cudaMemcpyDeviceToHost)
            # if iteration < opt.densify_until_iter:
            # # if iteration < opt.densify_until_iter:    
            #     def max(a,b):
            #         return jt.where(a>b,a,b)
            #     gaussians.max_radii2D[visibility_filter] = max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
            #         update_flag = True
            #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()
            #     jt.gc()
            # if (iteration in checkpoint_iterations):
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     jt.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pkl")

        if iteration < opt.iterations:
            if not update_flag:
                gaussians.optimizer.step()
            gaussians.optimizer.zero_grad()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    # if TENSORBOARD_FOUND:
    #     tb_writer = SummaryWriter(args.model_path)
    # else:
    #     print("Tensorboard not available: not logging progress")
    return tb_writer

# @torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        # torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = jt.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = jt.clamp(viewpoint.original_image, 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # torch.cuda.empty_cache()

if __name__ == "__main__":
    jt.gc()
    jt.sync()
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=60099)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=list(range(2000, 30000, 2000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 5_000, 6999,10000, 15000,20000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)


    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")