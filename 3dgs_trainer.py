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
import random
import jittor as jt
from jittor import nn
from random import randint
from utils.loss_utils import cal_local_normal, l1_loss, ssim,build_knn, repulsion_loss,global_median_loss, NNFMLoss, match_colors_for_image_set, color_histgram_match, mesh_restrict_loss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, StyleOptimizationParams
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import imageio
jt.flags.use_cuda = 1
jt.gc()
number = {'mic':300000,'chair':250000,'ship':300000,'materials':250000,'lego':300000,'drums':300000,'ficus':300000,'hotdog':120000,
          'bicycle':6000000, 'kitchen':1800000,'flowers':3600000, 'stump':4900000, 'garden':5800000, 'counter':1200000, 'bonsai':1200000, 'treehill':3500000, 'room':1500000,
          'drjohnson': 3400000, 'playroom':2500000,'bedroom':1500000,
          'train':1000000,'truck':2500000,
          'trex':100000000,'room2':100000000
          }

def save_image(mat,path):
    mat = mat.transpose(1,2,0)
    mat = mat[:,:,[2,1,0]].clamp(0,1) * 255
    cv2.imwrite(path,mat.numpy().astype(np.uint8))
def crop_nonzero_region(image, padding=1):
    # 找到非零元素的坐标
    nonzero_indices = jt.nonzero(image != 0)

    row, col = image.shape[-2:]
    if len(nonzero_indices) == 0:
        # 如果图像中没有非零元素，返回空张量或其他适当的处理
        return jt.tensor([])

    # 计算最小包围矩形的坐标范围
    min_row = jt.min(nonzero_indices[:, 2])
    max_row = jt.max(nonzero_indices[:, 2])
    min_col = jt.min(nonzero_indices[:, 3])
    max_col = jt.max(nonzero_indices[:, 3])


    # 切片原始图像以获得矩形区域
    cropped_image = image[:, :, max(0, min_row-padding):min(row, max_row + 1+padding), max(0, min_col-padding):min(col, max_col + 1+padding)]

    # print('cropped feat', cropped_image.shape)
    return cropped_image

def set_geometry_grad(gaussian_model, freeze):
    if freeze:
        # Jittor 中通过 stop_grad() 冻结梯度
        if gaussian_model.is_gsmesh:
            gaussian_model._bc.stop_grad()
            gaussian_model._distance.stop_grad()
        else:
            gaussian_model._xyz.stop_grad()
        gaussian_model._scaling.stop_grad()
        gaussian_model._rotation.stop_grad()
        gaussian_model._opacity.stop_grad()
    else:
        # Jittor 中通过 start_grad() 恢复梯度计算
        if gaussian_model.is_gsmesh:
            gaussian_model._bc.start_grad()
            gaussian_model._distance.start_grad()
        else:
            gaussian_model._xyz.start_grad()
        gaussian_model._scaling.start_grad()
        gaussian_model._rotation.start_grad()
        gaussian_model._opacity.start_grad()
# from jt.utils.tensorboard import SummaryWriter
TENSORBOARD_FOUND = True



import logging
# import rich
# from utils.graphics_utils import depth_to_normals


from scene.dataset import ColmapDataset
import shutil


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, point_cloud, is_stylized, input_mesh):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset,opt, pipe)
    # logging.basicConfig(filename=dataset.model_path+'/log.txt', level=logging.INFO)
    gaussians = GaussianModel(dataset.sh_degree, input_mesh)
    scene = Scene(dataset, gaussians)
    if point_cloud:
        xyz, o, s = gaussians.load_ply(point_cloud, reset_basis_dim=1)
        original_xyz, original_opacity, original_scale = jt.array(xyz), jt.array(o), jt.array(s)
        first_iter = 30_000    

    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = jt.load(checkpoint)
        gaussians.restore(model_params, opt)

    if is_stylized:
        nnfm_loss_fn = NNFMLoss(device='cuda')
        ###### ARF: resize style image such that its long side matches the long side of content images
        style_img = imageio.imread(args.style, pilmode="RGB").astype(np.float32) / 255.0 # pilmode="RGB"
        style_h, style_w = style_img.shape[:2]
        content_long_side = max([scene.img_width, scene.img_height])
        if style_h > style_w:
            style_img = cv2.resize(
                style_img,
                (int(content_long_side / style_h * style_w), content_long_side),
                interpolation=cv2.INTER_AREA,
            )
        else:
            style_img = cv2.resize(
                style_img,
                (content_long_side, int(content_long_side / style_w * style_h)),
                interpolation=cv2.INTER_AREA,
            )
        style_img = cv2.resize(
            style_img,
            (style_img.shape[1] // 2, style_img.shape[0] // 2),
            interpolation=cv2.INTER_AREA,
        )
        imageio.imwrite(
            os.path.join(args.model_path, "style_image.jpg"),
            np.clip(style_img * 255.0, 0.0, 255.0).astype(np.uint8),
        )
        style_img = jt.array(style_img)
        # Load style image mask or second style image
        if args.second_style:
            style_img2 = imageio.imread(args.second_style, pilmode="RGB").astype(np.float32) / 255.0
            style_img2 = cv2.resize(style_img2, (style_img.shape[1],style_img.shape[0]), interpolation=cv2.INTER_AREA)
            style_img2 = cv2.resize(
                style_img2,
                (style_img.shape[1], style_img.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            
            imageio.imwrite(
                os.path.join(args.model_path, "style_image2.jpg"),
                np.clip(style_img2 * 255.0, 0.0, 255.0).astype(np.uint8),
            )
            style_img2 = jt.array(style_img2)
            style_mask = None
        else:
            style_mask = None
            style_img2 = None
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = jt.array(bg_color, dtype=jt.float32)



    viewpoint_stack = None
    viewpoint_cams = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    loss_print_bar = tqdm(total=1)
    status_print_bar = tqdm(total=1)
    first_iter += 1
    color_diff = None
    train_cam = scene.getTrainCameras().copy()
    gt_img_list = []
    for view in train_cam:
        # print(args.preserve_color)
        if not args.preserve_color:
            gt_img_list.append(view.original_image.permute(1,2,0))
        else:
            gt_img_list.append(view.original_image)

    # prepare depth image & sam mask
    depth_img_list = []
    mask_img_list = []
    mask_half_list = []

    with jt.no_grad():
        gaussians.reset_viewspace_point()
        for i, view in enumerate(tqdm(scene.getTrainCameras(), desc="Rendering progress")):
            depth_render = render(view, gaussians, pipe, background)["depth"]
            depth_img_list.append(depth_render)
            select_mask = view.mask[0].numpy()
            select_mask = gaussian_filter(select_mask, sigma=1)
            mask_img_list.append(jt.array((cv2.resize(select_mask, (scene.img_width, scene.img_height),interpolation=cv2.INTER_AREA)).astype(np.int8)))
            mask_half_list.append(jt.array(cv2.resize(select_mask, (scene.img_width//2, scene.img_height//2),interpolation=cv2.INTER_AREA)))

        if is_stylized:
            if not args.preserve_color:
                gt_imgs = jt.stack(gt_img_list)
                
                if args.second_style:
                    gt_imgs1, color_ct = match_colors_for_image_set(gt_imgs, style_img)
                    gt_imgs2, color_ct2 = match_colors_for_image_set(gt_imgs, style_img2)
                    mask_imgs = jt.stack(mask_img_list).unsqueeze(-1).repeat(1,1,1,3)
                    gt_imgs = gt_imgs1 * (1-mask_imgs) + gt_imgs2 * mask_imgs
                else:
                    mask_imgs = jt.stack(mask_img_list).unsqueeze(-1).repeat(1,1,1,3)
                    if args.histgram_match:
                        gt_imgs, color_ct = color_histgram_match(gt_imgs, style_img) #.repeat(gt_imgs.shape[0],1,1,1))
                         
                    else:
                        gt_imgs, color_ct = match_colors_for_image_set(gt_imgs, style_img)

                    
                gt_img_list = [item.permute(2,0,1) for item in gt_imgs]
                imageio.imwrite(
                    os.path.join(args.model_path, "gt_image_recolor.png"),
                    np.clip(gt_img_list[0].permute(1,2,0).numpy() * 255.0, 0.0, 255.0).astype(np.uint8),
                )

        #YOU CAN USE IT TO REDUCE THE MEMORY, BUT MAY CANNOT DIRECTLY USE IN GSMESH OR STYLIZEDGS (NEED MODIFIED)
        # if scene.scene_type == 'colmap':
        #     def custom_collate(data):
        #         return data
        #     train_dataset = ColmapDataset(scene.getTrainCameras())
        #     train_dataloader = jt.dataset.DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=2,collate_fn=custom_collate, persistent_workers = True)
        #     test_dataset = ColmapDataset(scene.getTestCameras())
        #     test_dataloader = jt.dataset.DataLoader(test_dataset, batch_size=None, shuffle=False,num_workers=0,collate_fn=custom_collate)
        #     traindata_iter = iter(train_dataloader)
        # else:
        #     test_dataloader = None



    shutil.copy('3dgs_trainer.py', scene.model_path+'/train.py')
    knn_pos = None
    for iteration in range(first_iter, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)
        if iteration < opt.densify_until_iter:
            gaussians.reset_viewspace_point()
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        #YOU CAN USE IT TO REDUCE THE MEMORY, BUT MAY CANNOT DIRECTLY USE IN GSMESH OR STYLIZEDGS (NEED MODIFIED)
        # if scene.scene_type != 'colmap':
        #     if not viewpoint_stack:
        #         viewpoint_stack = scene.getTrainCameras().copy()
        #         need_densification = False
        #     view_idx = randint(0, len(viewpoint_stack)-1)
        #     # viewpoint_cam = viewpoint_stack[17]
        #     viewpoint_cam = viewpoint_stack.pop(view_idx)
        # else:
        #     if not viewpoint_cams:
        #         try:
        #             viewpoint_cams = next(traindata_iter)
        #         except StopIteration:
        #             traindata_iter = iter(train_dataloader)
        #             viewpoint_cams = next(traindata_iter)
        #     viewpoint_cam = viewpoint_cams.pop()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            gt_stack = gt_img_list.copy()
            depth_stack = depth_img_list.copy()
            mask_stack = mask_half_list.copy()
        view_idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(view_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, background,test = False, color_diff = None)
        image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"],render_pkg["depth"]

        # basic loss
        if is_stylized:
            # stylized loss
            gt_image = gt_stack.pop(view_idx)
            depth_gt = depth_stack.pop(view_idx)
            mask_image = mask_stack.pop(view_idx)

            gt_image = gt_image.unsqueeze(0)
            pred_image = image.unsqueeze(0)

            w_variance = jt.mean(jt.pow(pred_image[:, :, :, :-1] - pred_image[:, :, :, 1:], 2))
            h_variance = jt.mean(jt.pow(pred_image[:, :, :-1, :] - pred_image[:, :, 1:, :], 2))
            img_tv_loss = args.img_tv_weight * (h_variance + w_variance) / 2.0

            if iteration > first_iter + 400:
                set_geometry_grad(gaussians,False) # True -> Turn off the geo change
                style_img.stop_grad()
                if style_img2 is not None:

                    style_img2_tmp = style_img2.permute(2,0,1).unsqueeze(0)
                    # style_img2.stop_grad()
                    # mask_image = None
                else:
                    style_img2_tmp = None
            # if True:
                loss_dict = nnfm_loss_fn(
                    nn.interpolate(
                        pred_image,
                        size=None,
                        scale_factor=0.5,
                        mode="bilinear",
                    ),
                    style_img.permute(2,0,1).unsqueeze(0),
                    blocks=[
                        # args.vgg_block,
                        2,3,
                    ],
                    loss_names=["nnfm_loss", "content_loss", "spatial_loss"] if not args.preserve_color else ['lum_nnfm_loss','content_loss', "spatial_loss"],
                    contents=nn.interpolate(
                        gt_image,
                        size=None,
                        scale_factor=0.5,
                        mode="bilinear",
                    ),
                    x_mask= mask_image,
                    s_mask= style_mask,
                    styles2=style_img2_tmp,
                )
                # loss_dict_hist = hist_loss_fn.computeLoss(F.interpolate(pred_image,scale_factor=0.5,mode="bilinear"), gram=False)
                # loss_dict['hist_loss'] = loss_dict_hist['hist_loss']
                loss_dict['nnfm_loss' if not args.preserve_color else 'lum_nnfm_loss'] *= 2000
                loss_dict["content_loss"] *= 5
                loss_dict["img_tv_loss"] = img_tv_loss
                loss_dict['depth_loss'] = l1_loss(depth_gt, depth) * 100
                loss_dict['spatial_loss'] *= 20000 # 1e-6 for gram loss 50000 for cos loss
            else:
                # # pre prune oversized gaussians
                # gaussians.prune_over_sized_points(0.5)
                set_geometry_grad(gaussians,True)
                loss_dict = {}
                Ll1 = l1_loss(pred_image, gt_image)
                loss_dict['ddsm_loss'] = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(pred_image, gt_image))
            loss_dict['opacity_regu'] = l1_loss(gaussians._opacity, original_opacity) * 50
            loss_dict['scale_regu'] = l1_loss(gaussians._scaling, original_scale) * 50
        else:
            if viewpoint_cam.mask is not None:
                depth = depth * viewpoint_cam.mask.squeeze()
            out_alpha = render_pkg['alpha']
            gt_image = viewpoint_cam.original_image
            gt_mask = viewpoint_cam.mask
            
            loss_dict = {
                "Ll1": l1_loss(image, gt_image)*(1.0 - opt.lambda_dssim),
                "l_ssim": (1-ssim(image, gt_image))*opt.lambda_dssim,
                # "L_reg": 0,
                # "L_opacity": 0,
                # "L_alpha": 0,
                # "loss_median": 0,
                # "loss_repulsion": 0,
            }

            if opt.lambda_reg > 0 and iteration > opt.smooth_step:
                loss_dict["L_reg"] = (jt.log(out_alpha+1e-20) + jt.log(1 - out_alpha+1e-20)).mean() * opt.lambda_reg
            if opt.lambda_op > 0 and iteration > opt.smooth_step:
                loss_dict["L_opacity"] = (jt.log(gaussians.get_opacity+1e-20) + jt.log(1 - gaussians.get_opacity+1e-20)).mean() * opt.lambda_op
            if opt.lambda_alpha > 0:
                loss_dict["L_alpha"] = l1_loss(out_alpha,gt_mask) * opt.lambda_alpha

            if iteration > opt.smooth_step and (opt.lambda_smooth > 0 or opt.lambda_repulsion > 0):
                partial = gaussians.get_xyz.unsqueeze(0)
                if iteration % 100 ==0 or knn_pos is None:
                    x = gaussians.get_xyz.unsqueeze(0)
                    neighbor_num = 7
                    knn_result = build_knn(x,x, K=neighbor_num)
                alpha_weight = gaussians.get_opacity[knn_result.idx[:,:,1:],:].squeeze().unsqueeze(0).detach()
                knn_pos = knn_result.knn[:,:,1:,:]
                local_frame = None
                if opt.lambda_smooth > 0:
                    loss_median, local_frame = global_median_loss(knn_pos,partial,neighborhood_size = neighbor_num-1, weights = alpha_weight)
                    loss_dict["loss_median"] = opt.lambda_smooth * loss_median
                if opt.lambda_repulsion > 0:
                    if local_frame is None:
                        local_frame = cal_local_normal(knn_pos,neighborhood_size=neighbor_num-1).detach()
                    loss_repulsion, rep_weights = repulsion_loss(knn_pos,partial,h=opt.h, neighborhood_size = neighbor_num-1, alpha_weight = alpha_weight, local_frame = local_frame)
                    loss_dict["loss_repulsion"] = opt.lambda_repulsion * loss_repulsion

        if gaussians.is_gsmesh:
            loss_dict["mesh_restrict_loss"] = mesh_restrict_loss(gaussians.get_scaling,gaussians.vertex1,gaussians.vertex2,gaussians.vertex3,weight=opt.lambda_mrloss)
        loss = sum(list(loss_dict.values()))
        gaussians.optimizer.backward(loss)

        if iteration < opt.densify_until_iter:
            viewspace_point_tensor_grad = gaussians.get_viewspace_point_grad()
        update_flag = False

        with jt.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{3}f}"})
                progress_bar.update(10)
                radio = (gaussians.get_opacity ==1).sum() / gaussians.get_opacity.shape[0]
                loss_print_bar.set_postfix(loss_dict)
                loss_print_bar.update(0)
            if iteration == opt.iterations:
                progress_bar.close()
                loss_print_bar.close()


            # all_loss = {}
            # all_loss['l1'] = Ll1
            # all_loss['udf'] = L_udf
            # all_loss['proj'] = loss_dr_proj
            # all_loss['repel'] = loss_repel
            # all_loss['smooth'] = loss_median
            # all_loss['rep'] = loss_repulsion
            # all_loss['op'] = L_opacity
            # all_loss['depth'] = loss_depth
            # all_loss['reg'] = L_reg
            # all_loss['loss'] = loss
            # Log and save
            # training_report(tb_writer, iteration, all_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, (pipe, background),opt.iterations,test_dataloader)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                def jittormax(a,b):
                    return jt.where(a>b,a,b)
                gaussians.max_radii2D[visibility_filter] = jittormax(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)


                #CHOOSE APPLY WEIGHTS
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < number[args.source_path.split('/')[-1]]:# and iteration % opt.densification_interval == 0:
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 : # and iteration % opt.densification_interval == 0:
                    if(opt.is_apply_weight and viewpoint_cam.segment is not None):
                        print("Apply weights!")
                        selected_pts_mask = jt.zeros_like(gaussians.get_opacity)
                        for mask_cam in train_cam:
                            mask_cam.world_view_transform = mask_cam.world_view_transform
                            mask_cam.full_proj_transform = mask_cam.full_proj_transform
                            mask_cam.camera_center = mask_cam.camera_center

                            render_pkg = render(mask_cam, gaussians, pipe, background,test = False, color_diff = None)
                            image_ = render_pkg["render"]
                            gt_image_ = mask_cam.original_image

                            if mask_cam.mask is not None:
                                total_loss = jt.pow(image_.detach() - gt_image_.detach(),2).sum(0).sqrt().sum() / mask_cam.mask.sum()
                            else:
                                total_loss = jt.pow(image_.detach() - gt_image_.detach(),2).sum(0).sqrt().sum() /image_.shape[1] / image_.shape[2]

                            for i in range(mask_cam.segment.max().long().item()+1):
                                mask = (mask_cam.segment==i).float().contiguous()
                                partial_loss = jt.pow((image_.detach()  - gt_image_.detach()) * mask,2).sum(0).sqrt().sum() / mask.sum()
                                if partial_loss > total_loss:
                                    weights, weights_cnt= gaussians.apply_weights(mask_cam,  mask)
                                    selected_pts_mask = jt.logical_or(selected_pts_mask,weights > opt.weight_th)

                        selected_pts_mask = selected_pts_mask.squeeze()
                    else:
                        selected_pts_mask = None
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent, size_threshold,selected_pts_mask=selected_pts_mask,color_mask = None)
                    update_flag = True

                if iteration == opt.opacity_reset_interval or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

        
            # Optimizer step
        if iteration < opt.iterations:
                if not update_flag:
                    gaussians.optimizer.step()
                gaussians.optimizer.zero_grad()

        if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                jt.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        if iteration % 200 == 0 or iteration in args.save_iterations:
            with jt.no_grad():
                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                pred_image, depth_image, alpha_image = render_pkg["render"], render_pkg['depth'], render_pkg["alpha"]
                os.makedirs(os.path.join(args.model_path,'inter_res'),exist_ok=True)
                # print(viewpoint_cam.mask.shape)
                # print(alpha_image.shape)
                save_image(pred_image, os.path.join(args.model_path,'inter_res', 'rgb_{0:05d}'.format(iteration) + ".png"))
                # save_image(gt_image, os.path.join(args.model_path,'inter_res', 'gt_{0:05d}'.format(iteration) + ".png"))
                # save_image(viewpoint_cam.mask, os.path.join(args.model_path,'inter_res', 'gtmask_{0:05d}'.format(iteration) + ".png"))

    jt.gc()

def prepare_output_and_logger(args,opt, pipe):
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
        # cfg_log_f.write(str(Namespace(**vars(opt))))
        # cfg_log_f.write(str(Namespace(**vars(pipe))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        pass
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, all_loss,elapsed, testing_iterations, scene, renderArgs,all_iterations,test_dataloader):
    if tb_writer:

        for k,v in all_loss.items():
            tb_writer.add_scalar(f'train_loss_patches/{k}', v if type(v) != jt.Tensor else v.item(), iteration)

        tb_writer.add_scalar('iter_time', elapsed, iteration)


    # Report test and samples of training set
    if iteration in testing_iterations or iteration == all_iterations:
        jt.gc()


        l1_test = 0.0
        l_ssim = 0.0
        l_LPIPS = 0.0
        psnr_test = 0.0
        exp_name = tb_writer.log_dir.split('/')[-1]
        if not os.path.exists('static/render/'+exp_name):
            os.mkdir('static/render/'+exp_name)
        test_camera = scene.getTestCameras()

        if scene.scene_type != 'colmap':
            test_camera = scene.getTestCameras()
        else:
            test_camera = test_dataloader
        for idx, viewpoint in enumerate(test_camera):
            viewpoint.world_view_transform = viewpoint.world_view_transform
            viewpoint.full_proj_transform = viewpoint.full_proj_transform
            viewpoint.camera_center = viewpoint.camera_center
            render_result = render(viewpoint, scene.gaussians, *renderArgs, test = True)
            image = jt.clamp(render_result["render"], 0.0, 1.0)
            gt_image = jt.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            image = image.unsqueeze(0)
            gt_image = gt_image.unsqueeze(0)
            l_ssim += ssim(image, gt_image)
            # l_LPIPS += lpips(image, gt_image,net_type='vgg')
            psnr_test += psnr(image, gt_image)


        psnr_test /= len(test_camera)
        l_ssim /= len(test_camera)
        l_LPIPS /= len(test_camera)

        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} ssim {} LPIPS {}".format(iteration, 'test', l1_test, psnr_test, l_ssim, l_LPIPS))

        logging.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {} ssim {} LPIPS {}".format(iteration, 'test', l1_test, psnr_test, l_ssim, l_LPIPS))


        if tb_writer:
            tb_writer.add_scalar('test' + '/loss_viewpoint - l1_loss', l1_test, iteration)
            tb_writer.add_scalar('test' + '/loss_viewpoint - psnr', psnr_test, iteration)
            tb_writer.add_scalar('test' + '/loss_viewpoint - ssim', l_ssim, iteration)
            tb_writer.add_scalar('test' + '/loss_viewpoint - lpips', l_LPIPS, iteration)


        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        jt.gc()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--is_stylized', action='store_true', default=False)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[15000,16000,17000,18000,19000,20000,21000,22000,23000,24000,25000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000,15000,16000,17000,18000,19000,20000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[15000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--point_cloud", type=str, default = None)
    parser.add_argument("--gram_iteration", type=int, default = 30_300)

    #param for stylized param
    # ARF params
    parser.add_argument("--histgram_match", action="store_true", default=False)
    parser.add_argument("--style", type=str, help="path to style image")
    parser.add_argument("--content_weight", type=float, default=5e-3, help="content loss weight")
    # parser.add_argument("--img_tv_weight", type=float, default=1, help="image tv loss weight")
    parser.add_argument(
        "--vgg_block",
        type=int,
        default=2,
        help="vgg block for nnfm extracting feature maps",
    )
    parser.add_argument(
        "--reset_basis_dim",
        type=int,
        default=1,
        help="whether to reset the number of spherical harmonics basis to this specified number",
    )
    parser.add_argument("--preserve_color", action="store_true", default=False)
    parser.add_argument('--second_style', type=str, help="path to second style image")
    parser.add_argument("--no_post_ct", action="store_true", default=False)
    parser.add_argument("--save_image", action="store_true", default=False)
    parser.add_argument("--mask_dir", type=str, default="",help="The directory of 3D masks")
    parser.add_argument('--is_gsmesh', action='store_true', default=False)
    parser.add_argument('--is_applyweights', action='store_true', default=False)

    #param for gsmesh param
    parser.add_argument("--input_mesh", type=str, default ="")
    parser.add_argument("--is_exist_bg",action='store_true', default=False)


    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    if args.is_stylized:
        op = StyleOptimizationParams(parser)
    else:
        op = OptimizationParams(parser)


    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    # if parser.is_stylized:
        
    # else:
    #     op = OptimizationParams(parser)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    random.seed(args.seed)
    np.random.seed(args.seed)



    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.point_cloud, args.is_stylized, args.input_mesh)

    # All done
    print("\nTraining complete.")
