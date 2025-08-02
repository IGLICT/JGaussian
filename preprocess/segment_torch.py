import numpy as np
import cv2
import os
import pickle as pkl
import glob
import torch
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import cv2
def calculate_max_overlap(masks):
    # 初始化合并mask，像素值全为0
    merged_mask = np.zeros_like(masks[0])
    # 将每个mask的像素合并到合并mask中
    for mask in masks:
        merged_mask += mask
    # 统计合并mask中像素值等于n的像素的数量
    max_overlap = np.max(merged_mask)
    return max_overlap, merged_mask
from semantic_sam import prepare_image, plot_results, build_semantic_sam, SemanticSamAutomaticMaskGenerator


sam_checkpoint = "../sam_hq_vit_h.pth"
model_type = "vit_h"
device = "cuda"


mask_generator = SemanticSamAutomaticMaskGenerator(build_semantic_sam(model_type='L', ckpt='./ckps/swinl_only_sam_many2many.pth'), pred_iou_thresh = 0.88,min_mask_region_area=50) # model_type: 'L' / 'T', depends on your checkpint
total_len = 0
codelist = []
dataname = glob.glob('../llff/*')
# dataname = ['../nerf_synthetic/lego']
print(dataname)

def ssam_gen(original_image,input_image,bias = 1):
    masks = mask_generator.generate(input_image[:-1,:,:])
    # return
    # masks = samhq_mask_generator.generate(original_image[:,:,:3].astype(np.uint8))
    masks = sorted(masks, key=(lambda x: x['area']), reverse=False)
    masked = torch.zeros((1,input_image.shape[1], input_image.shape[2]))
    result = torch.zeros((1,input_image.shape[1], input_image.shape[2]))
    for i in range(len(masks)):
        seg = torch.from_numpy(masks[i]['segmentation'])
        seg = torch.where(seg, torch.ones_like(seg).float(), torch.zeros_like(seg).float())[None,:,:]
        seg = torch.clamp((seg - masked),0,1)
        result += seg * (i+bias)
        masked += seg
    return result


for name in dataname:
    if not os.path.exists(name+'/segment'):
        os.mkdir(name+'/segment')

    image_data = glob.glob(name+'/images_4/*')

    for imagepath in image_data:
        print(imagepath)
        imgfile = imagepath
        img_name = imagepath.split('/')[-1]

        if os.path.exists(name+"/segment/SEG_"+img_name[:-4]+".pkl"):
            continue


        original_image, input_image = prepare_image(image_pth=imgfile)


        result = ssam_gen(original_image,input_image)

        temp = result.squeeze()
        colored_pattern = plt.get_cmap('hsv')(temp / temp.max())
        colored_pattern[:,:,-1] = input_image[-1,:,:].cpu().numpy() / 255
        colored_pattern[(result==0).squeeze(),:-1] = 0


        cv2.imwrite(name+"/segment/mask_"+img_name[:-4]+".png",(colored_pattern * 255).astype(np.uint8))
        colored_pattern = colored_pattern * 0.5 + original_image / 255 * 0.5
        cv2.imwrite(name+"/segment/mix_"+img_name[:-4]+".png",(colored_pattern * 255).astype(np.uint8))

        with open(name+"/segment/SEG_"+img_name[:-4]+".pkl", "wb") as f:

            pkl.dump(result, f, protocol=pkl.HIGHEST_PROTOCOL)
    