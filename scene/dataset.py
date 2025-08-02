import jittor as jt
from jittor.dataset import Dataset
from PIL import Image
from utils.general_utils import PILtoJittor
import pickle
import os

# from torchvision import transforms
from jittor import transform
import glob
from scene.cameras import Camera



class ColmapDataset(Dataset):
    def __init__(self, cameras):
        self.cameras = cameras

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        viewpoint_cam = self.cameras[idx]
        out_cam = Camera(viewpoint_cam.colmap_id, viewpoint_cam.R, viewpoint_cam.T, viewpoint_cam.FoVx, viewpoint_cam.FoVy, viewpoint_cam.original_image, viewpoint_cam.gt_alpha_mask,viewpoint_cam.mask,viewpoint_cam.segment,
                 viewpoint_cam.image_name, viewpoint_cam.uid,viewpoint_cam.image_path,viewpoint_cam.width,viewpoint_cam.height,viewpoint_cam.resolution,
                 trans=viewpoint_cam.trans, scale=viewpoint_cam.scale, data_device = "cuda")

        if viewpoint_cam.original_image is None:
            image_path = glob.glob(viewpoint_cam.image_path + '/'+ viewpoint_cam.image_name + '.*')
            temp_image = Image.open(image_path[0])
            resized_image_rgb = PILtoJittor(temp_image, viewpoint_cam.resolution)
            original_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)#.to(viewpoint_cam.data_device)

            
            # segment = []
            if os.path.exists(os.path.join(viewpoint_cam.image_path, "../segment/SEG_"+viewpoint_cam.image_name+'.pkl')):
                with open(os.path.join(viewpoint_cam.image_path, "../segment/SEG_"+viewpoint_cam.image_name+'.pkl'),'rb') as F:
                    masks = pickle.load(F)
            # elif os.path.exists(os.path.join(viewpoint_cam.image_path, "../segment/SEG_"+viewpoint_cam.image_name+'.pt')):
            #     with open(os.path.join(viewpoint_cam.image_path, "../segment/SEG_"+viewpoint_cam.image_name+'.pt'),'rb') as F:
            #         masks = torch.load(F)

            # for i in range(1,masks.max().long().item()+1):
            #     mask = (masks==i).float()
            #     segment.append(mask)
        out_cam.original_image = original_image
        out_cam.segment = masks[:,:-1,:-1]
        return out_cam