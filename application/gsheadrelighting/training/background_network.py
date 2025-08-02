import jittor as jt
from jittor import nn

import numpy as np
from collections import OrderedDict
from jittorutils import misc
from jittorutils import persistence

from training.point_generator import ModulatedFullyConnectedLayer, LFF, PointUpsample_subpixel, get_scaled_directional_vector_from_quaternion, PointUpsample_modsubpixel
from training.networks_stylegan2 import FullyConnectedLayer

def inverse_sigmoid(x):
    return jt.log(x/(1-x))

# Very similar to point generator (w/ upsample)
@persistence.persistent_class
class PointBgGenerator(jt.nn.Module):
    def __init__(self,
            w_dim,                      # Intermediate latent (W) dimensionality.
            img_channels=3, 
            in_channels=128,
            hidden_channels=128,
            radius=3,
            num_upsample=1,
            upsample_ratio=4,
            options = {},
        ):
        super().__init__()
        
        self.num_upsample = num_upsample
        self.upsample_ratio = upsample_ratio
        
        self.conv_in = LFF(in_channels)
        
        weight_init = 1e-2
        _out_keys = {"xyz": [3, 1.0], 
                     "scale": [3, weight_init], 
                     "rotation": [4, weight_init], 
                     "color": [img_channels, 1], 
                     "opacity": [1, 1]}
        
        # self.network = nn.ModuleDict()
        # self.out_layers = nn.ModuleDict()
        # self.upsample_layer = nn.ModuleDict()
        network = OrderedDict()
        out_layers = OrderedDict()
        upsample_layer = OrderedDict()
        
        for i in range(num_upsample + 1):
            cur_block = f"block_{i}"
            
            network[cur_block] = nn.ModuleList()
            if i > 0:
                network[cur_block].append(PointUpsample_modsubpixel(in_channels, in_channels, pe_dim=0, upsample_ratio=upsample_ratio, resolution=512, w_dim=512))
            
            network[cur_block].append(ModulatedFullyConnectedLayer(in_channels, hidden_channels, w_dim=w_dim, activation="lrelu", residual=True))
            network[cur_block].append(ModulatedFullyConnectedLayer(hidden_channels, hidden_channels, w_dim=w_dim, activation="lrelu", residual=True))
            
            # self.out_layers[cur_block] = nn.ModuleDict()
            out_layers[cur_block] = OrderedDict()
            for k in _out_keys.keys():
                out_layers[cur_block][k] = OrderedDict()
                # self.out_layers[cur_block][k] = nn.ModuleDict()
            for k in _out_keys.keys():
                out_layers[cur_block][k][f"out_block"] = ModulatedFullyConnectedLayer(hidden_channels, _out_keys[k][0], w_dim=w_dim, activation="linear", weight_init=_out_keys[k][1])
            for k in _out_keys.keys():
                out_layers[cur_block][k] = nn.ModuleList(out_layers[cur_block][k])
            out_layers[cur_block] = nn.ModuleList(out_layers[cur_block])
            in_channels = hidden_channels
        self.network = nn.ModuleList(network)
        self.out_layers = nn.ModuleList(out_layers)
        self.upsample_layer = nn.ModuleList(upsample_layer)
        # self.register_buffer("scale_init", jt.ones([1, 3]) * options['scale_init_bg'])
        # self.register_buffer("scale_threshold", jt.ones([1, 3]) * options['scale_threshold_bg'])
        # self.register_buffer("rotation_init", jt.tensor([[1, 0, 0, 0]]))
        # self.register_buffer("color_init", jt.tensor(jt.zeros([1, img_channels])))
        # self.register_buffer("opacity_init", inverse_sigmoid(0.1 * jt.ones([1, 1])))
        self.scale_init = jt.ones([1, 3]).stop_grad() * options['scale_init_bg']
        self.scale_threshold = jt.ones([1, 3]).stop_grad() * options['scale_threshold_bg']
        self.rotation_init = jt.array([[1, 0, 0, 0]]).stop_grad()
        self.color_init = jt.zeros([1, img_channels]).stop_grad()
        self.opacity_init = inverse_sigmoid(0.1 * jt.ones([1, 1])).stop_grad()

        self.xyz_output_scale = options['xyz_output_scale'] # default: 0.1
        self.radius = radius
        self.color_output_scale = 1.
        
        self.split_ratio = 2 ** (np.log2(upsample_ratio) / 3)
        
    def execute(self, x, w):
        # x: [B, L, C]
        B, L, C = x.shape
        
        xyz = x
        x = self.conv_in(x.reshape(-1, C)).reshape([B, L, -1])
        
        for i in range(self.num_upsample + 1):
            cur_block = f"block_{i}"
            
            if i == 0:
                layer1, layer2 = self.network[cur_block]
            else:
                layer_up, layer1, layer2 = self.network[cur_block]
                x, (xyz_prev, scale_prev, rotation_prev, color_prev, opacity_prev) = layer_up(x, [xyz_prev, scale_prev, rotation_prev, color_prev, opacity_prev], w=w)
                
            x = layer1(x, w, demodulate=True)
            x = layer2(x, w, demodulate=True)
        
            if i == 0:
                xyz_cur = xyz # do not move initial anchor
                scale_cur = self.out_layers[cur_block]["scale"][f"out_block"](x, w, demodulate=False) + self.scale_init
                rotation_cur = self.out_layers[cur_block]["rotation"][f"out_block"](x, w, demodulate=False) + self.rotation_init
                color_cur = self.out_layers[cur_block]["color"][f"out_block"](x, w, demodulate=False) + self.color_init
                opacity_cur = self.out_layers[cur_block]["opacity"][f"out_block"](x, w, demodulate=False) + self.opacity_init

                scale_cur = - jt.nn.softplus(- (scale_cur - self.scale_threshold)) + self.scale_threshold
                
                # xyz on sphere
                xyz_cur = jt.normalize(xyz_cur, dim=-1) * self.radius
                
                xyz, scale, rotation, color, opacity = xyz_cur, scale_cur, rotation_cur, color_cur, opacity_cur
                
            else:
                pc = [self.out_layers[cur_block][k][f"out_block"](x, w, demodulate=False) for k in ["xyz", "scale", "rotation", "color", "opacity"]]
                pc_prev = [xyz_prev, scale_prev, rotation_prev, color_prev, opacity_prev]
                
                xyz_cur, scale_cur, rotation_cur, color_cur, opacity_cur = self.postprocessing_block(pc, pc_prev, \
                                                                            N=self.upsample_ratio, split_ratio=self.split_ratio)
                # xyz on sphere
                xyz_cur = jt.normalize(xyz_cur, dim=-1) * self.radius
                
                xyz = jt.cat([xyz, xyz_cur], dim=1)
                scale = jt.cat([scale, scale_cur], dim=1)
                rotation = jt.cat([rotation, rotation_cur], dim=1)
                color = jt.cat([color, color_cur], dim=1)
                opacity = jt.cat([opacity,  opacity_cur], dim=1)
                
                
            xyz_prev, scale_prev, rotation_prev, color_prev, opacity_prev = xyz_cur, scale_cur, rotation_cur, color_cur, opacity_cur
        
        xyz = jt.normalize(xyz, dim=-1) * self.radius
        
        return xyz, scale, rotation, color, opacity
    

    def postprocessing_block(self, pc, pc_prev, percent_dense=0.05, N=2, split_ratio=None, scale_upsample_reg=0):
        xyz, scale, rotation, color, opacity = pc_prev # previous outputs
        xyz_cur, scale_cur, rotation_cur, color_cur, opacity_cur = pc # current outputs
    
        rotation_new = rotation + rotation_cur
        color_new = color + color_cur * self.color_output_scale
        opacity_new = opacity + opacity_cur

        samples = jt.tanh(xyz_cur) # [B, L * N, 3]
        R = get_scaled_directional_vector_from_quaternion(rotation, scale) # [B, L * N, 3, 3]
        
        xyz_new = (R @ samples.unsqueeze(-1)).squeeze(-1) + xyz # [B, L * N, 3]
        
        # split and clone w/ prev scale
        if split_ratio is None:
            scale_split = jt.log(jt.exp(scale) / (0.8 * N)) # default setting of 3DGS
        else:
            scale_split = jt.log(jt.exp(scale) / split_ratio)
        
        # for remove -inf
        scale_split = jt.clamp(scale_split, -1e+2, 0)
        
        scale_max = jt.exp(scale).max(-1, keepdim=True)[0]
        split_idx = jt.where(scale_max > percent_dense, 1., 0.)
        
        scale = scale * (1. - split_idx) + scale_split * split_idx
        
        scale_new = scale + -1 * jt.nn.softplus(- (scale_cur - scale_upsample_reg)) + scale_upsample_reg
        
        # remove outliers
        scale_new = jt.clamp(scale_new, -10, 0)
        
        pc_upsampled = [xyz_new, scale_new, rotation_new, color_new, opacity_new]
        return pc_upsampled
    
    