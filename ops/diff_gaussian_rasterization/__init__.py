from typing import NamedTuple
from jittor import nn
import jittor as jt 
from . import rasterize_points

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : jt.Var
    scale_modifier : float
    viewmatrix : jt.Var
    projmatrix : jt.Var
    sh_degree : int
    campos : jt.Var
    prefiltered : bool
    debug : bool




class _RasterizeGaussians(jt.Function):
    
    def save_for_backward(self,*args):
        self.saved_tensors = args
    
    def execute(self,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,):
        
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )
        
        if raster_settings.debug:
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = rasterize_points.RasterizeGaussiansCUDA(*args)
            except Exception as ex:
                jt.save(args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, alpha, radii, geomBuffer, binningBuffer, imgBuffer = rasterize_points.RasterizeGaussiansCUDA(*args)
        self.raster_settings = raster_settings
        self.num_rendered = num_rendered
        self.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha)
        return color, radii, depth, alpha

    # def grad(self,grad_out_color, _):
    def grad(self,grad_out_color, grad_radii, grad_depth, grad_alpha):
        if(grad_out_color is None):
            grad_out_color = jt.zeros((3,grad_depth.shape[1],grad_depth.shape[2]))
        if(grad_alpha is None):
            grad_alpha = jt.zeros((1,grad_out_color.shape[1],grad_out_color.shape[2]))
        if(grad_depth is None):
            grad_depth = jt.zeros((1,grad_out_color.shape[1],grad_out_color.shape[2]))
        num_rendered = self.num_rendered
        raster_settings = self.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer,alpha = self.saved_tensors
        
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_depth,
                grad_alpha,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                alpha,
                raster_settings.debug)
        
        if raster_settings.debug:
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = rasterize_points.RasterizeGaussiansBackwardCUDA(*args)
            except Exception as ex:
                jt.save(args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = rasterize_points.RasterizeGaussiansBackwardCUDA(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )
        del self.saved_tensors
        return grads
    
class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings
        self.rasterizeFunc = _RasterizeGaussians()
        # self.apply_weightsFunc = rasterize_points.ApplyWeightsGaussiansCUDA()

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with jt.no_grad():
            raster_settings = self.raster_settings
            visible = rasterize_points.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible
    
    def execute(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = jt.array([])
        if colors_precomp is None:
            colors_precomp = jt.array([])

        if scales is None:
            scales = jt.array([])
        if rotations is None:
            rotations = jt.array([])
        if cov3D_precomp is None:
            cov3D_precomp = jt.array([])
        
        return self.rasterizeFunc(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )
    
    # def apply_weights(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None,cnt=None,image_weights=None,weights=None):
    def apply_weights(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, image_weights=None):
        assert image_weights is not None
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = jt.array([])
        if colors_precomp is None:
            colors_precomp = jt.array([])

        if scales is None:
            scales = jt.array([])
        if rotations is None:
            rotations = jt.array([])
        if cov3D_precomp is None:
            cov3D_precomp = jt.array([])
        
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            shs,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            image_weights,
            raster_settings.debug
        )
        
        num_rendered, weights, cnt, radii, geomBuffer, binningBuffer, imgBuffer = rasterize_points.ApplyWeightsGaussiansCUDA(*args)
        return weights,cnt

