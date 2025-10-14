import os 
import jittor as jt
from jittor import Function, exp, log
from typing import Tuple



header_path = os.path.join(os.path.dirname(__file__), 'cuda_rasterizer')
glm_path = os.path.join(os.path.dirname(__file__),'third_party','glm')
proj_options = {f'FLAGS: -I"{header_path}" -I"{glm_path}" -l"CudaRasterizer" -L"{os.path.dirname(__file__)}"':1}
# proj_options = {f'FLAGS: -I"./cuda_rasterizer" -I"./third_paart/glm" -l"CudaRasterizer" -L"./"':1}
jt.flags.use_cuda = 1
cuda_header = """
#include <math.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
//#include "config.h"
//#include "rasterizer.h"
#include "rasterizer_impl.h"

#include <fstream>
#include <string>
#include <functional>
std::function<char*(size_t N)> resizeFunctional(jittor::Var*& t) {
    auto lambda = [&t](size_t N) {
        t->set_shape({(long long)N});
		return reinterpret_cast<char*>(t->ptr<char>());
        //return t->ptr<char>();
    };
    return lambda;
}
"""

def mark_visible(means3D,viewmatrix,projmatrix):
    present = jt.zeros([means3D.shape[0]],dtype=jt.bool)
    out = jt.code(inputs=[means3D,viewmatrix,projmatrix],outputs=[present]
                  ,cuda_header=cuda_header,cuda_src='''
        @alias(means3D, in0)
        @alias(viewmatrix, in1)
        @alias(projmatrix,in2)
        @alias(present,out0)
        const int P = means3D_shape0;
        if(P != 0)
        {
            CudaRasterizer::Rasterizer::markVisible(P,
                means3D_p,
                viewmatrix_p,
                projmatrix_p,
                present_p;
        }
    ''')
    out.compile_options = proj_options
    return out

def bool_value(t):
    return 'true' if t else 'false' 

def compute_buffer_size(means3D,image_width,image_height):
    P = means3D.size(0)
    # print(type(means3D))
    geom_size = jt.zeros([1,],'int64')
    img_size = jt.zeros([1,],'int64')
    
    geom_size,img_size = jt.code(outputs=[geom_size,img_size],inputs=[means3D],
        cuda_header=cuda_header,cuda_src=f'''
        @alias(geom_size, out0)
        @alias(img_size, out1)
        
        const int P = in0_shape0;
        size_t a = CudaRasterizer::required<CudaRasterizer::GeometryState>(P);
        cudaMemcpy(geom_size->ptr<size_t>(),&a,sizeof(size_t),cudaMemcpyHostToDevice);
        a = CudaRasterizer::required<CudaRasterizer::ImageState>({image_width} * {image_height});
        cudaMemcpy(img_size->ptr<size_t>(),&a,sizeof(size_t),cudaMemcpyHostToDevice);        
        //  a = CudaRasterizer::required<CudaRasterizer::BinningState>(P * 16);      
        //  cudaMemcpy(binning_size->ptr<size_t>(),&a,sizeof(size_t),cudaMemcpyHostToDevice);        
        
    ''')
    geom_size.compile_options = proj_options
    # geom_size.sync()
    
    # print(geom_size)
    return geom_size[0].item(),img_size[0].item()

def RasterizeGaussiansCUDA(
    background:jt.Var,
    means3D:jt.Var,
    colors:jt.Var,
    opacity:jt.Var,
    scales:jt.Var,
    rotations:jt.Var,
    scale_modifier:float,
    cov3D_precomp:jt.Var,
    viewmatrix:jt.Var,
    projmatrix:jt.Var,
    tan_fovx:float,
    tan_fovy:float,
    image_height:int,
    image_width:int,
    sh:jt.Var,
    degree:int,
    campos:jt.Var,
    prefiltered:bool,
    debug:bool
) -> Tuple[int, jt.Var, jt.Var, jt.Var, jt.Var, jt.Var, jt.Var, jt.Var]:
    geom_size,img_size = compute_buffer_size(means3D,image_width,image_height)
    # geom_size = 80 * means3D.size(0)
    # img_size = 17  * image_width * image_height
    # binning_size = 37 * means3D.size(0) * 64

    
    
    with jt.flag_scope(compile_options=proj_options):
        geomBuffer = jt.array(jt.zeros([geom_size],dtype='uint8'))
        rendered = jt.array(jt.zeros([1],dtype='int32'))
        binning_size = jt.array(jt.zeros([1],dtype='int64'))
        radii = jt.array(jt.zeros([means3D.size(0)],dtype='int32'))
        
        rendered,binning_size,radii = jt.code(
            outputs=[rendered,binning_size,radii],inputs=[background,means3D,colors,opacity,
                                  scales,rotations,cov3D_precomp,
                                  viewmatrix,projmatrix,sh,campos,geomBuffer],
            data = {
                'image_height':image_height,
                'image_width':image_width,
                'scale_modifier':scale_modifier,
                'tan_fovx':tan_fovx,
                'tan_fovy':tan_fovy,  
                'degree':degree ,
            },
            cuda_header=cuda_header,
            cuda_src=f'''
                @alias(background, in0)
                @alias(means3D, in1)
                @alias(colors, in2)
                @alias(opacity, in3)
                @alias(scales, in4)
                @alias(rotations, in5)
                @alias(cov3D_precomp, in6)
                @alias(viewmatrix, in7)
                @alias(projmatrix, in8)
                @alias(sh, in9)
                @alias(campos, in10)
                @alias(geomBuffer, in11)
                
                @alias(rendered, out0)
                @alias(binning_size, out1)
                @alias(radii, out2)
                
                const int P = means3D_shape0;
                const int H = data["image_height"];
                const int W = data["image_width"];
                if(P != 0){{
                    int M = 0;
                    if(sh_shape0 != 0)
                    {{
                        M = sh->shape[1];
                    }}
                    else
                        sh_p = nullptr;
                    if(colors_shape0 == 0) colors_p = nullptr;
                    if(cov3D_precomp_shape0 == 0) cov3D_precomp_p = nullptr;
                    int num_rendered = CudaRasterizer::Rasterizer::forward_0(
                        geomBuffer->ptr<char>(),
                        P, data["degree"], M,
                        background_p,
                        W, H,
                        means3D_p,
                        sh_p,
                        colors_p,
                        opacity_p,
                        scales_p,
                        data["scale_modifier"],
                        rotations_p,
                        cov3D_precomp_p,
                        viewmatrix_p,
                        projmatrix_p,
                        campos_p,
                        data["tan_fovx"],
                        data["tan_fovy"],
                        {bool_value(prefiltered)},
                        
                        radii_p,
                        {bool_value(debug)});
                    cudaMemcpy(rendered->ptr<int>(),&num_rendered,sizeof(int),cudaMemcpyHostToDevice);
                    size_t a = CudaRasterizer::required<CudaRasterizer::BinningState>(num_rendered);   
                    cudaMemcpy(binning_size->ptr<size_t>(),&a,sizeof(size_t),cudaMemcpyHostToDevice);
                }}
            '''
        )
        
        binningBuffer = jt.array(jt.zeros([binning_size[0].item()],dtype='uint8'))
        imageBuffer = jt.array(jt.zeros([img_size],dtype='uint8'))
        out_color = jt.array(jt.zeros([3,image_height,image_width],dtype='float32'))
        out_depth = jt.array(jt.zeros([1,image_height,image_width],dtype='float32'))
        out_alpha = jt.array(jt.zeros([1,image_height,image_width],dtype='float32'))
        
        binningBuffer,imageBuffer,out_color,out_depth,out_alpha = jt.code(
            outputs=[binningBuffer ,imageBuffer ,out_color ,out_depth, out_alpha],
            inputs=[background,means3D,colors,opacity,scales,rotations,
            cov3D_precomp,viewmatrix,projmatrix,sh,campos,geomBuffer,radii],
            data = {
                'image_height':image_height,
                'image_width':image_width,
                'scale_modifier':scale_modifier,
                'tan_fovx':tan_fovx,
                'tan_fovy':tan_fovy,  
                'degree':degree ,
                'num_rendered':rendered[0].item(),
            },
            cuda_header=cuda_header,
            cuda_src=f'''
                @alias(background, in0)
                @alias(means3D, in1)
                @alias(colors, in2)
                @alias(opacity, in3)
                @alias(scales, in4)
                @alias(rotations, in5)
                @alias(cov3D_precomp, in6)
                @alias(viewmatrix, in7)
                @alias(projmatrix, in8)
                @alias(sh, in9)
                @alias(campos, in10)
                @alias(geomBuffer, in11)
                @alias(radii, in12)
                
                @alias(binningBuffer, out0)
                @alias(imageBuffer, out1)
                @alias(out_color, out2)
                @alias(out_depth, out3)
                @alias(out_alpha, out4)
                const int P = means3D_shape0;
                const int H = data["image_height"];
                const int W = data["image_width"];
                
                if(P != 0){{
                    int M = 0;
                    if(sh_shape0 != 0)
                    {{
                        M = sh->shape[1];
                    }}
                    else
                        sh_p = nullptr;
                    if(radii_shape0 == 0) radii_p = nullptr;
                    if(colors_shape0 == 0) colors_p = nullptr;
                    if(cov3D_precomp_shape0 == 0) cov3D_precomp_p = nullptr;
                    CudaRasterizer::Rasterizer::forward_1(
                        geomBuffer->ptr<char>(),    
                        binningBuffer->ptr<char>(),
                        imageBuffer->ptr<char>(),
                        P, data["degree"], M, data["num_rendered"],
                        background_p,
                        W, H,
                        means3D_p,
                        sh_p,
                        colors_p,
                        opacity_p,
                        scales_p,
                        data["scale_modifier"],
                        rotations_p,
                        cov3D_precomp_p,
                        viewmatrix_p,
                        projmatrix_p,
                        campos_p,
                        data["tan_fovx"],
                        data["tan_fovy"],
                        {bool_value(prefiltered)},
                        out_color_p,
                        out_depth_p,
                        out_alpha_p,
                        radii_p,
                        {bool_value(debug)}
                    );
                }}
            '''
        )
    # geomBuffer.sync()
    geomBuffer = geomBuffer.detach()
    binningBuffer = binningBuffer.detach()
    imageBuffer = imageBuffer.detach()
    return rendered[0].item(), out_color, out_depth, out_alpha ,radii, geomBuffer, binningBuffer, imageBuffer

def RasterizeGaussiansBackwardCUDA(
    background:jt.Var,
    means3D:jt.Var,
    radii:jt.Var,
    colors:jt.Var,
    scales:jt.Var,
    rotations:jt.Var,
    scale_modifier:float,
    cov3D_precomp:jt.Var,
    viewmatrix:jt.Var,
    projmatrix:jt.Var,
    tan_fovx:float,
    tan_fovy:float,
    dL_dout_color:jt.Var,
    dL_dout_depth:jt.Var,
    dL_dout_alpha:jt.Var,
    sh:jt.Var,
    degree:int,
    campos:jt.Var,
    geomBuffer:jt.Var,
    R:int, 
    binningBuffer:jt.Var,
    imageBuffer:jt.Var,
    alpha:jt.Var,
    debug:bool
) -> Tuple[jt.Var, jt.Var, jt.Var, jt.Var, jt.Var, jt.Var, jt.Var, jt.Var]:
    
    P = means3D.size(0)
    M = sh.size(1) if sh.size(0)!= 0 else 0
    dL_dmeans3D =jt.zeros([P,3],dtype='float32')
    dL_dmeans2D =jt.zeros([P,3],dtype='float32')
    dL_dcolors = jt.zeros([P,3],dtype='float32')
    dL_ddepth= jt.zeros([P,1],dtype='float32')
    dL_dopacity = jt.zeros([P,1],dtype='float32')
    dL_dcov3D = jt.zeros([P,6],dtype='float32')
    dL_dsh = jt.zeros([P,M,3],dtype='float32')
    dL_dscales = jt.zeros([P,3],dtype='float32') 
    dL_drotations = jt.zeros([P,4],dtype='float32')
    dL_dconic = jt.zeros([P,2,2],dtype='float32')
    # for i in [background,means3D,radii,colors,scales,rotations,
    #                 cov3D_precomp,viewmatrix,projmatrix,dL_dout_color,dL_dout_depth,dL_dout_alpha,sh,campos,geomBuffer,binningBuffer,imageBuffer]:
    #     print(i.shape)
    with jt.flag_scope(compile_options=proj_options):
        dL_dmeans2D, dL_dcolors, dL_ddepth, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations,dL_dconic = jt.code(
            inputs=[background,means3D,radii,colors,scales,rotations,
                    cov3D_precomp,viewmatrix,projmatrix,dL_dout_color,dL_dout_depth,dL_dout_alpha,sh,campos,geomBuffer,binningBuffer,imageBuffer,alpha],
            outputs=[dL_dmeans2D, dL_dcolors, dL_ddepth, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations,dL_dconic],
            data={
                    'R':R,
                    'scale_modifier':scale_modifier,
                    'tan_fovx':tan_fovx,
                    'tan_fovy':tan_fovy,
                    'degree':degree,
                },
            cuda_header=cuda_header,
            cuda_src=f'''
                @alias(background, in0)
                @alias(means3D, in1)
                @alias(radii, in2)
                @alias(colors, in3)
                @alias(scales, in4)
                @alias(rotations, in5)
                @alias(cov3D_precomp, in6)
                @alias(viewmatrix, in7)
                @alias(projmatrix, in8)
                @alias(dL_dout_color, in9)
                @alias(dL_dout_depth, in10)
                @alias(dL_dout_alpha, in11)
                @alias(sh, in12)
                @alias(campos, in13)
                @alias(geomBuffer, in14)
                @alias(binningBuffer, in15)
                @alias(imageBuffer, in16)
                @alias(alpha, in17)

                @alias(dL_dmeans2D, out0)
                @alias(dL_dcolors, out1)
                @alias(dL_ddepth, out2)
                @alias(dL_dopacity, out3)
                @alias(dL_dmeans3D, out4)
                @alias(dL_dcov3D, out5)
                @alias(dL_dsh, out6)
                @alias(dL_dscales, out7)
                @alias(dL_drotations, out8)
                @alias(dL_dconic, out9)

                const int P = means3D_shape0;
                const int H = dL_dout_color_shape1;
                const int W = dL_dout_color_shape2;
                int M = 0;
                if(sh_shape0 != 0){{
                    M = sh_shape1;
                }}
                
                
                
                if(P != 0){{
                    if(radii_shape0 == 0) radii_p = nullptr;
                    if(colors_shape0 == 0) colors_p = nullptr;
                    if(cov3D_precomp_shape0 == 0) cov3D_precomp_p = nullptr;
                    
                    CudaRasterizer::Rasterizer::backward(P, data["degree"], M, data["R"],
                    background_p,
                    W, H, 
                    means3D_p,
                    sh_p,
                    colors_p,
                    alpha_p,
                    scales_p,
                    data["scale_modifier"],
                    rotations_p,
                    cov3D_precomp_p,
                    viewmatrix_p,
                    projmatrix_p,
                    campos_p,
                    data["tan_fovx"],
                    data["tan_fovy"],
                    radii_p,
                    geomBuffer->ptr<char>(),
                    binningBuffer->ptr<char>(),
                    imageBuffer->ptr<char>(),
                    dL_dout_color_p,
                    dL_dout_depth_p,
                    dL_dout_alpha_p,
                    dL_dmeans2D_p,
                    dL_dconic_p,  
                    dL_dopacity_p,
                    dL_dcolors_p,
                    dL_ddepth_p,
                    dL_dmeans3D_p,
                    dL_dcov3D_p,
                    dL_dsh_p,
                    dL_dscales_p,
                    dL_drotations_p,
                    {bool_value(debug)});
                }}
            '''
        )
    # dL_dmeans2D.compile_options = proj_options
    return dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations

#extra input: image_weights  output: weights, cnt
def ApplyWeightsGaussiansCUDA(
    background:jt.Var,
    means3D:jt.Var,
    colors:jt.Var,
    opacity:jt.Var,
    scales:jt.Var,
    rotations:jt.Var,
    scale_modifier:float,
    cov3D_precomp:jt.Var,
    viewmatrix:jt.Var,
    projmatrix:jt.Var,
    tan_fovx:float,
    tan_fovy:float,
    image_height:int,
    image_width:int,
    sh:jt.Var,
    degree:int,
    campos:jt.Var,
    prefiltered:bool,
    image_weights:jt.Var,
    debug:bool
) -> Tuple[int, jt.Var, jt.Var, jt.Var, jt.Var, jt.Var, jt.Var]:
    geom_size,img_size = compute_buffer_size(means3D,image_width,image_height)
    # geom_size = 80 * means3D.size(0)
    # img_size = 17  * image_width * image_height
    # binning_size = 37 * means3D.size(0) * 64

    
    
    with jt.flag_scope(compile_options=proj_options):
        geomBuffer = jt.array(jt.zeros([geom_size],dtype='uint8'))
        rendered = jt.array(jt.zeros([1],dtype='int32'))
        binning_size = jt.array(jt.zeros([1],dtype='int64'))
        radii = jt.array(jt.zeros([means3D.size(0)],dtype='int32'))
        
        rendered,binning_size,radii = jt.code(
            outputs=[rendered,binning_size,radii],inputs=[background,means3D,colors,opacity,
                                  scales,rotations,cov3D_precomp,
                                  viewmatrix,projmatrix,sh,campos,geomBuffer],
            data = {
                'image_height':image_height,
                'image_width':image_width,
                'scale_modifier':scale_modifier,
                'tan_fovx':tan_fovx,
                'tan_fovy':tan_fovy,  
                'degree':degree ,
            },
            cuda_header=cuda_header,
            cuda_src=f'''
                @alias(background, in0)
                @alias(means3D, in1)
                @alias(colors, in2)
                @alias(opacity, in3)
                @alias(scales, in4)
                @alias(rotations, in5)
                @alias(cov3D_precomp, in6)
                @alias(viewmatrix, in7)
                @alias(projmatrix, in8)
                @alias(sh, in9)
                @alias(campos, in10)
                @alias(geomBuffer, in11)

                
                @alias(rendered, out0)
                @alias(binning_size, out1)
                @alias(radii, out2)
                
                const int P = means3D_shape0;
                const int H = data["image_height"];
                const int W = data["image_width"];
                if(P != 0){{
                    int M = 0;
                    if(sh_shape0 != 0)
                    {{
                        M = sh_shape1;
                    }}
                    if(colors_shape0 == 0) colors_p = nullptr;
                    if(cov3D_precomp_shape0 == 0) cov3D_precomp_p = nullptr;
                    int num_rendered = CudaRasterizer::Rasterizer::apply_weights_0(
                        geomBuffer->ptr<char>(),
                        P, data["degree"], M,
                        background_p,
                        W, H,
                        means3D_p,
                        sh_p,
                        colors_p,
                        opacity_p,
                        scales_p,
                        data["scale_modifier"],
                        rotations_p,
                        cov3D_precomp_p,
                        viewmatrix_p,
                        projmatrix_p,
                        campos_p,
                        data["tan_fovx"],
                        data["tan_fovy"],
                        {bool_value(prefiltered)},
                        radii_p,
                        {bool_value(debug)});
                    cudaMemcpy(rendered->ptr<int>(),&num_rendered,sizeof(int),cudaMemcpyHostToDevice);
                    size_t a = CudaRasterizer::required<CudaRasterizer::BinningState>(num_rendered);   
                    cudaMemcpy(binning_size->ptr<size_t>(),&a,sizeof(size_t),cudaMemcpyHostToDevice);
                }}
            '''
        )
        
        binningBuffer = jt.array(jt.zeros([binning_size[0].item()],dtype='uint8'))
        imageBuffer = jt.array(jt.zeros([img_size],dtype='uint8'))
        # out_color = jt.array(jt.zeros([3,image_height,image_width],dtype='float32'))
        # out_depth = jt.array(jt.zeros([1,image_height,image_width],dtype='float32'))
        # out_alpha = jt.array(jt.zeros([1,image_height,image_width],dtype='float32'))
        weights = jt.array(jt.zeros([means3D.shape[0],1],dtype=jt.float))
        cnt = jt.array(jt.zeros([means3D.shape[0],1],dtype=jt.int))
        binningBuffer,imageBuffer,weights,cnt = jt.code(
            outputs=[binningBuffer,imageBuffer,weights,cnt],
            inputs=[background,means3D,colors,opacity,scales,rotations,
            cov3D_precomp,viewmatrix,projmatrix,sh,campos,geomBuffer,radii,image_weights],
            data = {
                'image_height':image_height,
                'image_width':image_width,
                'scale_modifier':scale_modifier,
                'tan_fovx':tan_fovx,
                'tan_fovy':tan_fovy,  
                'degree':degree ,
                'num_rendered':rendered[0].item(),
                'num_channels':image_weights.shape[0]
            },
            cuda_header=cuda_header,
            cuda_src=f'''
                @alias(background, in0)
                @alias(means3D, in1)
                @alias(colors, in2)
                @alias(opacity, in3)
                @alias(scales, in4)
                @alias(rotations, in5)
                @alias(cov3D_precomp, in6)
                @alias(viewmatrix, in7)
                @alias(projmatrix, in8)
                @alias(sh, in9)
                @alias(campos, in10)
                @alias(geomBuffer, in11)
                @alias(radii, in12)
                @alias(image_weights, in13)

                @alias(binningBuffer, out0)
                @alias(imageBuffer, out1)
                @alias(weights, out2)
                @alias(cnt, out3)

                const int P = means3D_shape0;
                const int H = data["image_height"];
                const int W = data["image_width"];
                
                if(P != 0){{
                    int M = 0;
                    if(sh_shape0 != 0)
                    {{
                        M = sh_shape1;
                    }}
                    if(radii_shape0 == 0) radii_p = nullptr;
                    if(colors_shape0 == 0) colors_p = nullptr;
                    if(cov3D_precomp_shape0 == 0) cov3D_precomp_p = nullptr;
                    CudaRasterizer::Rasterizer::apply_weights_1(
                        geomBuffer->ptr<char>(),    
                        binningBuffer->ptr<char>(),
                        imageBuffer->ptr<char>(),
                        P, data["degree"], M, data["num_rendered"],
                        background_p,
                        W, H,
                        means3D_p,
                        sh_p,
                        weights_p,
                        opacity_p,
                        scales_p,
                        data["scale_modifier"],
                        rotations_p,
                        cov3D_precomp_p,
                        viewmatrix_p,
                        projmatrix_p,
                        campos_p,
                        data["tan_fovx"],
                        data["tan_fovy"],
                        {bool_value(prefiltered)},
                        image_weights_p,
                        radii_p,
                        cnt_p,
                        data["num_channels"],
                        {bool_value(debug)}
                    );
                }}
            '''
        )
        # print(weights)
    # geomBuffer.sync()
    geomBuffer = geomBuffer.detach()
    binningBuffer = binningBuffer.detach()
    imageBuffer = imageBuffer.detach()
    return rendered[0].item(), weights, cnt ,radii, geomBuffer, binningBuffer, imageBuffer
