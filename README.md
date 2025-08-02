<div align="center">
<h1>JGaussian</h1>
    <p>Institute of Computing Technology, Chinese Academy of Sciences</p>

<div align="center"><a href="http://www.geometrylearning.com/JittorGL"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>


## Overview

![teaser](assets\teaser.png)

The Gaussian Splatting library **JGaussian** not only extends the application of Gaussian Splatting to areas such as **rendering optimization**, **deformation representation**, **attribute decoupled**, **facial relighting** and **style transformation**, but also provides a unified algorithm interface and an efficient training framework, lowering the barrier for reusing Gaussian-based methods. Currently, JGaussian supports 2 base rendering pipeline(3DGS and 2DGS) and 5 methods: 

- [Gaussian-Mesh](https://dl.acm.org/doi/10.1145/3687756)
- [DeferredGS](https://arxiv.org/abs/2404.09412)
- [GSHeadRelight](https://dl.acm.org/doi/10.1145/3721238.3730614)
- SeG-Gaussian
- [StylizedGS](https://arxiv.org/abs/2404.05220)


## Environment Set-up

First, clone this repository to your local machine, and install the dependencies (jittor and other basic python package). 

```bash
conda create -n jgaussian python=3.10
conda activate jgaussian
python3.10 -m pip install jittor
pip install -r requirements.txt
```

Then, Compile the submodules based on C++ and Cuda. 

```bash
sh preprocess/submodule._install.sh 
```

There is an example of final data directory:

```
|---data
|   |---<object>
|   |   |---sparse/(.json file)
|   |   |---images
|   |   |---masks(no necessary need)
|   |   |---segments(no necessary need)
|   |   |---mesh(no necessary need)
```

**masks**: used in spatial stylization, GSmesh with bg (colmap dataset)

**segments**: used in SeG-Gaussian, it can be produced by SAM. Please refer to "preprocess/segment_torch.py".

**mesh**:  store reconstruted mesh and defomed mesh, used in GSmesh.



**!!!**We provide example multi-view dataset, style image(used in [StylizedGS](https://arxiv.org/abs/2404.05220)), environment map (used in [GSHeadRelight](https://dl.acm.org/doi/10.1145/3721238.3730614) and [DeferredGS](https://arxiv.org/abs/2404.09412))  here.

## Start

Vanilla 3DGS:

```bash
device_id=1
dataset_path=nerf_synthetic/lego
output_path=output/lego
CUDA_VISIBLE_DEVICES=${device_id} python 3dgs_trainer.py -s ${dataset_path} -m ${output_path} 
CUDA_VISIBLE_DEVICES=${device_id} python render.py -s ${dataset_path} -m ${output_path}
```



Now, try other methods with these scripts in "scripts", like this:

```bash
sh scripts/deferredgs.sh
```

**gsmesh.sh:** training, rendering and deformation in [Gaussian-Mesh](https://dl.acm.org/doi/10.1145/3687756)

**stylized_gaussian.sh**: spatial,color,base control stylization in [StylizedGS](https://arxiv.org/abs/2404.05220)

**deferredgs.sh**: training, rendering and relighting in [DeferredGS](https://arxiv.org/abs/2404.09412)

**gsheadrelighting.sh**: inference in [GSHeadRelight](https://dl.acm.org/doi/10.1145/3721238.3730614)



We provide the detailed usage of "apply_weight" in SeG-Gaussian,it can be used for gaussian split in any 3DGS-based method([Gaussian-Mesh](https://dl.acm.org/doi/10.1145/3687756),[StylizedGS](https://arxiv.org/abs/2404.05220)):

```python
for i in range(mask_cam.segment.max().long().item()+1):
	mask = (mask_cam.segment==i).float().contiguous()
	partial_loss = jt.pow((image_.detach()  - gt_image_.detach()) * mask,2).sum(0).sqrt().sum() / mask.sum()
	if partial_loss > total_loss:
	weights, weights_cnt= gaussians.apply_weights(mask_cam,  mask)
	selected_pts_mask = jt.logical_or(selected_pts_mask,weights > opt.weight_th)
gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent, size_threshold,selected_pts_mask=selected_pts_mask)
```



**3dgs_combine.sh**: We combine all 3DGS-based methods(Gaussian-Mesh, StylizedGS and SeG-Gaussian),like this:

![3DGScombination](assets\3DGScombination.png)

## Acknowledgements

Thanks to these great repositories: [Gaussian-splatting-Jittor](https://github.com/otakuxiang/gaussian-splatting-jittor.git), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting/),[2DGS]( https://github.com/hbb1/2d-gaussian-splatting),[GSGAN](https://github.com/hse1032/GSGANand ) and many other inspiring works in the community.

