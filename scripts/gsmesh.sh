device_id=1

dataset_path=/mnt/155_16T/zhangbotao/jgaussian/nerf_synthetic/lego
output_path=/mnt/155_16T/zhangbotao/jgaussian/output/lego

input_mesh=/mnt/155_16T/zhangbotao/jgaussian/data/MESH/lego/1/1.obj

CUDA_VISIBLE_DEVICES=${device_id} python 3dgs_trainer.py -s ${dataset_path} -m ${output_path} --input_mesh ${input_mesh}


deform_mesh=/mnt/155_16T/zhangbotao/jgaussian/data/MESH/lego/1/1.obj

CUDA_VISIBLE_DEVICES=${device_id} python render.py -s ${dataset_path} -m ${output_path}
#render deformed gaussian
CUDA_VISIBLE_DEVICES=${device_id} python render.py -s ${dataset_path} -m ${output_path} --deform_mesh ${input_mesh}