device_id=2

dataset_path=/mnt/155_16T/zhangbotao/jgaussian/nerf_synthetic/lego
output_path=/mnt/155_16T/zhangbotao/jgaussian/output/legomip

CUDA_VISIBLE_DEVICES=${device_id} python mip_trainer.py -s ${dataset_path} -m ${output_path}
CUDA_VISIBLE_DEVICES=${device_id} python mip_render.py -s ${dataset_path} -m ${output_path}
