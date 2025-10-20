device_id=0

dataset_path=data/dataset/lego
output_path=output/lego-mip

CUDA_VISIBLE_DEVICES=${device_id} python mip_trainer.py -s ${dataset_path} -m ${output_path}
CUDA_VISIBLE_DEVICES=${device_id} python mip_render.py -s ${dataset_path} -m ${output_path}
