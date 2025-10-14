device_id=1

dataset_path=data/dataset/refnerf/coffee
output_path=output/coffee

CUDA_VISIBLE_DEVICES=${device_id} python 2dgs_trainer.py -s ${dataset_path} -m ${output_path}

CUDA_VISIBLE_DEVICES=${device_id} python render_2dgs.py -s ${dataset_path} -m ${output_path}

#relighting
novel_brdf_envmap=data/env/city.exr
CUDA_VISIBLE_DEVICES=${device_id} python render_2dgs.py -s ${dataset_path} -m ${output_path} --novel_brdf_envmap ${novel_brdf_envmap}
