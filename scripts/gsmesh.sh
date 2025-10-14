device_id=1

dataset_path=data/dataset/lego
output_path=output/lego-gsmesh

input_mesh=data/dataset/lego/mesh/1.obj

CUDA_VISIBLE_DEVICES=${device_id} python 3dgs_trainer.py -s ${dataset_path} -m ${output_path} --input_mesh ${input_mesh}


deform_mesh=data/dataset/lego/mesh/2.obj

CUDA_VISIBLE_DEVICES=${device_id} python render.py -s ${dataset_path} -m ${output_path}
#render deformed gaussian
CUDA_VISIBLE_DEVICES=${device_id} python render.py -s ${dataset_path} -m ${output_path} --deform_mesh ${input_mesh}
