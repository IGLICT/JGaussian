device_id=1

dataset_path=data/dataset/lego
output_path=output/chair

input_mesh=data/dataset/lego/mesh/1.obj

#gaussian_mesh training(with apply_weights in seg-gaussian)
CUDA_VISIBLE_DEVICES=${device_id} python 3dgs_trainer.py -s ${dataset_path} -m ${output_path} --input_mesh ${input_mesh}

#gaussian_mesh stylization
iteration=15000
gaussian_name=point_cloud/iteration_${iteration}/point_cloud.ply
pre_gaussian_path=${output_path}/${gaussian_name}
style_image="data/style/starry_night.jpg"
CUDA_VISIBLE_DEVICES=${device_id} python 3dgs_trainer.py -s ${dataset_path} -m ${output_path} --point_cloud ${pre_gaussian_path} \
    --style ${style_image}  --is_stylized --histgram_match --input_mesh ${input_mesh}


deform_mesh=data/dataset/lego/mesh/2.obj
#gaussian_mesh rendering
CUDA_VISIBLE_DEVICES=${device_id} python render.py -s ${dataset_path} -m ${output_path}
#gaussian_mesh deformation rendering
CUDA_VISIBLE_DEVICES=${device_id} python render.py -s ${dataset_path} -m ${output_path} --deform_mesh ${deform_mesh}
