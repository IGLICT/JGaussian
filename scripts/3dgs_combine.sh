device_id=1

dataset_path=/mnt/155_16T/zhangbotao/jgaussian/nerf_synthetic/chair
output_path=/mnt/155_16T/zhangbotao/jgaussian/output/chair

input_mesh=/mnt/155_16T/zhangbotao/jgaussian/data/MESH/chair/1.obj

#gaussian_mesh training(with apply_weights in seg-gaussian)
CUDA_VISIBLE_DEVICES=${device_id} python 3dgs_trainer.py -s ${dataset_path} -m ${output_path} --input_mesh ${input_mesh}

#gaussian_mesh stylization
iteration=15000
gaussian_name=point_cloud/iteration_${iteration}/point_cloud.ply
pre_gaussian_path=${output_path}/${gaussian_name}
style_image="/mnt/155_16T/zhangbotao/jgaussian/data/style/3.jpg"
CUDA_VISIBLE_DEVICES=${device_id} python 3dgs_trainer.py -s ${dataset_path} -m ${output_path} --point_cloud ${pre_gaussian_path} \
    --style ${style_image}  --is_stylized --histgram_match --input_mesh ${input_mesh}


deform_mesh=/mnt/155_16T/zhangbotao/jgaussian/data/MESH/chair/2.obj
#gaussian_mesh rendering
CUDA_VISIBLE_DEVICES=${device_id} python render.py -s ${dataset_path} -m ${output_path}
#gaussian_mesh deformation rendering
CUDA_VISIBLE_DEVICES=${device_id} python render.py -s ${dataset_path} -m ${output_path} --deform_mesh ${deform_mesh}
