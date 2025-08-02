device_id=1

dataset_path=/mnt/155_16T/zhangbotao/jgaussian/data/nerf_llff_data/flower 
output_path=/mnt/155_16T/zhangbotao/jgaussian/output/style-flower-jittor2

iteration=30000
gaussian_name=point_cloud/iteration_${iteration}/point_cloud.ply
pre_gaussian_path=${output_path}/${gaussian_name}

style_image="/mnt/155_16T/zhangbotao/jgaussian/data/style/starry_night.jpg"

# first stage
CUDA_VISIBLE_DEVICES=${device_id} python 3dgs_trainer.py -s ${dataset_path} -m ${output_path}


# second stage: stylized
# base method
CUDA_VISIBLE_DEVICES=${device_id} python 3dgs_trainer.py -s ${dataset_path} -m ${output_path} --point_cloud ${pre_gaussian_path} \
    --style ${style_image}  --is_stylized ----histgram_match

# color control
CUDA_VISIBLE_DEVICES=${device_id} python 3dgs_trainer.py -s ${dataset_path} -m ${output_path} --point_cloud ${pre_gaussian_path} \
    --style ${style_image}  --is_stylized   --preserve_color

# spatial control
mask_dir="/mnt/155_16T/zhangbotao/jgaussian/data/nerf_llff_data/flower/masks"
second_style_image="/mnt/155_16T/zhangbotao/jgaussian/data/style/1.jpg"
CUDA_VISIBLE_DEVICES=${device_id} python 3dgs_trainer.py -s ${dataset_path} -m ${output_path} --point_cloud ${pre_gaussian_path} \
    --style ${style_image}  --second_style ${second_style_image} --mask_dir ${mask_dir} --is_stylized   

CUDA_VISIBLE_DEVICES=${device_id} python render.py -s ${dataset_path} -m ${output_path}
