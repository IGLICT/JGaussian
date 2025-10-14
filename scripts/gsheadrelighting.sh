#!/bin/bash
device_id=1
network_pkl=data/ckpt/gshead.pkl
resolution=512

trunc=0.7
seeds=0,9
opacity_ones=False
lighting_pattern='envmap'
outdir=out
grid=1x1
lighting_transfer_ids=62,68,82,90
with_bg=False

CUDA_VISIBLE_DEVICES=${device_id} python application/gsheadrelighting/gen_videos_gsparams.py --outdir=${outdir}/ --trunc=${trunc} --seeds=${seeds} --grid=${grid} \
    --network=${network_pkl} --image_mode=image \
    --nrr=${resolution} --opacity_ones=${opacity_ones} --lighting_pattern=${lighting_pattern} --lighting_transfer_ids=${lighting_transfer_ids} \
