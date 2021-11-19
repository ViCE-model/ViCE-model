#!/bin/bash

set -euo pipefail

# load modules
source /etc/profile.d/modules.sh
module load gcc/8.4.0
module load python/3.8.3
module load cuda/11.2.1
module load cudnn/8.1.1
module load nccl/2.8.4
module load openmpi_cuda/4.0.5

source /home/z44406a/.pyenv/versions/vissl/bin/activate

# https://github.com/pytorch/pytorch/issues/37377
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1

# distributed setting
MY_ADDR=$(hostname -I | awk '{print $1}')
MASTER_ADDR=$(head -n 1 ${PJM_O_NODEINF})
NODE_RANK=$(cat ${PJM_O_NODEINF} | awk '{print NR-1 " " $1}' | grep ${MY_ADDR}$ | awk '{print $1}')
echo "MY_ADDR=${MY_ADDR}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "NODE_RANK=${NODE_RANK}"

python tools/run_distributed_engines.py \
    config=pretrain/vice/vice_8node_resnet_coco_exp27.yaml \
    config.DISTRIBUTED.RUN_ID="${MASTER_ADDR}":29500 \
    node_id="${NODE_RANK}"