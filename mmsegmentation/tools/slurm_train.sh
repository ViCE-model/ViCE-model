#!/usr/bin/env bash

set -x

PARTITION=$1
MACHINE=$2
JOB_NAME=$3
CONFIG=$4
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

echo ${PY_ARGS}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
       -w ${MACHINE} \
       --job-name=${JOB_NAME} \
       --gres=gpu:${GPUS_PER_NODE} \
       --ntasks=${GPUS} \
       --ntasks-per-node=${GPUS_PER_NODE} \
       --cpus-per-task=${CPUS_PER_TASK} \
       --kill-on-bad-exit=1 \
       ${SRUN_ARGS} \
       python -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}