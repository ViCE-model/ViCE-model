#!/bin/bash -x
#PJM -L rscgrp=cxgfs-middle
#PJM -L node=8
#PJM -L elapse=72:00:00
#PJM -j
#PJM -S

module load gcc/8.4.0
module load python/3.8.3
module load cuda/11.2.1
module load cudnn/8.1.1
module load nccl/2.8.4
module load openmpi_cuda/4.0.5

# Replace YOUR-USERNAME
source /home/YOUR-USERNAME/.pyenv/versions/vissl/bin/activate

# distributed setting

# -np == #nodes
mpirun \
    -np 8 \
    -npernode 1 \
    -bind-to none \
    -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME="ib0" \
    -mca pml ob1 \
    -mca btl ^openib \
    -mca btl_tcp_if_include ib0 \
    -mca plm_rsh_agent /bin/pjrsh \
    -machinefile ${PJM_O_NODEINF} \
    # Replace YOUR-USERNAME
    /home/YOUR-USERNAME/projects/vissl/distributed_train.sh