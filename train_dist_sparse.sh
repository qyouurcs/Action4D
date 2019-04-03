#!/bin/bash

BATCH_SIZE=24
DATA_DIR="test3_cross_env/train.tar"
DATA_DIR_VAL="test3_cross_env/data_shell_f9_val.tar"
LIST_FN="test3_cross_env/train.lst"
LIST_FN_VAL="test3_cross_env/val.lst"

if [ $# -ge 1 ]; then
    BATCH_SIZE=$1
fi

if [ $CONTAINER_INDEX -eq 0 ]; then
    mpirun -- -npernode 1 python train_dist.py --data_dir_val $DATA_DIR_VAL --data_dir $DATA_DIR --list_fn $LIST_FN --list_fn_val $LIST_FN_VAL --batch_size $BATCH_SIZE

else
    bash -c "/usr/sbin/sshd -p $PHILLY_CONTAINER_SSHD_PORT; sleep infinity"
fi
