#!/bin/bash

export TARGET_DIR=/lustremi/users/$USER/ATOSA-47-reproducer
JOB_NAME="RCCL_reproducer"
SCRIPT_PATH="$TARGET_DIR/scripts/viper-gpu/google_vit/imagenet/vitc/google_vitc_b_16_B2.sh"
LOG_FILE="$HOME/submit.log"


RUNNING=$(squeue -u $USER -n $JOB_NAME -h -t RUNNING,PENDING | wc -l)
if [ $RUNNING -gt 0 ]; then
    echo "$(date): Job '$JOB_NAME' already running" >> $LOG_FILE
    exit 0
else
    # Submit the job
    sbatch $SCRIPT_PATH
    echo "$(date): Job '$JOB_NAME' submitted" >> $LOG_FILE
fi
