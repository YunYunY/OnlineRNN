#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
for i in {1003..1004}

do
    # rm -f stdoutHar/id${i}.txt
    # screen -L -Logfile stdoutHar/id${i}.txt -dmS har${i} \
    # python -u onlinernn/main_har_tune.py --istrain --taskid=$i 
    nohup python -u onlinernn/main_har_tune.py --istrain --taskid=$i >stdoutHar/id${i}.txt & 
done