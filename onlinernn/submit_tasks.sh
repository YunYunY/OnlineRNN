#!/bin/bash
CUDA_VISIBLE_DEVICES=1
for i in {100..100}
do
    screen -dmS har${i} \
    python onlinernn/main_har_tune.py --istrain --taskid=$i \
    >stdoutHar/id${i}.txt
done