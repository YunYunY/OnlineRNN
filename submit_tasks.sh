#!/bin/bash
export CUDA_VISIBLE_DEVICES=8

# outfolder="stdoutHartune10"
# task="HAR10"

# outfolder="Har"
# task="HAR"

outfolder="stdoutAdding1"
task="Adding1"
# PixelTrain
# outfolder="stdoutP1"
# task="M"

mkdir -p $outfolder
# 2015, 2022

for i in {804..805}

do
    rm -f $outfolder/id${i}.txt


    # MNIST train
    screen -L -Logfile $outfolder/id${i}.txt -dmS $task${i} \
    python -u onlinernn/main_add_tune.py --istrain --taskid=$i 


    # nohup python -u onlinernn/main_har_tune.py --istrain --taskid=$i >stdoutHar/id${i}.txt & 
done