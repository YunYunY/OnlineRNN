#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

# outfolder="stdoutHartune10"
# task="HAR10"

outfolder="Har"
task="HAR"

# PermuteTune
# outfolder="stdoutPermute5"
# task="Permute5"

# PixelTrain
# outfolder="stdoutP1"
# task="M"

mkdir -p $outfolder
# 2015, 2022

for i in {920..920}

do
    rm -f $outfolder/id${i}.txt


    # MNIST train
    screen -L -Logfile $outfolder/id${i}.txt -dmS $task${i} \
    python -u onlinernn/main_har.py --istrain --taskid=$i 


    # nohup python -u onlinernn/main_har_tune.py --istrain --taskid=$i >stdoutHar/id${i}.txt & 
done