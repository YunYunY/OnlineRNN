#!/bin/bash
export CUDA_VISIBLE_DEVICES=8

# outfolder="stdoutNoise5"
# task="Noise5"
# PixelTune
# outfolder="stdoutPixel10"
# task="Pixel10"

# PermuteTune
# outfolder="stdoutPermute10"
# task="Permute10"

# IndRNN
# outfolder="stdoutInd5"
# task="Ind5"

# PixelTrain
outfolder="stdoutP1"
task="M"

mkdir -p $outfolder
# 2015, 2022

for i in {113..113}

do
    rm -f $outfolder/id${i}.txt

    # MNIST tune
    # screen -L -Logfile $outfolder/id${i}.txt -dmS $task${i} \
    # python -u onlinernn/main_indRNN_tune.py --istrain --taskid=$i 

    # MNIST train
    screen -L -Logfile $outfolder/id${i}.txt -dmS $task${i} \
    python -u onlinernn/main_mnist.py --istrain --taskid=$i 

    # HAR-2
    # rm -f stdoutHar/id${i}.txt
    # screen -L -Logfile stdoutHar/id${i}.txt -dmS har${i} \
    # python -u onlinernn/main_har.py --istrain --taskid=$i 

    # nohup python -u onlinernn/main_har_tune.py --istrain --taskid=$i >stdoutHar/id${i}.txt & 
done