#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

# PixelTune
# outfolder="stdoutPixel10"
# task="Pixel10"

# PermuteTune
# outfolder="stdoutPermute10"
# task="Permute10"

# IndRNN
# outfolder="stdoutInd10"
# task="Ind10"

# PixelTrain
outfolder="stdoutP1"
task="M"

mkdir -p $outfolder

for i in {2400..2400}

do
    rm -f $outfolder/id${i}.txt

    # MNIST tune
    # screen -L -Logfile $outfolder/id${i}.txt -dmS $task${i} \
    # python -u onlinernn/main_permute_tune.py --istrain --taskid=$i 

    # MNIST train
    screen -L -Logfile $outfolder/id${i}.txt -dmS $task${i} \
    python -u onlinernn/main_mnist.py --istrain --taskid=$i 

    # HAR-2
    # rm -f stdoutHar/id${i}.txt
    # screen -L -Logfile stdoutHar/id${i}.txt -dmS har${i} \
    # python -u onlinernn/main_har.py --istrain --taskid=$i 

    # nohup python -u onlinernn/main_har_tune.py --istrain --taskid=$i >stdoutHar/id${i}.txt & 
done