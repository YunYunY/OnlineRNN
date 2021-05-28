#!/bin/bash
export CUDA_VISIBLE_DEVICES=3


outfolder="Har"
task="HAR"

# outfolder="stdoutAdding"
# task="Adding"

# PixelTrain
# outfolder="stdoutPixel"
# task="Pixel"

# CIFAR
# outfolder="stdoutCIFAR"
# task="CIFAR"


mkdir -p $outfolder

for i in {103011..103011}

do
    rm -f $outfolder/id${i}.txt

    screen -L -Logfile $outfolder/id${i}.txt -dmS $task${i} \
    python -u onlinernn/main_har_tune.py --istrain --taskid=$i 

   
    # nohup python -u onlinernn/main_har_tune.py --istrain --taskid=$i >stdoutHar/id${i}.txt & 
done
# done