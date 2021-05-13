#!/bin/bash
export CUDA_VISIBLE_DEVICES=5

# outfolder="stdoutHartune10"
# task="HAR10"

# outfolder="Har"
# task="HAR"

outfolder="stdoutAdding"
task="Adding"
# PixelTrain
# outfolder="stdoutP1"
# task="M"

mkdir -p $outfolder

for i in {111..111}

do
    rm -f $outfolder/id${i}.txt

    screen -L -Logfile $outfolder/id${i}.txt -dmS $task${i} \
    python -u onlinernn/main_add_tune.py --istrain --taskid=$i 

   
    # nohup python -u onlinernn/main_har_tune.py --istrain --taskid=$i >stdoutHar/id${i}.txt & 
done
# done