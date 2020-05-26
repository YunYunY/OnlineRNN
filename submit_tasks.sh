#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
outfolder="stdoutPixel10"
task="Pixel10"
mkdir -p $outfolder

for i in {404..404}

do
    # MNIST tune
    rm -f $outfolder/id${i}.txt
    screen -L -Logfile $outfolder/id${i}.txt -dmS $task${i} \
    python -u onlinernn/main_mnist_tune.py --istrain --taskid=$i 

    # MNIST
    # rm -f stdoutP1/id${i}.txt
    # screen -L -Logfile stdoutP1/id${i}.txt -dmS M${i} \
    # python -u onlinernn/main_mnist.py --istrain --taskid=$i 

    # HAR-2
    # rm -f stdoutHar/id${i}.txt
    # screen -L -Logfile stdoutHar/id${i}.txt -dmS har${i} \
    # python -u onlinernn/main_har.py --istrain --taskid=$i 

    # nohup python -u onlinernn/main_har_tune.py --istrain --taskid=$i >stdoutHar/id${i}.txt & 
done