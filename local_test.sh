#!/bin/bash
################################
# Unit test on a local computer (not on cluster)
# How to use this script?
# This script is used for unit tests
# in your local computer (laptop) terminal, type: ./local_test.sh [filename]
# the [filename] is the file that you want to run unit tests, for example, model_test.py
################################

# CUDA_VISIBLE_DEVICES=0,1 nosetests -v --nologcapture -s $1
# run with all available GPUs
nosetests -v --nologcapture -s $1