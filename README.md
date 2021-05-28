# This project is to work on Pytorch version >1.4 with GPU support on a linux server. 

- To install packages, run $pip install -e .

- To install extra required packages, run $pip install -e .[dev]

- To train model $CUDA_VISIBLE_DEVICES=1 python onlinernn/main_har_tune.py --istrain --taskid=103011

or 

- python submit_tasks.sh --istrain --taskid=103011

- To run nosetest $./local_test.sh onlinernn/tests/model_test.py 

- To test model $python onlinernn/main.py --taskid=1 --load_iter=299

- To continue run $python onlinernn/main.py --continue_train --epoch_count=30  # 30 is the restart epoch 

- To check time running $CUDA_VISIBLE_DEVICES=1 python -m cProfile -s cumtime onlinernn/main.py --istrain --taskid=22>id22time.txt