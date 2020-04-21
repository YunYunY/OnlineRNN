To install packages, run $pip install -e .
To install extra required packages, run $pip install -e .[dev]
To run nosetest $./local_test.sh onlinernn/model_test.py
To train model $python onlinernn/main.py --istrain
To test model $python onlinernn/main.py  
To continue run $python onlinernn/main.py --continue_train --epoch_count=30  # 30 is the restart epoch 