@echo off
FOR %%x IN (5 6 7 8 9 10) DO python train.py --input_length %%x --model_type LSTM
FOR %%x IN (15 20 25) DO python train.py --input_length %%x --model_type LSTM --learning_rate 0.0001 
