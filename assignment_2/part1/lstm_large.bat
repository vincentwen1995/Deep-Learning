@echo off
FOR %%x IN (21 22 23 24) DO python train.py --input_length %%x --model_type LSTM --learning_rate 0.0001
