@echo off
FOR %%x IN (17 18 19) DO python train.py --input_length %%x --model_type LSTM