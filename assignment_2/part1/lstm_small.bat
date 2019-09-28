@echo off
FOR %%x IN (11 12 13 14 16) DO python train.py --input_length %%x --model_type LSTM
