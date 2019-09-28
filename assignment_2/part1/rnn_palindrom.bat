@echo off
FOR /L %%x IN (5,1,100) DO python train.py --input_length %%x
