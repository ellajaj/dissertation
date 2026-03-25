#! /bin/bash

source ~/miniconda3/bin/activate
conda activate flenv 

nohup python cifar_train_gan.py 1>out.txt 2>err.txt & echo $! > pid.txt