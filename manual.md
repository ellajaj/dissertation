# Manual

## Set up
The code in this repository has been designed to run on a GPU. Before running, create a virtual environment and install requirements.txt. 
This code runs with python version 3.7.16

## How to run the code 
### Baseline code 
To execute the baseline code, first set the current directory to the `baseline` folder.  
Then, to evaluate on the CIFAR-10 dataset, run the command:
```bash
python examples_code_cifar.py
```
alternatively, for the Fashion MNIST dataset run:
```bash
python examples_code_f_minist.py
```
It can also be done using the start.sh file. The command that activates the virtual environemnt will need to be changed to for your set up



### Generator 
To pretrain the generator for the GAN-based SSFL, first set the current directory to the `generator` folder
To pretrain a generator for CIFAR-10, run
```bash
python training_script.py
```
alternatively, for the Fashion MNIST dataset run:
```bash
python fmnist_script.py
```

### SSFL
Set the current directory to the `ssl` folder. The generator must be pretrained first. 
To train the GAN-based SSFL for CIFAR-10 run the command:
```bash
python cifar_train_gan.py
```
To train the GAN-based SSFL for Fashion MNIST run the command: 
```bash
python fmnist_train_gan.py
```
It can also be done using the start.sh file, but the contents will need to be altered. The command that activates the virtual environemnt will need to be changed to for your set up
