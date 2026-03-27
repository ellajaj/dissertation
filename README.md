# GAN-based Semi-Supervised Federated Learning for Image Classification Applications
This repository contains the code for the GAN-based SSFL on CIFAR-10 and Fashion MNIST datasets. 

## Abstract
The application of traditional deep learning in healthcare presents problems with maintaining the privacy of patient data and obtaining an appropriate amount of data. For these reasons, GAN-based semi-supervised federated learning for image classification is proposed. The project designs an algorithm combining ideas from FedDC and TripleGAN to overcome the lack of labelled data, ensure privacy of data, and handle issues with heterogeneous data. We can achieve an accuracy of 66\% with moderately non-IID data on the CIFAR-10 dataset, and 87\% on the Fashion MNIST dataset, which are competitive results with other similar proposals. The image generation capabilities of the implementation are, however, limited, leaving room for further research.

## File Structure

* **\baseline**: Contains the code for the supervised baseline algorithm. Contains a subfolder `Results`, where the runs, models and processed data will be stored when the code is run.
* **\generator**: Contians the pretrainined generators, training script for these generators and the evaluation code.  
* **\ssl**: Contains the main code for the GAN-based SSFL. includes a training script for both datasets and a number of util files. the evaluation code can be found in the home directory of the project.Contains a subfolder `Results`, where the runs, models and processed data will be stored when the code is run.  
* **\Results**: contains a folder of raw runs which the plots are generated from 


## How to run the code 
The code in this repository has been designed to run on a GPU. Before running, create a virtual environment and install requirements.txt. 

### Baseline code 
To execute the baseline code, first set the current directory to the `baseline` folder.  
Then, to evaluate on the CIFAR-10 dataset, run the command:
```bash
python example_code_cifar10.py
```
alternatively, for the Fashion MNIST dataset run:
```bash
python example_code_f_minist.py
```
It can also be done using the start.sh file



### Generator 
To pretrain the generator for the GAN-based SSFL, first set the current directory to the `generator` folder
To pretrain a generator for CIFAR-10, run
```bash
python training_script.py
```
alternatively, for the Fashion MNIST dataset run:
```bash
python fmnist_training_script.py
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
It can also be done using the start.sh file

