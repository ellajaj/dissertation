from utils_libs import *
from utils_dataset import *
from utils_methods_FedDC import *
from utils_models import Discriminator, Generator, Classifier

#import torch
#import torch.nn as nn
import numpy as np
import copy


def main():
    # Hyperparameters 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on", device)
    n_clients = 100
    com_amount = 600      # Total communication rounds
    local_epochs = 6      # Local epochs per round
    batch_size = 64
    lr = 0.0002           # Standard GAN learning rate
    alpha_coef = 0.0001      # FedDC penalty coefficient
    #z_dim = 100           # Noise vector size
    num_classes = 10
    act_prob = 0.15 #just whiile developing 

    
    print("Setting up Federated Dataset...")
    # This uses your Dirichlet rule for Non-IID data
    data_obj = DatasetObject(dataset='CIFAR10',  n_client=n_clients,  seed=42,  rule='Drichlet',  rule_arg=0.6 )
    #data_obj = DatasetObject(dataset='CIFAR10', n_client=n_clients, seed=23, rule='iid', unbalanced_sgm=0)


    print("*Initializing Global Models...")
    global_G = Generator(n_label=num_classes).to(device)
    global_C = Classifier(num_classes=num_classes).to(device)
    
    local_discriminators = [
        Discriminator().to(device) for _ in range(n_clients)
    ]


    def get_combined_model_func():
        # This is a helper for FedDC to know the structure of G and C combined
        return global_G, global_C

    print(f"Starting Triple GAN FedDC with {n_clients} clients...")
    
    cur_cld_C, tst_sel_clt_perf = train_FedDC(
        data_obj=data_obj,
        model_func_G = Generator, 
        model_func_C = Classifier,
        model_func_D = Discriminator,
        init_model_G = global_G,  
        init_model_C = global_C,
        act_prob=act_prob,            
        n_minibatch=10,           # Approximate minibatches per epoch
        learning_rate=lr,
        batch_size=batch_size,
        epoch=local_epochs,
        com_amount=com_amount,
        print_per=1,
        weight_decay=1e-4,
        model_func=get_combined_model_func,
        init_model=(global_G, global_C),
        alpha_coef=alpha_coef,
        sch_step=10,
        sch_gamma=0.5,
        save_period=10,
        data_path='Folder/',
        rand_seed=42
    )

    print("Training Complete. Evaluating Final Global Classifier...")
    evaluate_global_model(global_C, data_obj, device)

'''def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        # DCGAN papers recommend N(0.0, 0.02)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)'''

if __name__ == "__main__":
    main()

 
