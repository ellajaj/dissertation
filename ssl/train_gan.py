from utils_libs import *
from utils_dataset import *
from utils_methods_FedDC import *
from utils_models import Discriminator, Generator, Classifier

#import torch
#import torch.nn as nn
import numpy as np
import copy


def main():
    torch.cuda.empty_cache()#remove
    # Hyperparameters 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on", device)
    n_clients = 100
    com_amount = 1000      # Total communication rounds
    local_epochs = 6      # Local epochs per round
    batch_size = 64
    lr = 0.0002           # Standard GAN learning rate
    alpha_coef = 0.0001      # FedDC penalty coefficient
    z_dim = 100           # Noise vector size
    num_classes = 10
    act_prob = 0.15
    
    print("Setting up Federated Dataset...")
    # This uses your Dirichlet rule for Non-IID data
    data_obj = DatasetObject(dataset='CIFAR10',  n_client=n_clients,  seed=42,  rule='Drichlet',  rule_arg=0.6 )
    #data_obj = DatasetObject(dataset='fashion_mnist',  n_client=n_clients,  seed=42,  rule='Drichlet',  rule_arg=0.6 )
    #data_obj = DatasetObject(dataset='CIFAR10', n_client=n_clients, seed=23, rule='iid', unbalanced_sgm=0)


    print("*Initializing Global Models...")
    global_G = Generator(data_obj.dataset, z_dim).to(device)
    global_C = Classifier(data_obj.dataset, num_classes=num_classes).to(device)

    def get_combined_model_func():
        # This is a helper for FedDC to know the structure of G and C combined
        return global_G, global_C


    if data_obj.dataset=='CIFAR10':
        print("Trying pretrained generator")

        # 2. Load the Pre-trained Weights
        print("Loading pre-trained generator...")
        pretrained_dict = torch.load("../generator/generator.pth", map_location=device)

        # Optional: If the file contains more than just the model (like optimizer states),
        # you might need to access the sub-dictionary, e.g., pretrained_dict['model_state_dict']
        if 'model_state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['model_state_dict']

        init_model_G = global_G

        # 3. Load carefully (strict=False ignores minor missing keys like running_mean in some cases)
        try:
            init_model_G.load_state_dict(pretrained_dict, strict=True)
            print("SUCCESS: Pre-trained generator loaded perfectly.")
        except RuntimeError as e:
            print(f"WARNING: Strict loading failed. Attempting partial load. Error: {e}")
            # Filter out unnecessary keys if needed
            model_dict = init_model_G.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            init_model_G.load_state_dict(model_dict)

    elif data_obj.dataset=='fashion_mnist':
        init_model_G = global_G

    '''local_discriminators = [
        Discriminator(data_obj.dataset).to(device) for _ in range(n_clients)
    ]'''

    print(f"Starting Triple GAN FedDC with {n_clients} clients...")
    
    cur_cld_C, tst_sel_clt_perf = train_FedDC(
        data_obj=data_obj,
        model_func_G = Generator, 
        model_func_C = Classifier,
        model_func_D = Discriminator,
        #init_model_G = global_G,  
        init_model_G = init_model_G,
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

 
