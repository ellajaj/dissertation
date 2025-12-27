from utils_libs import *
from utils_dataset import *
from utils_methods_FedDC import *
from utils_models import *

import torch
import torch.nn as nn
import numpy as np
import copy

# Assuming your classes (Generator, Classifier, Discriminator, DatasetObject) 
# and the train_FedDC functions are in the same script or imported.

def main():
    # --- 1. Hyperparameters ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on", device)
    n_clients = 10
    com_amount = 100      # Total communication rounds
    local_epochs = 5      # Local epochs per round
    batch_size = 64
    lr = 0.0002           # Standard GAN learning rate
    alpha_coef = 0.1      # FedDC penalty coefficient
    z_dim = 100           # Noise vector size
    num_classes = 10
    
    # --- 2. Data Initialization ---
    print("Setting up Federated Dataset...")
    # This uses your Dirichlet rule for Non-IID data
    data_obj = DatasetObject(dataset='CIFAR10',  n_client=n_clients,  seed=42,  rule='Drichlet',  rule_arg=0.3 )

    # --- 3. Model Initialization ---
    print("Initializing Global Models...")
    global_gen = Generator(z_dim=z_dim, n_label=num_classes).to(device)
    global_clf = Classifier(num_classes=num_classes).to(device)
    
    # Persistent Local Discriminators (One per client, not federated)
    # We store them in a list so they keep their weights across rounds
    local_discriminators = [
        Discriminator(n_label=num_classes).to(device) for _ in range(n_clients)
    ]

    # --- 4. Parameter Concatenation Setup ---
    # FedDC needs to treat (G + C) as one long vector
    def get_combined_model_func():
        # This is a helper for FedDC to know the structure of G and C combined
        return global_gen, global_clf

    # --- 5. Start Federated Training ---
    print(f"Starting Triple GAN FedDC with {n_clients} clients...")
    
    # We pass our modified Triple GAN training loop into your FedDC logic
    avg_ins, avg_cld, avg_all, trn_perf, tst_perf, _, _, _, _ = train_FedDC(
        data_obj=data_obj,
        act_prob=0.5,             # 50% of clients participate per round
        n_minibatch=10,           # Approximate minibatches per epoch
        learning_rate=lr,
        batch_size=batch_size,
        epoch=local_epochs,
        com_amount=com_amount,
        print_per=1,
        weight_decay=1e-4,
        model_func=get_combined_model_func, # Pass combined model
        init_model=(global_gen, global_clf),
        alpha_coef=alpha_coef,
        sch_step=10,
        sch_gamma=0.5,
        save_period=10,
        data_path='./results/',
        rand_seed=42
    )

    print("Training Complete. Evaluating Final Global Classifier...")
    # Final evaluation on test set
    evaluate_global_model(global_clf, data_obj, device)

if __name__ == "__main__":
    main()


'''#initalise global models 
G_global = Generator()
C_global = Classifier()

#get fed dc state variables??

#set params (comm rounds, num clients )
n_client = 100 
# Dirichlet (0.3)
data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.3, data_path=data_path)


comm_ammount = 100


for round in comm_ammount:


# for round in comm rounds 

    #select clients 
    #
    #for client in selected clients 

        #send G and C to clients 
        #do local training 

        #update local discriminator
        #update copy of G
        #update copy of C

    #after all clients trained 
    #use fed dc to aggregate G and C updates
    #update G and C '''
 
