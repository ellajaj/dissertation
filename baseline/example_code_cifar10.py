import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
print("Torch sees", torch.cuda.device_count(), "GPUs")

from utils_general import *
from utils_methods import *
from utils_methods_FedDC import train_FedDC

def main():
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("num GPUs: ", torch.cuda.device_count())

    # Dataset initialization
    data_path = 'Folder/' # The folder to save Data & Model
    
    n_client = 100
    # Generate IID or Dirichlet distribution
    # IID
    #data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0, data_path=data_path)
    # unbalanced
    #data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0.3, data_path=data_path)

    # Dirichlet (0.6)
    data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.6, data_path=data_path)
    data_obj.limit_dataset(max_samples=6000, min_per_class=60, verbose=True)
    # Dirichlet (0.3)
    #data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.3, data_path=data_path)
    #data_obj.limit_dataset(max_samples=6000, min_per_class=600, verbose=False)

    #model_name = 'Resnet18'
    print("running on resent18")
    model_name = 'cifar10_LeNet' # Model type
    #model_name = 'Resnet50'

    ###
    # Common hyperparameters

    com_amount = 600
    save_period = 100
    weight_decay = 1e-3
    batch_size = 50
    #act_prob = 1
    act_prob = 0.15
    suffix = model_name
    lr_decay_per_round = 0.998

    # Model function
    model_func = lambda : client_model(model_name)
    init_model = model_func()

    # Initalise the model for all methods with a random seed or load it from a saved initial model
    torch.manual_seed(37)
    init_model = model_func()
    if not os.path.exists('%sModel/%s/%s_init_mdl.pt' %(data_path, data_obj.name, model_name)):
        if not os.path.exists('%sModel/%s/' %(data_path, data_obj.name)):
            print("Create a new directory")
            os.mkdir('%sModel/%s/' %(data_path, data_obj.name))
        torch.save(init_model.state_dict(), '%sModel/%s/%s_init_mdl.pt' %(data_path, data_obj.name, model_name))
    else:
        # Load model
        init_model.load_state_dict(torch.load('%sModel/%s/%s_init_mdl.pt' %(data_path, data_obj.name, model_name)))    

    ####

    print('FedDC')

    epoch = 5
    alpha_coef = 1e-2
    learning_rate = 0.1
    print_per = epoch // 2

    n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
    n_iter_per_epoch  = np.ceil(n_data_per_client/batch_size)
    n_minibatch = (epoch*n_iter_per_epoch).astype(np.int64)

    [avg_ins_mdls, avg_cld_mdls, avg_all_mdls, trn_sel_clt_perf, tst_sel_clt_perf, trn_cur_cld_perf, tst_cur_cld_perf, trn_all_clt_perf, tst_all_clt_perf] = train_FedDC(data_obj=data_obj, act_prob=act_prob, n_minibatch=n_minibatch, 
                                        learning_rate=learning_rate, batch_size=batch_size, epoch=epoch, 
                                        com_amount=com_amount, print_per=print_per, weight_decay=weight_decay, 
                                        model_func=model_func, init_model=init_model, alpha_coef=alpha_coef,
                                        sch_step=1, sch_gamma=1,save_period=save_period, suffix=suffix, trial=True,
                                        data_path=data_path, lr_decay_per_round=lr_decay_per_round)
    #exit(0)
    ###
    # baselines

    
if __name__ == "__main__":
    # On Windows, this is REQUIRED when using multiprocessing or DataLoader(num_workers > 0)
    import multiprocessing
    multiprocessing.freeze_support()  # optional unless you’re building an .exe
    main()
