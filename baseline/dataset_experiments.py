from utils_general import *
from utils_methods import *
from utils_methods_FedDC import train_FedDC

data_path = 'Folder/' # The folder to save Data & Model

n_client = 100

data_obj = DatasetObject(dataset='mnist', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.5, data_path=data_path)
data_obj.limit_dataset(max_samples=1000, min_per_class=1, verbose=True)

model_name = 'mnist_2NN' # Model type

# Common hyperparameters
com_amount = 100
save_period = 100
weight_decay = 1e-3
batch_size = 50
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
        os.makedirs('%sModel/%s/' %(data_path, data_obj.name), exist_ok=True)
    torch.save(init_model.state_dict(), '%sModel/%s/%s_init_mdl.pt' %(data_path, data_obj.name, model_name))
else:
    # Load model
    init_model.load_state_dict(torch.load('%sModel/%s/%s_init_mdl.pt' %(data_path, data_obj.name, model_name)))



# # ####
print('FedDC')

epoch = 5
alpha_coef = 0.1
learning_rate = 0.1
print_per = epoch // 2

# Ensure n_data_per_client reflects the *limited* dataset size
n_data_per_client = data_obj.trn_x.shape[0] / n_client
n_iter_per_epoch  = np.ceil(n_data_per_client/batch_size)
n_minibatch = (epoch*n_iter_per_epoch).astype(np.int64)

print("Shape of trn_x (after limiting and client redistribution):", data_obj.trn_x.shape)
print("Shape of trn_y (after limiting and client redistribution):", data_obj.trn_y.shape)

print(1)
[avg_ins_mdls, avg_cld_mdls, avg_all_mdls, trn_sel_clt_perf, tst_sel_clt_perf, trn_cur_cld_perf, tst_cur_cld_perf, trn_all_clt_perf, tst_all_clt_perf] = train_FedDC(data_obj=data_obj, act_prob=act_prob, n_minibatch=n_minibatch,
                                    learning_rate=learning_rate, batch_size=batch_size, epoch=epoch,
                                    com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                    model_func=model_func, init_model=init_model, alpha_coef=alpha_coef,
                                    sch_step=1, sch_gamma=1,save_period=save_period, suffix=suffix, trial=False,
                                    data_path=data_path, lr_decay_per_round=lr_decay_per_round)

print("Shape of trn_x (after training):", data_obj.trn_x.shape)
print("Shape of trn_y (after training):", data_obj.trn_y.shape)
print(2)