'''code adapted from https://github.com/gaoliang13/FedDC -- MIT license'''

from utils_libs import *
from utils_dataset import *
from utils_models import *

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import time
max_norm = 10
# --- Evaluate a NN model
def get_acc_loss(data_x, data_y, model, dataset_name, w_decay = None):
    acc_overall = 0 
    loss_overall = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    batch_size = min(6000, data_x.shape[0])
    batch_size = min(2000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval(); model = model.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst/batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)

            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_overall += loss.item()

            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct


    loss_overall /= n_tst
    if w_decay != None:
        # Add L2 loss
        params = get_mdl_params([model], n_par=None)
        loss_overall += w_decay/2 * np.sum(params * params)

    model.train()
    return loss_overall, acc_overall / n_tst

def evaluate_global_model(model, data_obj, device, batch_size=64):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    # ---------- Forward pass on test data ----------
    with torch.no_grad():
        N = len(data_obj.tst_x)
        i = 0
        while i < N:
            j = min(N, i + batch_size)
            batch_x = np.stack(data_obj.tst_x[i:j], axis=0)
            inputs = torch.tensor(batch_x, dtype=torch.float32, device=device)
            outputs = model(inputs)

            if outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs.view(-1)).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
            else:
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

            y_true.extend(data_obj.tst_y[i:j])
            y_pred.extend(preds)
            y_prob.append(probs)
            i = j

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.concatenate(y_prob, axis=0)

    # ---------- Basic metrics ----------
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n===== FINAL GLOBAL MODEL EVALUATION =====")
    print(f"Accuracy:       {acc:.4f}")
    print(f"F1 (macro):     {f1_macro:.4f}")
    print(f"F1 (weighted):  {f1_weighted:.4f}")
    print(f"Precision:      {prec:.4f}")
    print(f"Recall:         {rec:.4f}")
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_true, y_pred, digits=4))

    # ---------- AUROC / PR-AUC ----------
    try:
        if y_prob.ndim == 1 or y_prob.shape[1] == 1:
            auroc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
        else:
            auroc = roc_auc_score(y_true, y_prob, multi_class='ovr')
            pr_auc = np.mean([
                average_precision_score((y_true == c).astype(int), y_prob[:, c])
                for c in np.unique(y_true)
            ])
        print(f"AUROC:          {auroc:.4f}")
        print(f"PR-AUC:         {pr_auc:.4f}")
    except Exception as e:
        print("AUROC/PR-AUC computation failed:", e)

    # ---------- Per-client metrics (if available) ----------
    if hasattr(data_obj, 'tst_x_per_client') and hasattr(data_obj, 'tst_y_per_client'):
        client_metrics = []
        for cid in range(len(data_obj.tst_x_per_client)):
            Xc, Yc = data_obj.tst_x_per_client[cid], data_obj.tst_y_per_client[cid]
            if len(Yc) == 0:
                continue
            yt, yp, _ = evaluate_client(model, Xc, Yc, device, batch_size)
            f1c = f1_score(yt, yp, average='macro')
            accc = accuracy_score(yt, yp)
            recc = recall_score(yt, yp, average='macro')
            client_metrics.append((cid, len(Yc), accc, f1c, recc))

        df = pd.DataFrame(client_metrics, columns=['client_id', 'n_samples', 'acc', 'f1_macro', 'recall_macro'])
        df.to_csv("per_client_eval.csv", index=False)
        print("\nPer-client metrics saved to per_client_eval.csv")
        print(df.describe())

    print("==========================================\n")
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision": prec,
        "recall": rec
    }

def evaluate_client(model, X, Y, device, batch_size=64):
    model.eval()
    yt, yp, yprob = [], [], []
    with torch.no_grad():
        N = len(X)
        i = 0
        while i < N:
            j = min(N, i + batch_size)
            bx = np.stack(X[i:j], axis=0)
            inp = torch.tensor(bx, dtype=torch.float32, device=device)
            out = model(inp)
            if out.shape[1] == 1:
                prob = torch.sigmoid(out.view(-1)).cpu().numpy()
                pred = (prob >= 0.5).astype(int)
            else:
                prob = torch.softmax(out, dim=1).cpu().numpy()
                pred = np.argmax(prob, axis=1)
            yt.extend(Y[i:j])
            yp.extend(pred)
            yprob.append(prob)
            i = j
    return np.array(yt), np.array(yp), np.concatenate(yprob, axis=0)

# --- Helper functions

def avg_models(mdl, clnt_models, weight_list):
    n_node = len(clnt_models)
    dict_list = list(range(n_node));
    for i in range(n_node):
        dict_list[i] = copy.deepcopy(dict(clnt_models[i].named_parameters()))

    param_0 = clnt_models[0].named_parameters()

    for name, param in param_0:
        param_ = weight_list[0] * param.data
        for i in list(range(1 ,n_node)):
            param_ = param_ + weight_list[i] * dict_list[i][name].data
        dict_list[0][name].data.copy_(param_)

    mdl.load_state_dict(dict_list[0])

    # Remove dict_list from memory
    del dict_list

    return mdl


def get_mdl_params(model_list, n_par=None):

    if n_par==None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)

def train_model_FedDC(model, model_func, alpha, local_update_last, global_update_last, global_model_param, hist_i, trn_x, trn_y,
                    learning_rate, batch_size, epoch, print_per,
                    weight_decay, dataset_name, sch_step, sch_gamma):

    n_trn = trn_x.shape[0]
    state_update_diff = torch.tensor(-local_update_last+ global_update_last,  dtype=torch.float32, device=device)
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    #loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    model = model.to(device)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]


    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)

            ## Get f_i estimate
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())


            loss_f_i = loss_f_i / list(batch_y.size())[0]

            local_parameter = None
            for param in model.parameters():
                if not isinstance(local_parameter, torch.Tensor):
                # Initially nothing to concatenate
                    local_parameter = param.reshape(-1)
                else:
                    local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

            loss_cp = alpha/2 * torch.sum((local_parameter - (global_model_param - hist_i))*(local_parameter - (global_model_param - hist_i)))
            loss_cg = torch.sum(local_parameter * state_update_diff)


            loss = loss_f_i + loss_cp + loss_cg
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients to prevent exploding
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e+1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (weight_decay)/2 * np.sum(params * params)

            #print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  #%(e+1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model

def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx+length].reshape(weights.shape)).to(device))
        idx += length

    mdl.load_state_dict(dict_param)
    return mdl
    
