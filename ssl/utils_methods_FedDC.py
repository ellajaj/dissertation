from utils_libs import *
from utils_dataset import *
from utils_models import *
from tensorboardX import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
### Methods



def train_FedDC(data_obj, model_func_G, model_func_C, model_func_D,
                init_model_G, init_model_C, 
                act_prob, n_minibatch, learning_rate, batch_size, epoch, com_amount, print_per,
                weight_decay,  model_func, init_model, alpha_coef,
                sch_step, sch_gamma, save_period,
                suffix = '', trial=False, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix  = 'FedDC_' + str(alpha_coef)+suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' %(save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' %rand_seed
    suffix += '_lrdecay%f' %lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y

    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' %(data_path, data_obj.name, suffix))):
        os.makedirs('%sModel/%s/%s' %(data_path, data_obj.name, suffix), exist_ok=True)

    n_par_G = len(get_mdl_params([init_model_G])[0])
    n_par_C = len(get_mdl_params([init_model_C])[0])
    n_par = n_par_G + n_par_C

    # --- 3. Persistent State Buffers ---
    parameter_drifts = np.zeros((n_clnt, n_par)).astype('float32')
    state_gadient_diffs = np.zeros((n_clnt + 1, n_par)).astype('float32') 
    clnt_params_list = np.zeros((n_clnt, n_par)).astype('float32')
    
    # Persistent local discriminators (not federated)
    local_Ds = [model_func_D().to(device) for _ in range(n_clnt)]
    
    # Initialize clnt_params_list with initial global params
    init_par_G = get_mdl_params([init_model_G])[0]
    init_par_C = get_mdl_params([init_model_C])[0]
    init_combined = np.concatenate([init_par_G, init_par_C])
    clnt_params_list = np.ones(n_clnt).reshape(-1, 1) * init_combined.reshape(1, -1)

    cur_cld_G = model_func_G().to(device)
    cur_cld_C = model_func_C().to(device)
    cur_cld_G.load_state_dict(init_model_G.state_dict())
    cur_cld_C.load_state_dict(init_model_C.state_dict())
    cld_mdl_param = init_combined.copy()

    n_save_instances = int(com_amount / save_period)
    avg_ins_mdls = list(range(n_save_instances))
    avg_all_mdls = list(range(n_save_instances))
    avg_cld_mdls = list(range(n_save_instances))

    trn_sel_clt_perf = np.zeros((com_amount, 2))
    tst_sel_clt_perf = np.zeros((com_amount, 2))

    trn_all_clt_perf = np.zeros((com_amount, 2))
    tst_all_clt_perf = np.zeros((com_amount, 2))

    trn_cur_cld_perf = np.zeros((com_amount, 2))
    tst_cur_cld_perf = np.zeros((com_amount, 2))

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns/%s/%s' %(data_path, data_obj.name, suffix))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            # Client selection
            inc_seed = 0
            while True:
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                selected_clnts = np.sort(np.where(act_list <= act_prob)[0])
                inc_seed += 1
                if len(selected_clnts) != 0: break

            global_mdl_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)
            delta_g_sum = np.zeros(n_par)

            for clnt in selected_clnts:
                print('---- Round %d, Training client %d' % (i, clnt))
                
                # Data Split (Semi-Supervised)
                tx, ty = clnt_x[clnt], clnt_y[clnt]
                idx = np.random.permutation(len(ty))
                split = int(len(ty) * 0.1) # 10% Labeled
                xl, yl = tx[idx[:split]], ty[idx[:split]]
                xu = tx[idx[split:]]

                # Setup local models
                local_G = model_func_G().to(device)
                local_C = model_func_C().to(device)
                set_combined_params(local_G, local_C, cld_mdl_param, n_par_G)
                
                # Local Trainer
                local_G, local_C, s_loss, a_loss, f_loss = train_model_TripleFedDC(
                    local_G, local_C, local_Ds[clnt],

                    alpha_coef / weight_list[clnt],
                    state_gadient_diffs[clnt], 
                    state_gadient_diffs[-1] / weight_list[clnt],
                    global_mdl_tensor, 
                    torch.tensor(parameter_drifts[clnt], device=device),
                    xl, yl, xu,
                    learning_rate * (lr_decay_per_round ** i),
                    batch_size, epoch, weight_decay,
                )
                            

                # Update FedDC tracking
                curr_par = get_combined_params(local_G, local_C, n_par)
                delta_p = curr_par - cld_mdl_param
                parameter_drifts[clnt] += delta_p
                
                beta = 1 / (n_minibatch * epoch) / (learning_rate * (lr_decay_per_round ** i))
                state_g = state_gadient_diffs[clnt] - (state_gadient_diffs[-1]/weight_list[clnt]) + beta * (-delta_p)
                
                delta_g_sum += (state_g - state_gadient_diffs[clnt]) * weight_list[clnt]
                state_gadient_diffs[clnt] = state_g
                clnt_params_list[clnt] = curr_par

                #print(f" Client {clnt} | Sup: {s_loss:.4f} | Adv: {a_loss:.4f} | Fed: {f_loss:.4f}")
                #print(len(curr_par))

            # --- 5. Aggregation ---
            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)
            state_gadient_diffs[-1] += (1 / n_clnt) * delta_g_sum
            cld_mdl_param = avg_mdl_param_sel + np.mean(parameter_drifts, axis=0)

            # Update global evaluation model (Classifier only for Accuracy)
            set_combined_params(cur_cld_G, cur_cld_C, cld_mdl_param, n_par_G)
            
            loss_t, acc_t = get_acc_loss(data_obj.tst_x, data_obj.tst_y, cur_cld_C, data_obj.dataset)
            tst_sel_clt_perf[i] = [loss_t, acc_t]
            #print(len(cld_mdl_param))
            print("**** Round %d, Test Accuracy: %.4f, loss = %.4f" % (i+1, acc_t, loss_t))

    return cur_cld_C, tst_sel_clt_perf


def train_model_TripleFedDC(G, C, D, alpha, 
                            local_update_last, global_update_last, global_mdl_param, hist_i, 
                            trn_x_labeled, trn_y_labeled, trn_x_unlabeled,
                            learning_rate, batch_size, epoch, weight_decay):
    torch.autograd.set_detect_anomaly(True) #debugging
    sup_loss_tot, adv_loss_tot, fed_loss_tot = 0, 0, 0
    count = 0

    # 1. Setup OptimizersS
    # GANs usually perform better with Adam than SGD
    opt_G = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    opt_C = torch.optim.Adam(C.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    #opt_C = torch.optim.SGD(C.parameters(), lr= learning_rate, momentum=0.9)
    opt_D = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # FedDC constant: state_update_diff = -local_update + global_update
    state_update_diff = torch.tensor(-local_update_last + global_update_last, dtype=torch.float32, device=device)
    
    # DataLoader yielding (x_l, y_l, x_u)
    trn_gen = torch.utils.data.DataLoader(
        TripleGANDataset(trn_x_labeled, trn_y_labeled, trn_x_unlabeled, train=True), 
        batch_size=batch_size, shuffle=True
    )

    G.train(); C.train(); D.train()

    for e in range(epoch):
        for img_l, label_l, img_u in trn_gen:
            img_l, label_l, img_u = img_l.to(device), label_l.to(device), img_u.to(device)
            batch_sz = img_l.size(0)

            # -------------------------
            # 1. Train Discriminator D
            # -------------------------
            opt_D.zero_grad()
            
            # Stream 1: Real Labeled Data
            d_real_loss = torch.mean(torch.nn.functional.softplus(-D(img_l, label_l.long())))
            
            # Stream 2: Synthetic Data from G
            z = torch.randn(batch_sz, G.z_dim).to(device)
            y_gen = torch.randint(0, 10, (batch_sz,)).to(device)
            img_fake = G(z, y_gen)
            d_fake_loss = torch.mean(torch.nn.functional.softplus(D(img_fake.detach(), y_gen)))
            
            # Stream 3: Pseudo-labeled Data from C
            y_pseudo = torch.argmax(C(img_u), dim=1)
            d_pseudo_loss = torch.mean(torch.nn.functional.softplus(D(img_u, y_pseudo.detach())))
            
            loss_D = d_real_loss + d_fake_loss + d_pseudo_loss
            loss_D.backward()
            opt_D.step()

            # ------------------------------------
            # 2. Train Generator G and Classifier C
            # ------------------------------------
            opt_G.zero_grad()
            opt_C.zero_grad()

            # Adversarial Losses (Fool the Discriminator)
            loss_G_adv = torch.mean(torch.nn.functional.softplus(-D(G(z, y_gen), y_gen)))
            # The index must be detached so it is treated as a fixed label for the indexing operation
            pseudo_label = torch.argmax(C(img_u), dim=1).detach()
            loss_C_adv = torch.mean(torch.nn.functional.softplus(-D(img_u, pseudo_label)))
            
            # Supervised Loss for C
            loss_C_sup = torch.nn.functional.cross_entropy(C(img_l), label_l.long().squeeze())

            # --- FedDC Penalty Logic ---
            # Separate params
            params_G = torch.cat([p.view(-1) for p in G.parameters()])
            params_C = torch.cat([p.view(-1) for p in C.parameters()])
            
            # Split global references
            global_G = global_mdl_param[:len(params_G)]
            global_C = global_mdl_param[len(params_G):]

            hist_G = hist_i[:len(params_G)]
            hist_C = hist_i[len(params_G):]

            state_diff_G = state_update_diff[:len(params_G)]
            state_diff_C = state_update_diff[len(params_G):]

            alpha_G = 1e-2
            alpha_C = 1e-4

            # FedDC losses (different strengths!)
            loss_cp_G = (alpha_G / 2) * torch.sum((params_G - (global_G - hist_G)) ** 2)
            loss_cp_C = (alpha_C / 2) * torch.sum((params_C - (global_C - hist_C)) ** 2)

            # Drift Correction Term: <theta, delta_theta>
            loss_cg_G = torch.sum(params_G * state_diff_G) / params_G.numel()
            loss_cg_C = torch.sum(params_C * state_diff_C) / params_C.numel()

            # Proximity Term: || theta - (Theta - h) ||^2
            loss_cp_G /= params_G.numel()
            loss_cp_C /= params_C.numel()

            total_G_loss = loss_G_adv + loss_cp_G + loss_cg_G
            total_C_loss = loss_C_adv + loss_C_sup + loss_cp_C + loss_cg_C

            total_G_loss.backward()
            total_C_loss.backward()

            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(C.parameters(), max_norm=1.0)
            
            opt_G.step()
            opt_C.step()

            # Accumulate values (convert to python floats to save memory)
            sup_loss_tot += loss_C_sup.item()
            adv_loss_tot += loss_C_adv.item()
            fed_loss_tot += total_C_loss.item()
            count += 1

    return G, C, sup_loss_tot/count, adv_loss_tot/count, fed_loss_tot/count
    
def train_model_FedDC(model, model_func, alpha, local_update_last, global_update_last, global_model_param, hist_i, trn_x, trn_y,
                    learning_rate, batch_size, epoch, print_per,
                    weight_decay, dataset_name, sch_step, sch_gamma):

    n_trn = trn_x.shape[0]
    state_update_diff = torch.tensor(-local_update_last+ global_update_last,  dtype=torch.float32, device=device)
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    #loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
        #for batch_x, batch_y in trn_gen:

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

def set_combined_params(G, C, combined_params, n_par_G):
    # Split the long vector back into G and C parts
    g_params = combined_params[:n_par_G]
    c_params = combined_params[n_par_G:]
    set_client_from_params(G, g_params)
    set_client_from_params(C, c_params)
    return G, C 

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

def get_combined_params(G, C, n_par):
    param_vec = np.zeros(n_par).astype('float32')
    # Fill with G params then C params
    g_params = get_mdl_params([G])[0]
    c_params = get_mdl_params([C])[0]
    return np.concatenate([g_params, c_params])

def get_acc_loss(data_x, data_y, model, dataset_name, w_decay = None):
    acc_overall = 0 
    loss_overall = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    batch_size = min(6000, data_x.shape[0])
    batch_size = min(2000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y), batch_size=batch_size, shuffle=False)
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
