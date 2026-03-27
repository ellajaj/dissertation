from utils_libs import *
from utils_dataset import *
from utils_models import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_FedDC(data_obj, model_func_G, model_func_C, model_func_D,
                init_model_G, init_model_C, 
                act_prob, n_minibatch, learning_rate, batch_size, epoch, com_amount, print_per,
                weight_decay,  model_func, init_model, alpha_coef,
                sch_step, sch_gamma, save_period,
                suffix = '', trial=False, data_path='', rand_seed=0, lr_decay_per_round=1,
                blend_local_global=True, blend_momentum_G=0.95, blend_momentum_C=0.90):
    suffix  = 'FedDC_' + str(alpha_coef)+suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' %(save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' %rand_seed
    suffix += '_lrdecay%f' %lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x_l = data_obj.clnt_x_l
    clnt_y_l = data_obj.clnt_y_l
    clnt_x_u = data_obj.clnt_x_u

    trn_eval_x = np.concatenate([np.asarray(xc) for xc in clnt_x_l], axis=0)
    trn_eval_y = np.concatenate([np.asarray(yc) for yc in clnt_y_l], axis=0)

    if isinstance(trn_eval_x, np.ndarray) and trn_eval_x.dtype == object:
        trn_eval_x = np.stack([np.asarray(x) for x in trn_eval_x], axis=0)
    if isinstance(trn_eval_y, np.ndarray) and trn_eval_y.dtype == object:
        trn_eval_y = np.stack([np.asarray(y) for y in trn_eval_y], axis=0)

    weight_list = np.asarray([len(clnt_y_l[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt

    trn_curr_cld_perf = np.zeros((com_amount, 2))
    tst_curr_cld_perf = np.zeros((com_amount, 2))

    model_dir = ('%sModel/%s/%s' %(data_path, data_obj.name, suffix))
    if not os.path.exists(model_dir):
        os.makedirs('%sModel/%s/%s' %(data_path, data_obj.name, suffix), exist_ok=True)

    n_par_G = len(get_mdl_params([init_model_G])[0])
    n_par_C = len(get_mdl_params([init_model_C])[0])
    n_par = n_par_G + n_par_C

    parameter_drifts = np.zeros((n_clnt, n_par)).astype('float32')
    state_gadient_diffs = np.zeros((n_clnt + 1, n_par)).astype('float32') 
    clnt_params_list = np.zeros((n_clnt, n_par)).astype('float32')
    
    # local discriminators (not federated)
    local_Ds = [model_func_D(data_obj.dataset).to(device) for _ in range(n_clnt)]
    
    init_par_G = get_mdl_params([init_model_G])[0]
    init_par_C = get_mdl_params([init_model_C])[0]
    init_combined = np.concatenate([init_par_G, init_par_C])
    clnt_params_list = np.ones(n_clnt).reshape(-1, 1) * init_combined.reshape(1, -1)

    cur_cld_G = model_func_G(data_obj.dataset).to(device)
    cur_cld_C = model_func_C(data_obj.dataset).to(device)
    cur_cld_G.load_state_dict(init_model_G.state_dict())
    cur_cld_C.load_state_dict(init_model_C.state_dict())
    cld_mdl_param = init_combined.copy()

    n_save_instances = int(com_amount / save_period)

    writer = SummaryWriter('%sRuns/%s/%s' %(data_path, data_obj.name, suffix))
    #writer = SummaryWriter('/Results/Runs/final results')

    #Initialize Inception 
    #commented out as FID calculations are computationaly heavy
    '''inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.fc = nn.Identity()
    inception_model.eval()

    # Pre-calculate Real Stats once
    print("Pre-calculating real image statistics for FID...")
    
    mu_real, sigma_real = precalculate_real_stats(data_obj.tst_x, inception_model, device)'''

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
            
            # Data Split (Semi-Supervised)
            xl = clnt_x_l[clnt]
            yl = clnt_y_l[clnt]
            xu = clnt_x_u[clnt]

            local_G = model_func_G(data_obj.dataset).to(device)
            local_C = model_func_C(data_obj.dataset).to(device)
            if blend_local_global:
                # Blend client previous local state with current global state for a softer round start.
                prev_par = clnt_params_list[clnt]
                blended_par = cld_mdl_param.copy()
                blended_par[:n_par_G] = (
                    blend_momentum_G * cld_mdl_param[:n_par_G] +
                    (1.0 - blend_momentum_G) * prev_par[:n_par_G]
                )
                blended_par[n_par_G:] = (
                    blend_momentum_C * cld_mdl_param[n_par_G:] +
                    (1.0 - blend_momentum_C) * prev_par[n_par_G:]
                )
                set_combined_params(local_G, local_C, blended_par, n_par_G)
            else:
                set_combined_params(local_G, local_C, cld_mdl_param, n_par_G)

            #run local training of G, C and D
            local_G, local_C, s_loss, a_loss, p_loss, d_loss = train_model_TripleFedDC(
                local_G, local_C, local_Ds[clnt],
                alpha_coef / weight_list[clnt], i, data_obj,
                state_gadient_diffs[clnt], 
                state_gadient_diffs[-1] / weight_list[clnt],
                global_mdl_tensor, 
                torch.tensor(parameter_drifts[clnt], device=device),
                xl, yl, xu,
                learning_rate * (lr_decay_per_round ** i),
                sch_step, sch_gamma,
                batch_size, epoch, weight_decay,
            )

            # Update FedDC tracking    
            curr_par = get_combined_params(local_G, local_C, n_par)
            delta_p = curr_par - cld_mdl_param
            parameter_drifts[clnt] += delta_p
            
            beta = 0.05
            state_step = beta * (-delta_p)
            state_g = state_gadient_diffs[clnt] - (state_gadient_diffs[-1]/weight_list[clnt]) + state_step
            
            delta_g_sum += (state_g - state_gadient_diffs[clnt]) * weight_list[clnt]
            state_gadient_diffs[clnt] = state_g
            clnt_params_list[clnt] = curr_par

            print(f" Client {clnt} | Sup: {s_loss:.4f} | Adv: {a_loss:.4f} | prox: {p_loss:.4f} | drift: {d_loss:.4f}")

        avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)
        state_gadient_diffs[-1] += (1 / n_clnt) * delta_g_sum
        cld_mdl_param = avg_mdl_param_sel + np.mean(parameter_drifts, axis=0)#avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)
        
        # Update global evaluation model (Classifier only for Accuracy)
        global_G, global_C = set_combined_params(cur_cld_G, cur_cld_C, cld_mdl_param, n_par_G)
        
        ##get test accuracy
        loss_t, acc_t = get_acc_loss(data_obj.tst_x, data_obj.tst_y, cur_cld_C, data_obj.dataset)
        tst_curr_cld_perf[i] = [loss_t, acc_t]

        writer.add_scalars(
            "accuracy/test",
            {"SSFL": acc_t},i)
        writer.add_scalars(
            "Loss/test",
            {"SSFL": loss_t,},i)

        ##get train accuracy
        loss_trn, acc_trn = get_acc_loss(trn_eval_x, trn_eval_y, cur_cld_C, data_obj.dataset)
        trn_curr_cld_perf[i] = [loss_trn, acc_trn]
        writer.add_scalars(
            "accuracy/train",
            {"SSFL": acc_trn},i)
        writer.add_scalars(
            "Loss/train",
            {"SSFL": loss_trn,},i)

        #commented out as FID calculations are computationally heavy
        '''if i % 10 == 0:
            #fid = compute_fid(global_G, data_obj, device)
            fid = compute_fid(global_G, inception_model, mu_real, sigma_real, device, num_fake=5000)
            print(f"Round {i} | FID: {fid:.2f}")
            writer.add_scalars(
            "FID",
            {"cifar ssl 0.4 nogen": fid,},i)
        print("**** Round %d, Test Accuracy: %.4f, Train accuracy: %.4f, loss = %.4f" % (i+1, acc_t, acc_trn, loss_t))'''

        if (i + 1) % save_period == 0:
            save_path = os.path.join(model_dir, f'checkpoint_round_{i+1}.pt')

            torch.save({
                'round': i + 1,
                'model_G_state': cur_cld_G.state_dict(),
                'model_C_state': cur_cld_C.state_dict(),
            }, save_path)

            print(f"Saved checkpoint at round {i+1}")

    return cur_cld_C, tst_curr_cld_perf


def train_model_TripleFedDC(G, C, D, alpha, round_idx, data_obj,
                            local_update_last, global_update_last, global_mdl_param, hist_i, 
                            trn_x_labeled, trn_y_labeled, trn_x_unlabeled,
                            learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay): 
    sup_loss_tot, adv_loss_tot, prox_loss_tot, drift_loss_tot = 0, 0, 0, 0
    count = 0

    n_cls = data_obj.n_cls if hasattr(data_obj, "n_cls") else 10

    ssl_round = max(0, round_idx - 40)
    ramp_weight = sigmoid_rampup(ssl_round, 100)

    g_freeze_rounds = 50
    d_feat_warmup_start = 20
    ssl_warmup_rounds = 40
    d_feat_full_round = 120

    # Keep FedDC pressure off G during warm start, then reintroduce gently.
    alpha_G = 0.02 * alpha if round_idx >= 120 else 0.0
    alpha_C = alpha if round_idx > 20 else 0.0

    opt_G = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    c_lr_mult = 50.0 if data_obj.dataset == "CIFAR10" else 20.0
    opt_C = torch.optim.SGD(C.parameters(), lr=learning_rate * c_lr_mult, weight_decay=weight_decay, momentum=0.9, nesterov=True)

    sched_G = torch.optim.lr_scheduler.StepLR(opt_G, step_size=sch_step, gamma=sch_gamma)
    sched_C = torch.optim.lr_scheduler.StepLR(opt_C, step_size=sch_step, gamma=sch_gamma)
    sched_D = torch.optim.lr_scheduler.StepLR(opt_D, step_size=sch_step, gamma=sch_gamma)

    state_update_diff = torch.tensor(-local_update_last + global_update_last, dtype=torch.float32, device=device)
    
    trn_gen = torch.utils.data.DataLoader(
        TripleGANDataset(trn_x_labeled, trn_y_labeled, trn_x_unlabeled, data_obj.dataset), 
        batch_size=batch_size, shuffle=True
    )
    
    G.train(); C.train(); D.train()

    for e in range(epoch):
        for img_l, label_l, img_u in trn_gen:
            img_l, label_l, img_u = img_l.to(device), label_l.to(device), img_u.to(device)
            if round_idx < d_feat_warmup_start:
                feat_gate = 0.0
            else:
                feat_gate = min(1.0, float(round_idx - d_feat_warmup_start) / float(d_feat_full_round - d_feat_warmup_start))
            
            if label_l.dim() > 1 and label_l.shape[1] > 1:
                label_l = torch.argmax(label_l, dim=1)
            
            if label_l.dim() > 1 and label_l.shape[1] == 1:
                label_l = label_l.squeeze()
            
            batch_sz = img_l.size(0) 

            #####################################
            #Train Discriminator D 
            #####################################
            opt_D.zero_grad()

            ### real - labelled data 
            with torch.no_grad():
                _, feat_l = C(img_l, return_features=True)
            feat_l_d = feat_l.detach() * feat_gate
            
            d_real = D(img_l, label_l.long(), feat_l_d)
            d_real_loss = torch.mean(F.relu(1.0 - d_real))
            y_wrong = torch.randint(0, n_cls, (batch_sz,), device=device)
            y_wrong = (y_wrong + (y_wrong == label_l.long()).long()) % n_cls
            d_wrong_loss = torch.mean(F.softplus(D(img_l, y_wrong, feat_l_d)))

            ## Fake data 
            # Generate one fake batch per step and reuse it across losses.
            z = torch.randn(batch_sz, G.z_dim, device=device)
            y_gen = torch.randint(0, n_cls, (batch_sz,), device=device)
            img_fake = G(z, y_gen)

            # Get features for the fake image (D checks consistency of Fake Img + Fake Feat)
            with torch.no_grad():
                _, feat_fake = C(img_fake, return_features=True)
            feat_fake_d = feat_fake.detach() * feat_gate

            d_fake =  D(img_fake.detach(), y_gen, feat_fake_d)
            d_fake_loss = torch.mean(F.relu(1.0 + d_fake))

            ### Pseudo labelled data 
            #logits_u = C(img_u, return_features=True).detach()#classifier sees clean images
            with torch.no_grad():
                logits_u, feat_u = C(img_u, return_features=True)
            y_pseudo = torch.argmax(logits_u, dim=1)
            pseudo_conf = torch.softmax(logits_u, dim=1).max(dim=1)[0]
            if round_idx < ssl_warmup_rounds + 20:
                d_pseudo_thr = 0.80
            elif round_idx < ssl_warmup_rounds + 60:
                d_pseudo_thr = 0.75
            else:
                d_pseudo_thr = 0.70
            pseudo_mask = (pseudo_conf >= d_pseudo_thr).float()

            # Pseudo-labeled unlabeled data should be treated as real only when confidence is high.
            feat_u_d = feat_u.detach() * feat_gate
            d_pseudo_raw = F.softplus(-D(img_u, y_pseudo, feat_u_d))
            d_pseudo_loss = (d_pseudo_raw * pseudo_mask).sum() / torch.clamp(pseudo_mask.sum(), min=1.0)
            gan_decay = 1.0 if round_idx < 120 else max(0.2, 1.0 - (round_idx - 120) / 200.0)
            loss_D = d_real_loss + gan_decay * (d_fake_loss + 0.15 * ramp_weight * d_pseudo_loss)
            
            #d_loss is too harsh early on 
            if round_idx > 10:
                loss_D += d_wrong_loss

            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=10.0)

            opt_D.step()

            ###############################
            # Train Generator G 
            ###############################
            fed_weight = max(0.05, 0.25 * (1.0 - round_idx / 600))

            params_G = torch.cat([p.view(-1) for p in G.parameters()])
            state_diff_G = state_update_diff[:len(params_G)]

            global_G = global_mdl_param[:len(params_G)]
            hist_G = hist_i[:len(params_G)]
            loss_cp_G = (alpha_G / 2) * torch.sum((params_G - (global_G - hist_G).detach()) ** 2)
            loss_cg_G = torch.sum(params_G * state_diff_G.detach())
            loss_cg_G = torch.clamp(loss_cg_G, min=-10.0, max=10.0)

            if round_idx >= g_freeze_rounds:
                opt_G.zero_grad()

                fake_g = img_fake

                _, feat_g = C(fake_g, return_features=True)
                feat_g_d = feat_g.detach() * feat_gate

                loss_G_adv = -torch.mean(D(fake_g, y_gen, feat_g_d))

                # Keep FedDC influence on G small even after warm start.
                loss_cg_G *= fed_weight
                loss_cp_G *= fed_weight

                # Smoothly ramp adversarial updates after warm start.
                g_adv_weight = min(1.0, float(round_idx - g_freeze_rounds + 1) / 50.0)
                total_G_loss = loss_G_adv + (loss_cp_G + loss_cg_G)

                total_G_loss.backward()
                opt_G.step()


            ##########################
            # Train Classifier (C)
            ##########################
            opt_C.zero_grad()

            # Supervised Loss
            logits_l = C(img_l)
            loss_C_sup = torch.nn.functional.cross_entropy(logits_l, label_l.long())
            
            logits_u = C(img_u)
            probs_u = torch.softmax(logits_u, dim=1)

            max_probs, pseudo_labels = torch.max(probs_u, dim=1)
            if round_idx < ssl_warmup_rounds + 20:
                c_pseudo_thr = 0.80
            elif round_idx < ssl_warmup_rounds + 60:
                c_pseudo_thr = 0.75
            else:
                c_pseudo_thr = 0.70
            mask = (max_probs >= c_pseudo_thr).float()

            # Pseudo-label SSL loss (masked CE). The previous expression was always zero.
            ssl_ce = F.cross_entropy(logits_u, pseudo_labels.detach(), reduction='none')
            loss_C_ssl = ramp_weight * (ssl_ce * mask).sum() / torch.clamp(mask.sum(), min=1.0)

            # Adversarial Loss 
            y_pred_u = torch.argmax(probs_u, dim=1).detach() 
            d_scores_u = D(img_u, y_pred_u, feat_u_d)
            loss_C_adv = torch.mean(torch.nn.functional.relu(1.0 - d_scores_u))
            
            params_C = torch.nn.utils.parameters_to_vector(C.parameters())
            global_C = global_mdl_param[len(params_G):]
            hist_C = hist_i[len(params_G):]
            state_diff_C = state_update_diff[len(params_G):]

            loss_cp_C = (alpha_C / 2) * torch.sum((params_C - (global_C - hist_C).detach()) ** 2)
            loss_cg_C = torch.sum(params_C * state_diff_C.detach())
            loss_cg_C = torch.clamp(loss_cg_C, min=-10.0, max=10.0)
            loss_cg_C *= fed_weight

            # Keep classifier anchored to supervised/SSL signals; adversarial pressure is a small regularizer.
            c_adv_weight = 0.05 * ramp_weight * gan_decay
            total_C_loss = loss_C_sup + loss_C_ssl + c_adv_weight * loss_C_adv + 0.1 * (loss_cp_C + loss_cg_C)

            total_C_loss.backward()
            opt_C.step()

            count += 1
            sup_loss_tot += loss_C_sup.item()
            prox_loss_tot += loss_cp_C.item()
            drift_loss_tot += loss_cg_C.item() 
            adv_loss_tot += loss_C_adv.item()

        sched_G.step()
        sched_C.step()
        sched_D.step()

    save_gan_images(G, round_idx, device)
    return G, C, sup_loss_tot/count, adv_loss_tot/count, prox_loss_tot/count, drift_loss_tot/count
    
def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx+length].reshape(weights.shape)).to(device))
        idx += length

    mdl.load_state_dict(dict_param, strict=False)
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
    # Normalize potentially ragged/object client arrays into dense numeric arrays.
    data_x = np.asarray(data_x)
    if data_x.dtype == object:
        elems_x = [np.asarray(x) for x in data_x]
        try:
            data_x = np.stack(elems_x, axis=0)
        except ValueError:
            data_x = np.concatenate(elems_x, axis=0)

    data_y = np.asarray(data_y)
    if data_y.dtype == object:
        data_y = np.concatenate([np.asarray(y).reshape(-1) for y in data_y], axis=0)
    data_y = np.asarray(data_y).reshape(-1).astype(np.int64)

    acc_overall = 0 
    loss_overall = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    batch_size = min(6000, data_x.shape[0])
    batch_size = min(2000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(dataset_name, data_x, data_y), batch_size=batch_size, shuffle=False)
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
    eval_loader = data.DataLoader(
        Dataset(data_obj.dataset, data_obj.tst_x, data_obj.tst_y),
        batch_size=batch_size,
        shuffle=False
    )
    with torch.no_grad():
        for batch_x, batch_y in eval_loader:
            inputs = batch_x.to(device)
            outputs = model(inputs)

            if outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs.view(-1)).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
            else:
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

            y_true.extend(batch_y.reshape(-1).cpu().numpy())
            y_pred.extend(preds)
            y_prob.append(probs)

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
            yt, yp, _ = evaluate_client(model, Xc, Yc, device, batch_size, data_obj.dataset)
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

def evaluate_client(model, X, Y, device, batch_size=64, dataset_name=None):
    model.eval()
    yt, yp, yprob = [], [], []
    if dataset_name is None:
        sample = np.asarray(X[0])
        if sample.ndim == 3 and (sample.shape[0] == 1 or sample.shape[0] == 3):
            dataset_name = "fashion_mnist" if sample.shape[0] == 1 else "CIFAR10"
        elif sample.ndim == 2:
            dataset_name = "fashion_mnist"
        else:
            dataset_name = "CIFAR10"
    eval_loader = data.DataLoader(
        Dataset(dataset_name, X, Y),
        batch_size=batch_size,
        shuffle=False
    )
    with torch.no_grad():
        for bx, by in eval_loader:
            inp = bx.to(device)
            out = model(inp)
            if out.shape[1] == 1:
                prob = torch.sigmoid(out.view(-1)).cpu().numpy()
                pred = (prob >= 0.5).astype(int)
            else:
                prob = torch.softmax(out, dim=1).cpu().numpy()
                pred = np.argmax(prob, axis=1)
            yt.extend(by.reshape(-1).cpu().numpy())
            yp.extend(pred)
            yprob.append(prob)
    return np.array(yt), np.array(yp), np.concatenate(yprob, axis=0)

# --- Helper functions

def save_gan_images(generator, round_idx, device):
    generator.eval()
    with torch.no_grad():
        n_per_class = 8
        y = torch.arange(10).repeat_interleave(n_per_class).to(device)
        z = torch.randn(len(y), generator.z_dim, device=device)

        fake_imgs = generator(z, y).detach().cpu()
        fake_imgs = fake_imgs * 0.5 + 0.5
        grid = vutils.make_grid(fake_imgs, nrow=n_per_class, padding=0, normalize=False)

        plt.figure(figsize=(8,10))
        plt.axis("off")
        plt.title(f"Generated Images - Round {round_idx}")
        if grid.shape[0] == 1:
            plt.imshow(grid.squeeze(0), cmap="gray", vmin=0.0, vmax=1.0)
        else:
            plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.savefig(f"gen_images/gan_round_{round_idx}.png")
        plt.close()
    generator.train()

def compute_fid(generator, inception_model, mu_real, sigma_real, device, num_fake=5000):
    generator.eval()
    fake_features = []
    
    # Increase batch size for inference to saturate GPU
    inf_batch_size = 128 
    
    n = 0
    with torch.no_grad():
        while n < num_fake:
            z = torch.randn(inf_batch_size, generator.z_dim, device=device)
            y = torch.randint(0, 10, (inf_batch_size,), device=device)
            
            fake = generator(z, y)
            fake = (fake * 0.5 + 0.5).clamp(0,1)
            
            # Inline extraction to avoid function call overhead
            fake_up = nn.functional.interpolate(fake, size=(299, 299), mode='bilinear', align_corners=False)

            # ImageNet normalization (required for pretrained Inception)
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
            fake_up = (fake_up - mean) / std

            feat = inception_model(fake_up)
            
            fake_features.append(feat.cpu().numpy())
            n += inf_batch_size

    fake_features = np.concatenate(fake_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    
    generator.train()
    return fid_score

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2

    import scipy.linalg
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

def precalculate_real_stats(real_images, inception_model, device, batch_size=128):
    inception_model.eval()
    real_features = []
    
    # Ensure real_images is a tensor
    if isinstance(real_images, np.ndarray):
        real_images = torch.from_numpy(real_images).float()
    
    # Split into batches
    batches = torch.split(real_images, batch_size)
    
    print(f"Extracting features from {real_images.shape[0]} real images...")
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)
            # Normalize [-1, 1] -> [0, 1] if necessary
            #if batch.min() < 0:
                #batch = batch * 0.5 + 0.5

            # Undo CIFAR normalization
            cifar_mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1,3,1,1)
            cifar_std  = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1,3,1,1)

            batch = batch * cifar_std + cifar_mean
            batch = batch.clamp(0,1)
            
            # Upscale and extract
            batch_up = nn.functional.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)

            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
            batch_up = (batch_up - mean) / std

            feat = inception_model(batch_up)
            real_features.append(feat.cpu().numpy())
    
    real_features = np.concatenate(real_features, axis=0)
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    return mu_real, sigma_real

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def _clip_np_by_value(x, clip_value):
    return np.clip(x, -clip_value, clip_value)
