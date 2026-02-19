from utils_libs import *
from utils_dataset import *
from utils_models import *
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import torchvision.utils as vutils
#from torchmetrics.image.fid import FrechetInceptionDistance
from scipy.linalg import fractional_matrix_power
import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    unlabeled_x = data_obj.unlabeled_x

    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt

    model_dir = ('%sModel/%s/%s' %(data_path, data_obj.name, suffix))
    if not os.path.exists(model_dir):
        os.makedirs('%sModel/%s/%s' %(data_path, data_obj.name, suffix), exist_ok=True)

    n_par_G = len(get_mdl_params([init_model_G])[0])
    n_par_C = len(get_mdl_params([init_model_C])[0])
    n_par = n_par_G + n_par_C

    parameter_drifts = np.zeros((n_clnt, n_par)).astype('float32')
    state_gadient_diffs = np.zeros((n_clnt + 1, n_par)).astype('float32') 
    clnt_params_list = np.zeros((n_clnt, n_par)).astype('float32')
    
    # Persistent local discriminators (not federated)
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
    avg_ins_mdls = list(range(n_save_instances))
    avg_all_mdls = list(range(n_save_instances))
    avg_cld_mdls = list(range(n_save_instances))

    trn_sel_clt_perf = np.zeros((com_amount, 2))
    tst_sel_clt_perf = np.zeros((com_amount, 2))

    trn_all_clt_perf = np.zeros((com_amount, 2))
    tst_all_clt_perf = np.zeros((com_amount, 2))

    trn_cur_cld_perf = np.zeros((com_amount, 2))
    tst_cur_cld_perf = np.zeros((com_amount, 2))

    writer = SummaryWriter('%sRuns/%s/%s' %(data_path, data_obj.name, suffix))

    #Initialize Inception 
    '''inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.fc = nn.Identity()
    inception_model.eval()

    # Pre-calculate Real Stats once
    print("Pre-calculating real image statistics for FID...")
    
    mu_real, sigma_real = precalculate_real_stats(data_obj.tst_x, inception_model, device)'''

    '''# store optimizer states for G, C, D
    client_opt_states = {
        client_id: {'G': None, 'C': None, 'D': None} 
        for client_id in range(n_clnt)
    }'''

    for i in range(com_amount):
        #print("round ",i)
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
            #print('---- Round %d, Training client %d' % (i, clnt))
            
            # Data Split (Semi-Supervised)
            #tx, ty = clnt_x[clnt], clnt_y[clnt]
            '''idx = np.random.permutation(len(ty))
            split = int(len(ty) * 0.7) # 10% Labeled
            #xl, yl = tx[idx[:split]], ty[idx[:split]]
            xl, yl = stratified_labeled_split(tx, ty, frac=0.7)
            xu = tx[idx[split:]]'''
            xl, yl = clnt_x[clnt], clnt_y[clnt]
            n_u = len(xl) * 2  
            idx = np.random.choice(len(unlabeled_x), n_u, replace=False)
            xu = unlabeled_x[idx]
            #xu = unlabeled_x
            #print("xl size:", xl.shape)
            #print("xu size:", xu.shape)

            local_G = model_func_G(data_obj.dataset).to(device)
            local_C = model_func_C(data_obj.dataset).to(device)
            set_combined_params(local_G, local_C, cld_mdl_param, n_par_G)

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
            delta_p = np.zeros(n_par)
            curr_par_C = get_mdl_params([local_C])[0]
            global_par_C = cld_mdl_param[n_par_G:] # Slicing out the C part
            delta_p_C = curr_par_C - global_par_C
            delta_p[n_par_G:] = delta_p_C
            #curr_par = get_combined_params(local_G, local_C, n_par)
            #delta_p = curr_par - cld_mdl_param
            parameter_drifts[clnt] += delta_p
            
            #beta = 1 / (n_minibatch * epoch) / (learning_rate * (lr_decay_per_round ** i))
            beta = 0.5
            #beta = 1.0 / (n_minibatch * epoch * learning_rate)
            state_g = state_gadient_diffs[clnt] - (state_gadient_diffs[-1]/weight_list[clnt]) + beta * (-delta_p)
            
            delta_g_sum += (state_g - state_gadient_diffs[clnt]) * weight_list[clnt]
            state_gadient_diffs[clnt] = state_g
            clnt_params_list[clnt] = get_combined_params(local_G, local_C, n_par)

            print(f" Client {clnt} | Sup: {s_loss:.4f} | Adv: {a_loss:.4f} | prox: {p_loss:.4f} | drift: {d_loss:.4f}")

        avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)
        state_gadient_diffs[-1] += (1 / n_clnt) * delta_g_sum
        cld_mdl_param = avg_mdl_param_sel + np.mean(parameter_drifts, axis=0)#avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)
        
        # Update global evaluation model (Classifier only for Accuracy)
        global_G, global_C =set_combined_params(cur_cld_G, cur_cld_C, cld_mdl_param, n_par_G)
        
        loss_t, acc_t = get_acc_loss(data_obj.tst_x, data_obj.tst_y, cur_cld_C, data_obj.dataset)
        tst_sel_clt_perf[i] = [loss_t, acc_t]
        #print(len(cld_mdl_param)
        writer.add_scalars(
            "accuracy",
            {"Accuracy fm2": acc_t},i)
        writer.add_scalars(
            "loss",
            {"Loss fm2": loss_t,},i)

        '''if i % 10 == 0:
            #fid = compute_fid(global_G, data_obj, device)
            fid = compute_fid(global_G, inception_model, mu_real, sigma_real, device, num_fake=5000)
            print(f"Round {i} | FID: {fid:.2f}")
            writer.add_scalars(
            "FID",
            {"FID gantest15": fid,},i)'''
        print("**** Round %d, Test Accuracy: %.4f, loss = %.4f" % (i+1, acc_t, loss_t))

        if (i + 1) % save_period == 0:
            save_path = os.path.join(model_dir, f'checkpoint_round_{i+1}.pt')

            torch.save({
                'round': i + 1,
                'model_G_state': cur_cld_G.state_dict(),
                'model_C_state': cur_cld_C.state_dict(),
            }, save_path)

            print(f"Saved checkpoint at round {i+1}")

    return cur_cld_C, tst_sel_clt_perf


def train_model_TripleFedDC(G, C, D, alpha, round_idx, data_obj,
                            local_update_last, global_update_last, global_mdl_param, hist_i, 
                            trn_x_labeled, trn_y_labeled, trn_x_unlabeled,
                            learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay): 
    sup_loss_tot, adv_loss_tot, prox_loss_tot, drift_loss_tot = 0, 0, 0, 0
    count = 0

    C_T = type(C)(data_obj.dataset).to(device) 
    C_T.load_state_dict(C.state_dict())
    for p in C_T.parameters(): p.requires_grad_(False)
    C_T.train()

    rampup_len   = 100  
    ramp_weight  = sigmoid_rampup(round_idx, rampup_len)

    alpha_G = alpha * 0.1 if round_idx > 20 else 0.0
    alpha_C = alpha * 0.1 if round_idx > 20 else 0.0

    opt_G = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    #opt_C = torch.optim.Adam(C.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=learning_rate * 0.2, betas=(0.5, 0.999))
    #opt_G = torch.optim.SGD(G.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    opt_C = torch.optim.SGD(C.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    #opt_D = torch.optim.SGD(D.parameters(), lr=learning_rate * 0.5, weight_decay=weight_decay, momentum=0.9)

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
            
            if label_l.dim() > 1 and label_l.shape[1] > 1:
                label_l = torch.argmax(label_l, dim=1)
            
            # 2. Handle Extra Dimensions (e.g., shape [32, 1] -> [32])
            if label_l.dim() > 1 and label_l.shape[1] == 1:
                label_l = label_l.squeeze()
            
            batch_sz = img_l.size(0) 
            #decay_factor = max(0, 1.0 - (round_idx / 50.0))
            #current_noise_std = 0.05 + (0.05 * decay_factor)

            ######## Train Discriminator D
            opt_D.zero_grad()

            #make discriminator inputs noisy
            #img_l_noisy = add_instance_noise(img_l, current_noise_std)
                
            #d_real = D(img_l, label_l.long())

            ### real - labelled data 
            with torch.no_grad():
                _, feat_l = C(img_l, return_features=True)
            
            d_real = D(img_l, label_l.long(), feat_l.detach())
            #d_real_loss = torch.mean(torch.nn.functional.softplus(1.0 - d_real))
            d_real_loss = torch.mean(F.relu(1.0 - d_real))
            y_wrong = torch.randint(0, 10, (batch_sz,), device=device)
            d_wrong_loss = torch.mean(F.softplus(D(img_l, y_wrong, feat_l.detach())))

            ## Fake data 
            
            # Synthetic Data from G
            z = torch.randn(batch_sz, G.z_dim).to(device)
            y_gen = torch.randint(0, 10, (batch_sz,)).to(device)
            img_fake = G(z, y_gen).detach()# detach so doesnt flow to g

            #img_fake_noisy = add_instance_noise(img_fake.detach(), current_noise_std)
            #img_u_noisy    = add_instance_noise(img_u, current_noise_std)
            # Get features for the fake image (D checks consistency of Fake Img + Fake Feat)
            with torch.no_grad():
                _, feat_fake = C(img_fake, return_features=True)

            #d_fake_loss = torch.mean(torch.nn.functional.softplus(1.0 + D(img_fake, y_gen, feat_fake.detach())))
            #d_fake_loss = torch.mean(torch.nn.functional.softplus(D(img_fake, y_gen)))
            d_fake =  D(img_fake, y_gen, feat_fake.detach())
            d_fake_loss = torch.mean(F.relu(1.0 + d_fake))

            ### Pseudo labelled data 
            #logits_u = C(img_u, return_features=True).detach()#classifier sees clean images
            with torch.no_grad():
                logits_u, feat_u = C(img_u, return_features=True)
            y_pseudo = torch.argmax(logits_u, dim=1)

            #probs_u = torch.softmax(logits_u, dim=1)
            #d_pseudo_loss = torch.mean(torch.nn.functional.softplus(D(img_u, y_pseudo)))
            d_pseudo_loss = torch.mean(torch.nn.functional.softplus(1.0 + D(img_u, y_pseudo, feat_u.detach())))#treat pseudo labelled data as real
            
            loss_D = d_real_loss + d_fake_loss + 0.5 * d_pseudo_loss
            #d_loss is too harsh early on 
            if round_idx > 10:
                loss_D += d_wrong_loss

            '''if count % 50 == 0:
                print(f"[D scores] real: {d_real.mean().item():.2f} | "
                    f"fake: {D(img_fake, y_gen, feat_fake.detach()).mean().item():.2f}")'''

            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=10.0)

            '''if count % 50 == 0:
                print(f"[D] loss: {loss_D.item():.4f} | grad_norm: {grad_norm(D):.4e}")'''

            opt_D.step()

            ########### Train Generator G 
            opt_G.zero_grad()

            #fresh z and y to prevent morisation collapse 
            z_g = torch.randn(batch_sz, G.z_dim, device=device)
            y_g = torch.randint(0, 10, (batch_sz,), device=device)
            fake_g = G(z_g, y_g)
            z2 = torch.randn(batch_sz, G.z_dim, device=device)
            fake_g2 = G(z2, y_g)

            lz_image = torch.mean(torch.abs(fake_g - fake_g2)) 
            lz_z     = torch.mean(torch.abs(z_g - z2))

            with torch.no_grad():
                _, feat_g = C(fake_g, return_features=True)
            
            eps = 1e-5
            # We want images to differ when Z differs.
            # Inverse distance: minimizing this MAXIMIZES the distance between images
            loss_diversity = 1.0 / (lz_image / (lz_z + eps) + eps)
            ratio = lz_image / (lz_z + eps)
            loss_diversity = torch.max(torch.tensor(0.0).to(device), 0.8 - ratio)

            #loss_G_adv = torch.mean(torch.nn.functional.softplus(-D(fake_g, y_g)))
            #fake_g_noisy = add_instance_noise(fake_g, current_noise_std)
            loss_G_adv = -torch.mean(D(fake_g, y_g, feat_g.detach()))

            params_G = torch.cat([p.view(-1) for p in G.parameters()])
            state_diff_G = state_update_diff[:len(params_G)]

            global_G = global_mdl_param[:len(params_G)]
            hist_G = hist_i[:len(params_G)]
            loss_cp_G = (alpha_G / 2) * torch.sum((params_G - (global_G - hist_G).detach()) ** 2)
            loss_cg_G = torch.sum(params_G * state_diff_G.detach())
            loss_cg_G = torch.clamp(loss_cg_G, min=-10.0, max=10.0)

            #class loss to get some class condtional differences in G images 
            logits = C(fake_g)
            loss_G_class = torch.nn.functional.cross_entropy(logits, y_g) * 0.5
            loss_G_class = torch.clamp(loss_G_class, max=2.0)

            fed_weight = max(0.1, 1.0 - round_idx / 600)
            loss_cg_G *= fed_weight

            total_G_loss = loss_G_adv + (loss_cp_G + loss_cg_G) + loss_G_class +loss_diversity

            '''with torch.no_grad():
                _, feat_g2 = C(fake_g, return_features=True)
            fake_feats = D(fake_g, y_g, feat_g2.detach())

            real_feats = D(img_l, label_l, feat_l.detach())
            #fake_feats = D(fake_g, y_g, feat_fake.detach())
            loss_feature_matching = torch.mean(torch.abs(real_feats.mean(0) - fake_feats.mean(0)))
            # Scale it (10.0 is usually a good starting weight)
            total_G_loss += 0.1 * loss_feature_matching'''
            
            '''if count % 50 == 0:
                print("[G breakdown]",
                    "adv:", loss_G_adv.item(),
                    "cp:", loss_cp_G.item(),
                    "cg:", loss_cg_G.item(),
                    "class",loss_G_class.item())
                    #"fm:", loss_feature_matching.item())'''

            total_G_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=10.0)

            '''if count % 50 == 0:
                print(f"[G] adv: {loss_G_adv.item():.4f} | "
                    f"cp: {loss_cp_G.item():.4f} | "
                    f"cg: {loss_cg_G.item():.4f} | "
                    f"class: {loss_G_class.item():.4f} | "
                    #f"fm: {loss_feature_matching.item():.4f} | "
                    f"grad_norm: {grad_norm(G):.4e}")'''

            # Discriminator Confidence
            # If D is outputting 50.0 or -50.0, the gradients for G will vanish
            #d_real_raw = D(img_l, label_l.long()).mean().item()
            #d_fake_raw = D(fake_g.detach(), y_g).mean().item()

            opt_G.step()



            ############ Train Classifier (C)
            opt_C.zero_grad()

            # Supervised Loss
            logits_l = C(img_l)
            #loss_C_sup = torch.nn.functional.cross_entropy(logits_l, label_l.long())
            loss_C_sup = F.cross_entropy(logits_l, label_l.long(), reduction='sum')
            loss_C_sup = loss_C_sup / batch_sz

            # SSL Loss (Mean Teacher + Masking)
            with torch.no_grad():
                logits_u_T = C_T(img_u)
                probs_u_T = torch.nn.functional.softmax(logits_u_T, dim=1)
                max_probs, _ = torch.max(probs_u_T, dim=1)
                mask = max_probs.ge(0.70).float()

            '''if count % 100 == 0:
                conf = max_probs.mean().item()
                used = mask.mean().item()
                print(f"[Pseudo] avg_conf: {conf:.3f} | used_frac: {used:.3f}")'''

            logits_u_S = C(img_u)
            probs_u_S = torch.nn.functional.softmax(logits_u_S, dim=1)

            # Consistency loss (MSE) - Note: .detach() on probs_u_T is redundant but safe
            loss_C_ssl = torch.mean(torch.sum((probs_u_S - probs_u_T)**2, dim=1) * mask)

            # Adversarial Loss 
            #y_pred_u = torch.argmax(probs_u_S, dim=1).detach() 
            #d_scores_u = D(img_u, y_pred_u)
            #loss_C_adv = torch.mean(torch.nn.functional.relu(1.0 - d_scores_u))
            logits_u_adv, feat_u_adv = C(img_u, return_features=True)
            y_pred_u_adv = torch.argmax(logits_u_adv, dim=1)
            d_scores_u = D(img_u, y_pred_u_adv.detach(), feat_u_adv)
            loss_C_adv = ramp_weight * torch.mean(torch.nn.functional.softplus(1.0 - d_scores_u))
            
            params_C = torch.cat([p.view(-1) for p in C.parameters()])
            global_C = global_mdl_param[len(params_G):]
            hist_C = hist_i[len(params_G):]
            state_diff_C = state_update_diff[len(params_G):]
            loss_cp_C = (alpha_C / 2) * torch.sum((params_C - (global_C - hist_C).detach()) ** 2)
            loss_cg_C = torch.sum(params_C * state_diff_C.detach())
            loss_cg_C = loss_cg_C / sum(p.numel() for p in C.parameters())# normalise by oarameter count to prevent from dominating 
            #loss_cg_C *= fed_weight

            '''loss_cg_G = 0
            loss_cg_C = 0'''

            # Classifier losses - balance supervised and semi-supervised
            loss_C_sup_weight = 1.0
            loss_C_ssl_weight = ramp_weight * 1.0  # Match supervised loss scale
            loss_C_adv_weight = ramp_weight * 0.01  # smaller than supervised

            total_C_loss = (loss_C_sup_weight * loss_C_sup + 
                            loss_C_ssl_weight * loss_C_ssl + 
                            loss_C_adv_weight * loss_C_adv + 
                            0.1 * loss_cp_C + 0.1 * loss_cg_C)

            #total_C_loss = loss_C_sup + loss_C_ssl + loss_C_adv + loss_cp_C + loss_cg_C

            '''if count % 50 == 0:
                print("[C breakdown]",
                    "sup:", loss_C_sup.item(),
                    "ssl:", loss_C_ssl.item(),
                    "adv:", loss_C_adv.item(),
                    "cp:", loss_cp_C.item(),
                    "cg:", loss_cg_C.item())'''

            total_C_loss.backward()
            torch.nn.utils.clip_grad_norm_(C.parameters(), max_norm=10.0)

            '''if count % 50 == 0:
                print(f"[C] sup: {loss_C_sup.item():.4f} | "
                    f"ssl: {loss_C_ssl.item():.4f} | "
                    f"adv: {loss_C_adv.item():.4f} | "
                    f"cp: {loss_cp_C.item():.4f} | "
                    f"cg: {loss_cg_C.item():.4f} | "
                    f"grad_norm: {grad_norm(C):.4e}")'''

            opt_C.step()

            # Update Teacher (EMA)
            update_ema(C, C_T, alpha=0.99)

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

def stratified_labeled_split(x, y, frac=0.1):
    xs, ys = [], []
    for c in range(10):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        k = max(1, int(len(idx) * frac))
        sel = np.random.choice(idx, k, replace=False)
        xs.append(x[sel])
        ys.append(y[sel])
    return np.concatenate(xs), np.concatenate(ys)

def save_gan_images(generator, round_idx, device):
    generator.eval()
    #generator.train()
    with torch.no_grad():
        # Generate 64 images (8x8 grid)
    
        n_per_class = 8
        y = torch.arange(10).repeat_interleave(n_per_class).to(device)
        z = torch.randn(len(y), generator.z_dim, device=device)

        fake_imgs = generator(z, y).detach().cpu()
        
        # Denormalize if necessary (assuming you normalized to [-1, 1])
        #fake_imgs = (fake_imgs + 1) / 2.0
        fake_imgs = fake_imgs * 0.5 + 0.5
        
        plt.figure(figsize=(8,10))
        plt.axis("off")
        plt.title(f"Generated Images - Round {round_idx}")
        plt.imshow(np.transpose(vutils.make_grid(fake_imgs, nrow=n_per_class, padding=2, normalize=False), (1,2,0)))
        plt.savefig(f"gen_images/gan_round_{round_idx}.png")
        plt.close()
    generator.train()


def add_instance_noise(images, std=0.1):
    if std <= 0:
        return images
    noise = torch.randn_like(images) * std
    return images + noise

def get_inception_features(imgs, model, device):
    """Extracts 2048-dim features from InceptionV3"""
    model.eval()
    # Inception expects 299x299 inputs
    # If your images are 32x32, we must upscale them
    imgs = nn.functional.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
    
    with torch.no_grad():
        features = model(imgs) # InceptionV3 returns a special object or logit
    return features.detach().cpu().numpy()

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
            fake = fake * 0.5 + 0.5
            
            # Inline extraction to avoid function call overhead
            fake_up = nn.functional.interpolate(fake, size=(299, 299), mode='bilinear', align_corners=False)
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
    # Faster matrix square root
    import scipy.linalg
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

def precalculate_real_stats(real_images, inception_model, device, batch_size=128):
    """
    Computes FID statistics for the real dataset once.
    real_images: numpy array or torch tensor
    """
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
            if batch.min() < 0:
                batch = batch * 0.5 + 0.5
            
            # Upscale and extract
            batch_up = nn.functional.interpolate(batch, size=(299, 299), 
                                               mode='bilinear', align_corners=False)
            feat = inception_model(batch_up)
            real_features.append(feat.cpu().numpy())
    
    real_features = np.concatenate(real_features, axis=0)
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    return mu_real, sigma_real

def update_ema(model, ema_model, alpha=0.99):
    """Updates the EMA teacher model."""
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

#debug, rememebr to remove 
def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5
