from utils_libs import * 

'''def local_training(G_weights, C_weights):
    #get labelled data 
    #get unlabelled data 

    #update local discriminator
        #update copy of G
        #update copy of C

    #return updates'''

def local_training(G, C, D, alpha, global_mdl_params, hist_i, 
                             trn_x_labeled, trn_y_labeled, trn_x_unlabeled,
                             lr, batch_size, epochs):
    # G and C are local copies of Global models
    # D is the persistent local discriminator
    
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_C = torch.optim.Adam(C.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    for e in range(epochs):
        # 1. Prepare Data
        # Real Labeled: (x_l, y_l)
        # Synthetic: (G(z, y_fake), y_fake)
        # Pseudo: (x_unlabeled, C(x_unlabeled))
        
        # --- Train Discriminator D ---
        opt_D.zero_grad()
        
        # Loss on Real Data
        d_real = D(x_labeled, y_labeled)
        loss_d_real = torch.mean(F.softplus(-d_real))
        
        # Loss on Synthetic Data (from G)
        z = torch.randn(batch_size, G.z_dim).to(device)
        y_fake = torch.randint(0, 10, (batch_size,)).to(device)
        x_gen = G(z, y_fake)
        d_gen = D(x_gen.detach(), y_fake)
        loss_d_gen = torch.mean(F.softplus(d_gen))
        
        # Loss on Pseudo-labeled Data (from C)
        y_pseudo = torch.argmax(C(x_unlabeled), dim=1)
        d_pseudo = D(x_unlabeled, y_pseudo)
        loss_d_pseudo = torch.mean(F.softplus(d_pseudo))
        
        loss_D = loss_d_real + loss_d_gen + loss_d_pseudo
        loss_D.backward()
        opt_D.step()

        # --- Train Classifier C and Generator G with FedDC ---
        opt_C.zero_grad()
        opt_G.zero_grad()
        
        # Adversarial Losses
        loss_G_adv = torch.mean(F.softplus(-D(G(z, y_fake), y_fake)))
        loss_C_adv = torch.mean(F.softplus(-D(x_unlabeled, torch.argmax(C(x_unlabeled), dim=1))))
        
        # Standard Supervised Loss for C
        loss_C_sup = F.cross_entropy(C(x_labeled), y_labeled)
        
        # FedDC Penalty (Calculated on concatenated G and C parameters)
        curr_params = torch.cat([p.view(-1) for p in list(G.parameters()) + list(C.parameters())])
        # global_mdl_params and hist_i must be the concatenated G+C versions from the server
        loss_feddc = alpha/2 * torch.norm(curr_params - (global_mdl_params - hist_i))**2
        
        total_loss = loss_G_adv + loss_C_adv + loss_C_sup + loss_feddc
        total_loss.backward()
        opt_G.step()
        opt_C.step()
        
    return G, C 