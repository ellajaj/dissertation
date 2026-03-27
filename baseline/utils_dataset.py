'''code adapted from https://github.com/gaoliang13/FedDC -- MIT license'''
from utils_libs import *

class DatasetObject:
    def __init__(self, dataset, n_client, seed, rule, unbalanced_sgm=0, rule_arg='', data_path=''):
        self.dataset  = dataset
        self.n_client = n_client
        self.rule     = rule
        self.rule_arg = rule_arg
        self.seed     = seed
        rule_arg_str = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        self.name = "%s_%d_%d_%s_%s" %(self.dataset, self.n_client, self.seed, self.rule, rule_arg_str)
        self.name += '_%f' %unbalanced_sgm if unbalanced_sgm!=0 else ''
        self.unbalanced_sgm = unbalanced_sgm
        self.data_path = data_path
        self.set_data()
        # self.limit_dataset(max_samples=60000, min_per_class=100, verbose=False) # Moved outside __init__ for more control

    def _redistribute_to_clients(self, trn_x_to_split, trn_y_to_split):
        """Internal helper to split (trn_x, trn_y) into client-specific (clnt_x, clnt_y)."""
        np.random.seed(self.seed)

        # Shuffle Data (for consistency, even if already shuffled)
        rand_perm = np.random.permutation(len(trn_y_to_split))
        trn_x_to_split = trn_x_to_split[rand_perm]
        trn_y_to_split = trn_y_to_split[rand_perm]

        n_data_per_clnt = int((len(trn_y_to_split)) / self.n_client)
        clnt_data_list = (np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=self.unbalanced_sgm, size=self.n_client)).astype(int)
        clnt_data_list = (clnt_data_list/np.sum(clnt_data_list)*len(trn_y_to_split)).astype(int)
        diff = np.sum(clnt_data_list) - len(trn_y_to_split)

        if diff!= 0:
            for clnt_i in range(self.n_client):
                if clnt_data_list[clnt_i] > diff:
                    clnt_data_list[clnt_i] -= diff
                    break

        if self.rule == 'Drichlet':
            print('splitting data using dirichlet distribution, alpha = %s' % (self.rule_arg))

            idx_list = [np.where(trn_y_to_split == c)[0] for c in range(self.n_cls)]
            for c in range(self.n_cls):
                np.random.shuffle(idx_list[c])

            class_client_priors = np.random.dirichlet(
                [self.rule_arg] * self.n_client,
                size=self.n_cls
            )
            clnt_x = [[] for _ in range(self.n_client)]
            clnt_y = [[] for _ in range(self.n_client)]

            for c in range(self.n_cls):
                idx_c = idx_list[c]
                n_c = len(idx_c)

                alloc_float = class_client_priors[c] * n_c
                alloc_int = np.floor(alloc_float).astype(int)

                leftover = n_c - alloc_int.sum()
                if leftover > 0:
                    frac = alloc_float - alloc_int
                    top = np.argsort(-frac)[:leftover]
                    alloc_int[top] += 1

                start = 0
                for i in range(self.n_client):
                    end = start + alloc_int[i]
                    if end > start:
                        clnt_x[i].append(trn_x_to_split[idx_c[start:end]])
                        clnt_y[i].append(trn_y_to_split[idx_c[start:end]])
                    start = end

            for i in range(self.n_client):
                if len(clnt_x[i]) > 0:
                    clnt_x[i] = np.concatenate(clnt_x[i], axis=0)
                    clnt_y[i] = np.concatenate(clnt_y[i], axis=0)
                else:
                    clnt_x[i] = np.zeros((0, self.channels, self.height, self.width), dtype=np.float32)
                    clnt_y[i] = np.zeros((0,), dtype=np.int64)

            #clnt_x = np.array(clnt_x, dtype=object)
            #clnt_y = np.array(clnt_y, dtype=object)

            cls_means = np.zeros((self.n_client, self.n_cls))
            for i in range(self.n_client):
                for c in range(self.n_cls):
                    cls_means[i, c] = np.mean(clnt_y[i] == c)

            prior_real_diff = np.abs(cls_means - class_client_priors.T)

            print('--- Max deviation from prior: %.4f' % np.max(prior_real_diff))
            print('--- Min deviation from prior: %.4f' % np.min(prior_real_diff))


        elif self.rule == 'iid' and self.dataset == 'CIFAR100' and self.unbalanced_sgm==0:
            assert len(trn_y_to_split)//100 % self.n_client == 0

            idx = np.argsort(trn_y_to_split[:, 0])
            n_data_per_clnt_actual = len(trn_y_to_split) // self.n_client
            clnt_x = np.zeros((self.n_client, n_data_per_clnt_actual, 3, 32, 32), dtype=np.float32)
            clnt_y = np.zeros((self.n_client, n_data_per_clnt_actual, 1), dtype=np.float32)
            trn_x_to_split = trn_x_to_split[idx]
            trn_y_to_split = trn_y_to_split[idx]
            n_cls_sample_per_device = n_data_per_clnt_actual // 100
            for i in range(self.n_client):
                for j in range(100):
                    clnt_x[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :, :, :] = trn_x_to_split[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :, :, :]
                    clnt_y[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :] = trn_y_to_split[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :]


        elif self.rule == 'iid':
            clnt_x = [ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
            clnt_y = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ]

            clnt_data_list_cum_sum = np.concatenate(([0], np.cumsum(clnt_data_list)))
            for clnt_idx_ in range(self.n_client):
                clnt_x[clnt_idx_] = trn_x_to_split[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]
                clnt_y[clnt_idx_] = trn_y_to_split[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]

            #clnt_x = np.array(clnt_x, dtype=object)
            #clnt_y = np.array(clnt_y, dtype=object)

        self.clnt_x = clnt_x
        self.clnt_y = clnt_y

        print('Class frequencies:')
        for clnt in range(self.n_client):
            print("Client %3d: " %clnt +
                  ', '.join(["%.3f" %np.mean(self.clnt_y[clnt]==cls) for cls in range(self.n_cls)]) +
                  ', Amount:%d' %self.clnt_y[clnt].shape[0])

    def set_data(self):
        # Prepare data if not ready
        if not os.path.exists('%sData/%s' %(self.data_path, self.name)):
            # Get Raw data
            if self.dataset == 'mnist':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                trnset = torchvision.datasets.MNIST(root='%sData/Raw' %self.data_path,
                                                    train=True , download=True, transform=transform)
                tstset = torchvision.datasets.MNIST(root='%sData/Raw' %self.data_path,
                                                    train=False, download=True, transform=transform)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=60000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;

            if self.dataset == 'CIFAR10':
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])

                trnset = torchvision.datasets.CIFAR10(root='%sData/Raw' %self.data_path,
                                                      train=True , download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR10(root='%sData/Raw' %self.data_path,
                                                      train=False, download=True, transform=transform)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;

            if self.dataset == 'CIFAR100':
                print(self.dataset)
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                                     std=[0.2675, 0.2565, 0.2761])])
                trnset = torchvision.datasets.CIFAR100(root='%sData/Raw' %self.data_path,
                                                      train=True , download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR100(root='%sData/Raw' %self.data_path,
                                                      train=False, download=True, transform=transform)
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=0)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=0)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;

            if self.dataset == 'fashion_mnist':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
                trnset = torchvision.datasets.FashionMNIST(root='%sData/Raw' %self.data_path,
                                                           train=True, download=True, transform=transform)
                tstset = torchvision.datasets.FashionMNIST(root='%sData/Raw' %self.data_path,
                                                           train=False, download=True, transform=transform)
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=60000, shuffle=False, num_workers =1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;


            if self.dataset != 'emnist':
                trn_itr = trn_load.__iter__(); tst_itr = tst_load.__iter__()
                trn_x, trn_y = trn_itr.__next__()
                tst_x, tst_y = tst_itr.__next__()

                trn_x = trn_x.numpy(); trn_y = trn_y.numpy().reshape(-1,1)
                tst_x = tst_x.numpy(); tst_y = tst_y.numpy().reshape(-1,1)


            if self.dataset == 'emnist':
                emnist = io.loadmat(self.data_path + "Data/Raw/matlab/emnist-letters.mat")
                x_train = emnist["dataset"][0][0][0][0][0][0]
                x_train = x_train.astype(np.float32)
                y_train = emnist["dataset"][0][0][0][0][0][1] - 1
                trn_idx = np.where(y_train < 10)[0]
                y_train = y_train[trn_idx]
                x_train = x_train[trn_idx]
                mean_x = np.mean(x_train)
                std_x = np.std(x_train)

                x_test = emnist["dataset"][0][0][1][0][0][0]
                x_test = x_test.astype(np.float32)
                y_test = emnist["dataset"][0][0][1][0][0][1] - 1
                tst_idx = np.where(y_test < 10)[0]
                y_test = y_test[tst_idx]
                x_test = x_test[tst_idx]

                x_train = x_train.reshape((-1, 1, 28, 28))
                x_test  = x_test.reshape((-1, 1, 28, 28))

                trn_x = (x_train - mean_x) / std_x
                trn_y = y_train

                tst_x = (x_test  - mean_x) / std_x
                tst_y = y_test

                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;

            # Set initial trn_x and trn_y (full dataset)
            self.trn_x = trn_x
            self.trn_y = trn_y
            self.tst_x = tst_x
            self.tst_y = tst_y

            # Redistribute to clients using the defined rule
            self._redistribute_to_clients(self.trn_x, self.trn_y.reshape(-1, 1)) # Pass reshaped trn_y

            # Save data
            os.makedirs('%sData/%s' %(self.data_path, self.name), exist_ok=True)

            np.save('%sData/%s/clnt_x.npy' %(self.data_path, self.name), self.clnt_x)
            np.save('%sData/%s/clnt_y.npy' %(self.data_path, self.name), self.clnt_y)

            np.save('%sData/%s/tst_x.npy'  %(self.data_path, self.name),  tst_x)
            np.save('%sData/%s/tst_y.npy'  %(self.data_path, self.name),  tst_y)
            # Also save trn_x and trn_y for consistency, if needed later
            np.save('%sData/%s/trn_x.npy'  %(self.data_path, self.name),  self.trn_x)
            np.save('%sData/%s/trn_y.npy'  %(self.data_path, self.name),  self.trn_y)

        else:
            print("Data is already downloaded")
            self.clnt_x = np.load('%sData/%s/clnt_x.npy' %(self.data_path, self.name),allow_pickle=True)
            self.clnt_y = np.load('%sData/%s/clnt_y.npy' %(self.data_path, self.name),allow_pickle=True)
            self.n_client = len(self.clnt_x)

            self.tst_x  = np.load('%sData/%s/tst_x.npy'  %(self.data_path, self.name),allow_pickle=True)
            self.tst_y  = np.load('%sData/%s/tst_y.npy'  %(self.data_path, self.name),allow_pickle=True)

            # Reconstruct trn_x and trn_y from clnt_x and clnt_y for existing data (full dataset)
            # If you want the loaded trn_x/y to be already limited, you'd need to save them limited or re-limit them here.
            self.trn_x = np.concatenate(self.clnt_x, axis=0)
            self.trn_y = np.concatenate(self.clnt_y, axis=0)

            if self.dataset == 'mnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'CIFAR10':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
            if self.dataset == 'CIFAR100':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;
            if self.dataset == 'fashion_mnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'emnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;

        print('Class frequencies:')
        count = 0
        for clnt in range(self.n_client):
            print("Client %3d: " %clnt +
                  ', '.join(["%.3f" %np.mean(self.clnt_y[clnt]==cls) for cls in range(self.n_cls)]) +
                  ', Amount:%d' %self.clnt_y[clnt].shape[0])
            count += self.clnt_y[clnt].shape[0]



    def limit_dataset(self, max_samples, min_per_class=10, verbose=True):
          """
          Reduce training set size while keeping class imbalance.
          Ensures each class has at least `min_per_class` samples.
          After limiting the overall dataset, it re-distributes this limited data
          among clients based on the original splitting rule.

          Args:
              max_samples (int): Maximum total samples to keep.
              min_per_class (int): Minimum samples per class.
              verbose (bool): Print diagnostics.
          """

          trn_x_original = self.trn_x
          trn_y_original = self.trn_y.reshape(-1)

          #trn_y_original = np.array(trn_y_original, dtype=np.int64)
          trn_y_original = np.asarray(trn_y_original).astype(np.int64)


          N = len(trn_y_original)
          if max_samples >= N:
              if verbose:
                  print(f"[limit_dataset] Requested max_samples={max_samples}, "
                        f"dataset has only {N}. No limiting done.")
              # Still redistribute to clients to ensure initial split if not done yet
              # self._redistribute_to_clients(trn_x_original, trn_y_original.reshape(-1,1))
              return

          np.random.seed(self.seed)

          # 1. Show original distribution
          if verbose:
              print("\n[limit_dataset] Original per-class counts:")
              orig_counts = np.bincount(trn_y_original, minlength=self.n_cls)
              for c in range(self.n_cls):
                  print(f"  class {c}: {orig_counts[c]}")

          # 2. Build per-class index pools
          cls_idx = [np.where(trn_y_original == c)[0] for c in range(self.n_cls)]
          for c in range(self.n_cls):
              np.random.shuffle(cls_idx[c])

          # 3. Ensure per-class minimum
          kept_idx = []

          for c in range(self.n_cls):
              available = len(cls_idx[c])
              need = min(min_per_class, available)
              kept_idx.append(cls_idx[c][:need])
              cls_idx[c] = cls_idx[c][need:]   # leftover pool

          kept_idx = list(kept_idx)
          used = sum(len(x) for x in kept_idx)
          remaining_quota = max_samples - used

          if remaining_quota <= 0:
              # Only mins fit — trim to exactly max_samples
              kept_idx_flat = np.concatenate(kept_idx)
              if len(kept_idx_flat) > max_samples:
                  kept_idx_flat = kept_idx_flat[:max_samples]
              kept_idx = kept_idx_flat
          else:
              # 4. Allocate the remaining quota proportionally to original class frequencies
              orig_counts = np.bincount(trn_y_original, minlength=self.n_cls)
              prop = orig_counts / orig_counts.sum()
              extra_per_class = np.floor(prop * remaining_quota).astype(int)

              # Adjust rounding so total matches remaining_quota
              diff = remaining_quota - extra_per_class.sum()
              if diff > 0:
                  # allocate leftovers by largest fractional parts
                  frac = (prop * remaining_quota) - extra_per_class
                  order = np.argsort(-frac)
                  for i in order[:diff]:
                      extra_per_class[i] += 1

              # Take extra samples
              for c in range(self.n_cls):
                  take = min(extra_per_class[c], len(cls_idx[c]))
                  if take > 0:
                      kept_idx[c] = np.concatenate((kept_idx[c], cls_idx[c][:take]))

              kept_idx = np.concatenate(kept_idx)

          # 5. Apply final subset, shuffle it, store results
          np.random.shuffle(kept_idx)

          self.trn_x = trn_x_original[kept_idx]
          self.trn_y = trn_y_original[kept_idx].reshape(-1, 1)

          # NEW: Re-distribute the limited trn_x and trn_y to clients
          self._redistribute_to_clients(self.trn_x, self.trn_y)

          # 6. Diagnostics after limiting
          if verbose:
              print(f"\n[limit_dataset] Final dataset size: {len(self.trn_y)}")
              final_counts = np.bincount(self.trn_y.reshape(-1), minlength=self.n_cls)
              print("[limit_dataset] Final per-class counts:")
              for c in range(self.n_cls):
                  print(f"  class {c}: {final_counts[c]}")
              print()


class DatasetSynthetic:
    def __init__(self, alpha, beta, iid_sol, iid_data, n_dim, n_clnt, n_cls, avg_data, data_path, name_prefix):
        self.dataset = 'synt'
        self.name  = name_prefix + '_'
        theta=0
        self.name += '%d_%d_%d_%d_%f_%f_%s_%s' %(n_dim, n_clnt, n_cls, avg_data,
                alpha, beta, iid_sol, iid_data)

        if (not os.path.exists('%sData/%s/' %(data_path, self.name))):
            # Generate data
            print('Sythetize')
            data_x, data_y = generate_syn_logistic(dimension=n_dim, n_clnt=n_clnt, n_cls=n_cls, avg_data=avg_data,
                                        alpha=alpha, beta=beta, theta=theta,
                                        iid_sol=iid_sol, iid_dat=iid_data)
            os.mkdir('%sData/%s/' %(data_path, self.name))
            os.mkdir('%sModel/%s/' %(data_path, self.name))
            np.save('%sData/%s/data_x.npy' %(data_path, self.name), data_x)
            np.save('%sData/%s/data_y.npy' %(data_path, self.name), data_y)
        else:
            # Load data
            print('Load')
            data_x = np.load('%sData/%s/data_x.npy' %(data_path, self.name),allow_pickle=True)
            data_y = np.load('%sData/%s/data_y.npy' %(data_path, self.name),allow_pickle=True)

        for clnt in range(n_clnt):
            print(', '.join(['%.4f' %np.mean(data_y[clnt]==t) for t in range(n_cls)]))

        self.clnt_x = data_x
        self.clnt_y = data_y

        self.tst_x = np.concatenate(self.clnt_x, axis=0)
        self.tst_y = np.concatenate(self.clnt_y, axis=0)
        self.n_client = len(data_x)
        print(self.clnt_x.shape)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name
        if self.name == 'mnist' or self.name == 'synt' or self.name == 'emnist' or self.name == 'fashion_mnist':
            # Handle data_x
            if isinstance(data_x, np.ndarray) and data_x.dtype == object:
                if data_x.ndim == 0:
                    self.X_data = torch.tensor(data_x.item()).float()
                else:
                    try:
                        # Attempt to convert to a standard numeric numpy array
                        # This assumes elements are numeric arrays or scalar values
                        temp_data_x = np.asarray(data_x.tolist())
                        # If it's still object dtype after tolist, it means elements themselves are not directly convertible.
                        # In that case, try to stack them assuming they are compatible arrays.
                        if temp_data_x.dtype == object:
                             temp_data_x = np.stack(data_x, axis=0) if data_x.size > 0 else np.array([])
                        self.X_data = torch.tensor(temp_data_x).float()
                    except Exception as e:
                        print(f"Warning: Failed to convert data_x (object dtype, ndim > 0) to tensor using asarray/stack. Error: {e}")
                        # Fallback: try direct conversion of the object array, though this is what originally failed.
                        self.X_data = torch.tensor(data_x).float()
            else:
                self.X_data = torch.tensor(data_x).float()

            # Handle data_y
            self.y_data = data_y
            if not isinstance(data_y, bool):
                if isinstance(data_y, np.ndarray) and data_y.dtype == object:
                    if data_y.ndim == 0:
                        self.y_data = torch.tensor(data_y.item()).long()
                    else:
                        try:
                            temp_data_y = np.asarray(data_y.tolist())
                            if temp_data_y.dtype == object:
                                temp_data_y = np.stack(data_y, axis=0) if data_y.size > 0 else np.array([])
                            self.y_data = torch.tensor(temp_data_y).long()
                        except Exception as e:
                            print(f"Warning: Failed to convert data_y (object dtype, ndim > 0) to tensor using asarray/stack. Error: {e}")
                            self.y_data = torch.tensor(data_y).long()
                else:
                    self.y_data = torch.tensor(data_y).long()

        elif self.name == 'CIFAR10' or self.name == 'CIFAR100':
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])

            self.X_data = data_x
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32') # Should be int for labels


    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if self.name == 'mnist' or self.name == 'synt' or self.name == 'emnist' or self.name == 'fashion_mnist':
            X = self.X_data[idx, :]
            if isinstance(self.y_data, bool):
                return X
            else:
                y = self.y_data[idx]
                return X, y

        elif self.name == 'CIFAR10' or self.name == 'CIFAR100':
            img = self.X_data[idx]
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img # Horizontal flip
                if (np.random.rand() > .5):
                # Random cropping
                    pad = 4
                    extended_img = np.zeros((3,32 + pad *2, 32 + pad *2)).astype(np.float32)
                    extended_img[:,pad:-pad,pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:,dim_1:dim_1+32,dim_2:dim_2+32]
            img = np.moveaxis(img, 0, -1)
            if isinstance(img, np.ndarray) and img.dtype == object:
                img = np.array(img, dtype=np.uint8)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y

