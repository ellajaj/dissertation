from utils_libs import *
from PIL import Image

class DatasetObject:
    def __init__(self, dataset, n_client, seed, rule, unbalanced_sgm=0, rule_arg='', data_path='Folder/'):
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

    def set_data(self):
        rng = np.random.default_rng(seed=42)
        labeled_fraction = 0.10
        unlabeled_to_labeled_ratio = 1.0

        def _match_unlabeled_to_labeled(clnt_x_l, clnt_y_l, clnt_x_u, rng):
            # Enforce per-client unlabeled count to match labeled count exactly.
            matched_x_u = []
            for clnt in range(len(clnt_x_l)):
                x_l_i = np.asarray(clnt_x_l[clnt])
                y_l_i = np.asarray(clnt_y_l[clnt])
                x_u_i = np.asarray(clnt_x_u[clnt])

                n_l = len(y_l_i)
                if n_l == 0:
                    matched_x_u.append(x_u_i[:0].astype(np.float32))
                    continue

                if len(x_u_i) >= n_l:
                    sel = rng.choice(len(x_u_i), size=n_l, replace=False)
                else:
                    sel = rng.choice(len(x_u_i), size=n_l, replace=True)
                matched_x_u.append(x_u_i[sel].astype(np.float32))

            return np.array(matched_x_u, dtype=object)

        if not os.path.exists('%sData/%s' %(self.data_path, self.name)):
            if self.dataset == 'CIFAR10':
                transform = transforms.Compose([transforms.ToTensor(),
                                                #transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                                                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ])

                trnset = torchvision.datasets.CIFAR10(root='%sData/Raw' %self.data_path,
                                                        train=True , download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR10(root='%sData/Raw' %self.data_path,
                                                        train=False, download=True, transform=transform)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
            if self.dataset == 'fashion_mnist':
                # Keep raw pixel scale here; normalization is applied later in Dataset/TripleGANDataset.
                transform = transforms.Compose([transforms.ToTensor()])
                trnset = torchvision.datasets.FashionMNIST(root='%sData/Raw' %self.data_path,
                                                           train=True, download=True, transform=transform)
                tstset = torchvision.datasets.FashionMNIST(root='%sData/Raw' %self.data_path,
                                                           train=False, download=True, transform=transform)
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=60000, shuffle=False, num_workers =1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;

            trn_itr = trn_load.__iter__(); tst_itr = tst_load.__iter__()
            # labels are of shape (n_data,)
            trn_x, trn_y = trn_itr.__next__()
            tst_x, tst_y = tst_itr.__next__()

            trn_x = trn_x.numpy(); trn_y = trn_y.numpy().reshape(-1,1)
            tst_x = tst_x.numpy(); tst_y = tst_y.numpy().reshape(-1,1)

            trn_x = (trn_x * 255).astype(np.uint8)
            tst_x = (tst_x * 255).astype(np.uint8)

            # Shuffle Data
            np.random.seed(self.seed)
            rand_perm = np.random.permutation(len(trn_y))
            trn_x = trn_x[rand_perm]
            trn_y = trn_y[rand_perm]

            # Use 10% labeled and match unlabeled pool size to labeled pool size.
            n_total = len(trn_y)
            n_labeled = int(labeled_fraction * n_total)
            n_unlabeled = int(unlabeled_to_labeled_ratio * n_labeled)

            X_labeled = trn_x[:n_labeled]
            y_labeled = trn_y[:n_labeled]

            X_unlabeled = trn_x[n_labeled:n_labeled + n_unlabeled]
            y_unlabeled = trn_y[n_labeled:n_labeled + n_unlabeled]

            #determine size for labelled clients 
            n_labeled_per_client = len(y_labeled) // self.n_client

            clnt_data_list = rng.lognormal(
                mean=np.log(n_labeled_per_client),
                sigma=self.unbalanced_sgm,
                size=self.n_client
            ).astype(int)

            clnt_data_list = (
                clnt_data_list / clnt_data_list.sum() * len(y_labeled)
            ).astype(int)

            diff = clnt_data_list.sum() - len(y_labeled)
            if diff != 0:
                for i in range(self.n_client):
                    if clnt_data_list[i] > abs(diff):
                        clnt_data_list[i] -= diff
                        break
            print(
                f"[Split Setup] total_train={n_total}, labeled={len(y_labeled)}, "
                f"unlabeled_pool={len(y_unlabeled)}, clients={self.n_client}"
            )
            print(
                f"[Split Setup] target_labeled_per_client(min/mean/max)="
                f"{clnt_data_list.min()}/{clnt_data_list.mean():.2f}/{clnt_data_list.max()}"
            )

            self.trn_x = trn_x
            self.trn_y = trn_y
            self.tst_x = tst_x
            self.tst_y = tst_y

            clnt_data_list_base = clnt_data_list.copy()

            if self.rule == 'Drichlet':
                #split for labelled data
                cls_priors   = np.random.dirichlet(alpha=[self.rule_arg]*self.n_cls,size=self.n_client)
                prior_cumsum = np.cumsum(cls_priors, axis=1)
                idx_list = [np.where(y_labeled==i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx) for idx in idx_list]

                clnt_x = [ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ]

                while(np.sum(clnt_data_list)!=0):
                    curr_clnt = np.random.randint(self.n_client)
                    # If current node is full resample a client
                    print('Remaining Data: %d' %np.sum(clnt_data_list))
                    if clnt_data_list[curr_clnt] <= 0:
                        continue
                    clnt_data_list[curr_clnt] -= 1
                    curr_prior = prior_cumsum[curr_clnt]
                    while True:
                        print("cls_amount:", cls_amount)

                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # Redraw class label if trn_y is out of that class
                        if cls_amount[cls_label] <= 0:
                            continue
                        cls_amount[cls_label] -= 1
                        idx = idx_list[cls_label][cls_amount[cls_label]]

                        clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = X_labeled[idx]
                        clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = y_labeled[idx]
                        break

                #split for unlabelled data
                idx_list_u = [np.where(y_unlabeled==i)[0] for i in range(self.n_cls)]
                cls_amount_u = [len(idx) for idx in idx_list_u]

                clnt_x_u = [np.zeros((clnt_data_list_base[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client)]
                clnt_data_list_u = clnt_data_list_base.copy()

                while(np.sum(clnt_data_list_u)!=0):
                    curr_clnt = np.random.randint(self.n_client)
                    # If current node is full resample a client
                    print('Remaining Data: %d' %np.sum(clnt_data_list))
                    if clnt_data_list_u[curr_clnt] <= 0:
                        continue
                    curr_prior = prior_cumsum[curr_clnt]
                    while True:
                        print("cls_amount:", cls_amount_u)

                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # Redraw class label if trn_y is out of that class
                        if cls_amount_u[cls_label] <= 0:
                            continue
                        cls_amount_u[cls_label] -= 1
                        idx = idx_list_u[cls_label][cls_amount_u[cls_label]]

                        insert_idx = clnt_data_list_u[curr_clnt] - 1
                        clnt_x_u[curr_clnt][insert_idx] = X_unlabeled[idx]

                        clnt_data_list_u[curr_clnt] -= 1
                        break

                # Add explicit conversion to object dtype numpy array for inhomogeneous data
                clnt_x_l = np.array(clnt_x, dtype=object)
                clnt_y_l = np.array(clnt_y, dtype=object)
                clnt_x_u = np.array(clnt_x_u, dtype=object)
                clnt_x_u = _match_unlabeled_to_labeled(clnt_x_l, clnt_y_l, clnt_x_u, rng)

                cls_means = np.zeros((self.n_client, self.n_cls))
                for clnt in range(self.n_client):
                    for cls in range(self.n_cls):
                        cls_means[clnt,cls] = np.mean(clnt_y[clnt]==cls)
                prior_real_diff = np.abs(cls_means-cls_priors)
                print('--- Max deviation from prior: %.4f' %np.max(prior_real_diff))
                print('--- Min deviation from prior: %.4f' %np.min(prior_real_diff))


            elif self.rule == 'iid':

                clnt_x = [ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ]

                clnt_data_list_cum_sum = np.concatenate(([0], np.cumsum(clnt_data_list)))
                for clnt_idx_ in range(self.n_client):
                    clnt_x[clnt_idx_] = trn_x[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]
                    clnt_y[clnt_idx_] = trn_y[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]


                clnt_x = np.array(clnt_x, dtype=object)
                clnt_y = np.array(clnt_y, dtype=object)

                clnt_x_u = [np.zeros((clnt_data_list_base[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client)]
                clnt_data_list_u = clnt_data_list_base.copy()
                u_start = 0
                for clnt_idx_ in range(self.n_client):
                    u_end = u_start + clnt_data_list_u[clnt_idx_]
                    clnt_x_u[clnt_idx_] = X_unlabeled[u_start:u_end]
                    u_start = u_end
                clnt_x_u = np.array(clnt_x_u, dtype=object)
                clnt_x_u = _match_unlabeled_to_labeled(clnt_x, clnt_y, clnt_x_u, rng)


            self.clnt_x_l = clnt_x
            self.clnt_y_l = clnt_y
            self.clnt_x_u = clnt_x_u
            self.tst_x  = tst_x
            self.tst_y  = tst_y
            print(f"Split summary: labeled={len(y_labeled)}, unlabeled={len(y_unlabeled)}, ratio={len(y_unlabeled)/max(1, len(y_labeled)):.2f}")

            labeled_counts = np.array([len(np.asarray(self.clnt_y_l[c])) for c in range(self.n_client)], dtype=int)
            unlabeled_counts = np.array([len(np.asarray(self.clnt_x_u[c])) for c in range(self.n_client)], dtype=int)
            print(
                f"[Split Check] labeled_total={labeled_counts.sum()} (expected={len(y_labeled)}), "
                f"unlabeled_total={unlabeled_counts.sum()} (after matching to labeled)"
            )
            print(
                f"[Split Check] per-client labeled min/mean/max="
                f"{labeled_counts.min()}/{labeled_counts.mean():.2f}/{labeled_counts.max()} | "
                f"unlabeled min/mean/max={unlabeled_counts.min()}/{unlabeled_counts.mean():.2f}/{unlabeled_counts.max()}"
            )
            for clnt in range(self.n_client):
                l_cnt = labeled_counts[clnt]
                u_cnt = unlabeled_counts[clnt]
                lbl = np.asarray(self.clnt_y_l[clnt]).reshape(-1)
                cls_hist = np.bincount(lbl, minlength=self.n_cls).tolist() if l_cnt > 0 else [0] * self.n_cls
                print(
                    f"[Split Check][Client {clnt:02d}] labeled={l_cnt}, unlabeled={u_cnt}, "
                    f"match_ok={u_cnt == l_cnt}, class_hist={cls_hist}"
                )


            # Save data
            os.mkdir('%sData/%s' %(self.data_path, self.name))

            np.save('%sData/%s/clnt_x_l.npy' %(self.data_path, self.name), clnt_x_l)
            np.save('%sData/%s/clnt_y_l.npy' %(self.data_path, self.name), clnt_y_l)
            np.save('%sData/%s/clnt_x_u.npy' %(self.data_path, self.name), clnt_x_u)
            np.save('%sData/%s/tst_x.npy'  %(self.data_path, self.name),  tst_x)
            np.save('%sData/%s/tst_y.npy'  %(self.data_path, self.name),  tst_y)
            print("split complete")

        else:
            print("Data is already downloaded")
            self.clnt_x_l = np.load('%sData/%s/clnt_x_l.npy' %(self.data_path, self.name),allow_pickle=True)
            self.clnt_y_l = np.load('%sData/%s/clnt_y_l.npy' %(self.data_path, self.name),allow_pickle=True)
            self.clnt_x_u = np.load('%sData/%s/clnt_x_u.npy' %(self.data_path, self.name),allow_pickle=True)
            self.n_client = len(self.clnt_x_u)

            self.tst_x  = np.load('%sData/%s/tst_x.npy'  %(self.data_path, self.name),allow_pickle=True)
            self.tst_y  = np.load('%sData/%s/tst_y.npy'  %(self.data_path, self.name),allow_pickle=True)

            if(self.dataset == "CIFAR10"):
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
                
            elif(self.dataset == "fashion_mnist"):
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;

            self.clnt_x_u = _match_unlabeled_to_labeled(self.clnt_x_l, self.clnt_y_l, self.clnt_x_u, rng)

        count = 0
        for clnt in range(self.n_client):
            count += self.clnt_y_l[clnt].shape[0]
        unl_count = sum(len(np.asarray(self.clnt_x_u[clnt])) for clnt in range(self.n_client))
        print(f"Loaded client totals: labeled={count}, unlabeled={unl_count}, ratio={unl_count / max(1, count):.2f}")
        loaded_l_counts = np.array([len(np.asarray(self.clnt_y_l[c])) for c in range(self.n_client)], dtype=int)
        loaded_u_counts = np.array([len(np.asarray(self.clnt_x_u[c])) for c in range(self.n_client)], dtype=int)
        print(
            f"[Loaded Split Check] clients={self.n_client}, "
            f"labeled(min/mean/max)={loaded_l_counts.min()}/{loaded_l_counts.mean():.2f}/{loaded_l_counts.max()}, "
            f"unlabeled(min/mean/max)={loaded_u_counts.min()}/{loaded_u_counts.mean():.2f}/{loaded_u_counts.max()}"
        )
        print(f"[Loaded Split Check] all_clients_unlabeled_match_labeled={np.all(loaded_l_counts == loaded_u_counts)}")
    

class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name, data_x, data_y=True, train=False):
        self.name = str(dataset_name)
        self.X_data = data_x
        self.y_data = data_y
        if self.name == "CIFAR10":
            self.train = train
            cifar_mean = (0.4914, 0.4822, 0.4465)
            cifar_std = (0.2023, 0.1994, 0.2010)
            if self.train:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar_mean, cifar_std),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(cifar_mean, cifar_std),
                ])

            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('int64') # Should be int for labels
        elif self.name == "fashion_mnist":
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            # Handle data_x
            if isinstance(data_x, np.ndarray) and data_x.dtype == object:
                if data_x.ndim == 0:
                    self.X_data = torch.tensor(data_x.item()).float()
                else:
                    try:
                        temp_data_x = np.asarray(data_x.tolist())
                        if temp_data_x.dtype == object:
                             temp_data_x = np.stack(data_x, axis=0) if data_x.size > 0 else np.array([])
                        self.X_data = torch.tensor(temp_data_x).float()
                    except Exception as e:
                        print(f"Warning: Failed to convert data_x (object dtype, ndim > 0) to tensor using asarray/stack. Error: {e}")
                        # Fallback: try direct conversion of the object array, though this is what originally failed.
                        self.X_data = torch.tensor(data_x).float()
            else:
                self.X_data = torch.tensor(data_x).float()
        else:
            raise ValueError(f"Unknown dataset name: {self.name}")


    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if self.name == "CIFAR10":
            img = self.X_data[idx]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            img = np.asarray(img)

            # Convert CHW -> HWC for torchvision image transforms.
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                if np.max(img) <= 1.0:
                    img = (img * 255.0).clip(0, 255)
                img = img.astype(np.uint8)

            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y
        elif self.name == "fashion_mnist":
            X = self.X_data[idx]
            if isinstance(X, torch.Tensor):
                X = X.cpu().numpy()
            X = np.asarray(X)
            if X.ndim == 3 and X.shape[0] == 1:
                X = np.squeeze(X, axis=0)
            if X.dtype != np.uint8:
                if np.max(X) <= 1.0:
                    X = (X * 255.0).clip(0, 255)
                X = X.astype(np.uint8)
            X = self.transform(X)
            if isinstance(self.y_data, bool):
                return X
            else:
                y = self.y_data[idx]
                return X, y
        else:
            raise ValueError(f"Unknown dataset name: {self.name}")


class TripleGANDataset(torch.utils.data.Dataset):
    def __init__(self, x_l, y_l, x_u, dataset_name):
        self.name = str(dataset_name)
        self.x_l, self.y_l = x_l, y_l
        self.x_u = x_u
        self.l_perm = np.random.permutation(len(self.y_l))
        self.u_perm = np.random.permutation(len(self.x_u))
        if self.name == "fashion_mnist":
            self.transform_l = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ])
            self.transform_u = self.transform_l
        elif self.name == "CIFAR10":
            cifar_mean = (0.4914, 0.4822, 0.4465)
            cifar_std = (0.2023, 0.1994, 0.2010)
            weak_aug = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            self.transform_l = transforms.Compose(
                [transforms.ToPILImage()] + weak_aug + [transforms.ToTensor(), transforms.Normalize(cifar_mean, cifar_std)]
            )
            self.transform_u = transforms.Compose(
                [transforms.ToPILImage()] + weak_aug + [transforms.ToTensor(), transforms.Normalize(cifar_mean, cifar_std)]
            )

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar_mean, cifar_std)
            ])

    def __len__(self):
        return min(len(self.x_u), len(self.y_l))

    def __getitem__(self, idx):
        # Unlabeled sample
        u_idx = self.u_perm[idx % len(self.x_u)]
        img_u = self.x_u[u_idx]
        
        # Labeled sample (cycling through the smaller labeled set)
        #l_idx = idx % len(self.y_l)
        l_idx = self.l_perm[idx % len(self.y_l)]
        img_l = self.x_l[l_idx]
        label_l = self.y_l[l_idx]

        img_l = self._fix_shape(self.x_l[l_idx])
        img_u = self._fix_shape(self.x_u[u_idx])

        # Apply transforms
        img_l = self.transform_l(img_l)
        img_u = self.transform_u(img_u)

        label_l = np.array(label_l).astype(np.int64)

        
        return img_l, torch.from_numpy(label_l), img_u

    def _fix_shape(self, img):
        img = np.asarray(img)
        # If (1, 28, 28) -> squeeze to (28, 28) for ToTensor
        if img.ndim == 3 and img.shape[0] == 1:
            img = np.squeeze(img, axis=0)
        # If (3, 32, 32) -> transpose to (32, 32, 3) for ToTensor
        elif img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        # Keep uint8 so ToTensor scales to [0,1] correctly before normalization.
        if img.dtype != np.uint8:
            if np.max(img) <= 1.0:
                img = (img * 255.0).clip(0, 255)
            img = img.astype(np.uint8)
        return img

