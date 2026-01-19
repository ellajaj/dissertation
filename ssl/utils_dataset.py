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
        if not os.path.exists('%sData/%s' %(self.data_path, self.name)):
            transform = transforms.Compose([transforms.ToTensor(),
                                            #transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ])

            trnset = torchvision.datasets.CIFAR10(root='%sData/Raw' %self.data_path,
                                                    train=True , download=True, transform=transform)
            tstset = torchvision.datasets.CIFAR10(root='%sData/Raw' %self.data_path,
                                                    train=False, download=True, transform=transform)

            trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=1)
            tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
            self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;

            trn_itr = trn_load.__iter__(); tst_itr = tst_load.__iter__()
            # labels are of shape (n_data,)
            trn_x, trn_y = trn_itr.__next__()
            tst_x, tst_y = tst_itr.__next__()

            trn_x = trn_x.numpy(); trn_y = trn_y.numpy().reshape(-1,1)
            tst_x = tst_x.numpy(); tst_y = tst_y.numpy().reshape(-1,1)

            # Shuffle Data
            np.random.seed(self.seed)
            rand_perm = np.random.permutation(len(trn_y))
            trn_x = trn_x[rand_perm]
            trn_y = trn_y[rand_perm]

            print("1")

            self.trn_x = trn_x
            self.trn_y = trn_y
            self.tst_x = tst_x
            self.tst_y = tst_y

            ###
            n_data_per_clnt = int((len(trn_y)) / self.n_client)
            # Draw from lognormal distribution
            clnt_data_list = (np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=self.unbalanced_sgm, size=self.n_client)).astype(int)
            clnt_data_list = (clnt_data_list/np.sum(clnt_data_list)*len(trn_y)).astype(int)
            diff = np.sum(clnt_data_list) - len(trn_y)

            print("2")

            # Add/Subtract the excess number starting from first client
            if diff!= 0:
                for clnt_i in range(self.n_client):
                    if clnt_data_list[clnt_i] > diff:
                        clnt_data_list[clnt_i] -= diff
                        break
            ###

            if self.rule == 'Drichlet':
                cls_priors   = np.random.dirichlet(alpha=[self.rule_arg]*self.n_cls,size=self.n_client)
                prior_cumsum = np.cumsum(cls_priors, axis=1)
                idx_list = [np.where(trn_y==i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]

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
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # Redraw class label if trn_y is out of that class
                        if cls_amount[cls_label] <= 0:
                            continue
                        cls_amount[cls_label] -= 1

                        clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = trn_x[idx_list[cls_label][cls_amount[cls_label]]]
                        clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = trn_y[idx_list[cls_label][cls_amount[cls_label]]]

                        break

                # Add explicit conversion to object dtype numpy array for inhomogeneous data
                clnt_x = np.array(clnt_x, dtype=object)
                clnt_y = np.array(clnt_y, dtype=object)

                cls_means = np.zeros((self.n_client, self.n_cls))
                for clnt in range(self.n_client):
                    for cls in range(self.n_cls):
                        cls_means[clnt,cls] = np.mean(clnt_y[clnt]==cls)
                prior_real_diff = np.abs(cls_means-cls_priors)
                print('--- Max deviation from prior: %.4f' %np.max(prior_real_diff))
                print('--- Min deviation from prior: %.4f' %np.min(prior_real_diff))


                #self.clnt_x = clnt_x; self.clnt_y = clnt_y
                #self.tst_x  = tst_x;  self.tst_y  = tst_y

            elif self.rule == 'iid':

                clnt_x = [ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ]

                clnt_data_list_cum_sum = np.concatenate(([0], np.cumsum(clnt_data_list)))
                for clnt_idx_ in range(self.n_client):
                    clnt_x[clnt_idx_] = trn_x[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]
                    clnt_y[clnt_idx_] = trn_y[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]


                clnt_x = np.array(clnt_x, dtype=object)
                #clnt_x = np.asarray(clnt_x)
                #clnt_y = np.asarray(clnt_y)
                clnt_y = np.array(clnt_y, dtype=object)


            self.clnt_x = clnt_x; self.clnt_y = clnt_y

            self.tst_x  = tst_x;  self.tst_y  = tst_y


            # Save data
            os.mkdir('%sData/%s' %(self.data_path, self.name))

            np.save('%sData/%s/clnt_x.npy' %(self.data_path, self.name), clnt_x)
            np.save('%sData/%s/clnt_y.npy' %(self.data_path, self.name), clnt_y)

            np.save('%sData/%s/tst_x.npy'  %(self.data_path, self.name),  tst_x)
            np.save('%sData/%s/tst_y.npy'  %(self.data_path, self.name),  tst_y)
            print("split complete")

        else:
            print("Data is already downloaded")
            self.clnt_x = np.load('%sData/%s/clnt_x.npy' %(self.data_path, self.name),allow_pickle=True)
            self.clnt_y = np.load('%sData/%s/clnt_y.npy' %(self.data_path, self.name),allow_pickle=True)
            self.n_client = len(self.clnt_x)

            self.tst_x  = np.load('%sData/%s/tst_x.npy'  %(self.data_path, self.name),allow_pickle=True)
            self.tst_y  = np.load('%sData/%s/tst_y.npy'  %(self.data_path, self.name),allow_pickle=True)

            self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;

        count = 0
        for clnt in range(self.n_client):
            count += self.clnt_y[clnt].shape[0]
    

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_x, data_y=True, train=False):
        self.train = train
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.X_data = data_x
        self.y_data = data_y
        if not isinstance(data_y, bool):
            self.y_data = data_y.astype('int64') # Should be int for labels


    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):

        img = self.X_data[idx]

        if isinstance(img, np.ndarray):
            img = img.astype(np.float32)

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
        img = self.transform(img)
        if isinstance(self.y_data, bool):
            return img
        else:
            y = self.y_data[idx]
            return img, y


class TripleGANDataset(torch.utils.data.Dataset):
    def __init__(self, x_l, y_l, x_u, train=True):
        self.x_l, self.y_l = x_l, y_l
        self.x_u = x_u
        self.train = train
        self.l_perm = np.random.permutation(len(self.y_l))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.x_u) # Unlabeled set is usually larger

    def __getitem__(self, idx):
        # Unlabeled sample
        img_u = self.x_u[idx]
        
        # Labeled sample (cycling through the smaller labeled set)
        #l_idx = idx % len(self.y_l)
        l_idx = self.l_perm[idx % len(self.y_l)]

        img_l = self.x_l[l_idx]
        label_l = self.y_l[l_idx]
        img_l = np.array(img_l).astype(np.float32)
        img_u = np.array(img_u).astype(np.float32)
 
        #img_l = np.array(img_l, dtype=np.uint8)
        #img_u = np.array(img_u, dtype=np.uint8)
        '''if img_l.shape[0] == 3: 
            img_l = np.moveaxis(img_l, 0, -1) # Converts (3, 32, 32) -> (32, 32, 3)
        if img_u.shape[0] == 3:
            img_u = np.moveaxis(img_u, 0, -1)

        #if img_l.size == 3072: # 32*32*3
            #img_l = img_l.reshape(32, 32, 3)
            #img_u = img_u.reshape(32, 32, 3)

        if self.transform is not None:
            img_l = self.transform(img_l)
            img_u = self.transform(img_u)'''

        label_l = np.array(label_l).astype(np.int64)
        #label_l = torch.tensor(label_l, dtype=torch.long)

        #label_l = torch.tensor(label_l).long().squeeze()
        #label_l = torch.from_numpy(np.array(label_l).astype(np.int64)).squeeze()
        return img_l, torch.from_numpy(label_l), img_u

