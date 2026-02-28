from utils_libs import *
import torch.nn.utils.spectral_norm as spectral_norm
import torch
import torch.nn as nnq3
import numpy as np
import math
from torch.nn import init
from torch.nn import utils

'''class Classifier(nn.Module):
    def __init__(self, dataset_name, num_classes=10):
        super(Classifier, self).__init__()
        self.name = dataset_name
        if self.name=='CIFAR10':
            self.channels=3
        elif self.name=='fashion_mnist':
            self.channels=1
            ''' '''self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, self.n_cls)''' '''
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(self.channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        replace_bn_with_gn(self.model, num_groups=8)  
        num_ftrs = self.model.fc.in_features # 512
        self.model.fc = nn.Sequential(
            #nn.Dropout(0.1), 
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x, return_features=False):
        #if self.name == 'CIFAR10':
        x = self.model.conv1(x)
        x = self.model.bn1(x) # or gn1
        x = self.model.relu(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        feat = torch.flatten(x, 1)
        logits = self.model.fc(feat)
        if return_features:
            return logits, feat
        return logits

        elif self.name == 'fashion_mnist':
            x = x.reshape(x.size(0), -1)
            x = F.relu(self.fc1(x))
            feat = F.relu(self.fc2(x))   
            logits = self.fc3(feat)      
            if return_features:
                return logits, feat
            return logits'''

class Classifier(nn.Module):
    def __init__(self, dataset_name, num_classes=10):
        super(Classifier, self).__init__()
        self.name = dataset_name
        if self.name == 'fashion_mnist':
            in_channels = 1
        else:
            in_channels = 3

        resnet18 = models.resnet18()
        resnet18.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        resnet18.maxpool = nn.Identity()
        resnet18.fc = nn.Linear(512, num_classes)

        # Change BN to GN 
        resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

        resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

        resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
        resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

        resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
        resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

        resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
        resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        assert len(dict(resnet18.named_parameters()).keys()) == len(resnet18.state_dict().keys()), 'More BN layers are there...'
        
        self.model = resnet18

    def forward(self, x, return_features=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x) # or gn1
        x = self.model.relu(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        feat = torch.flatten(x, 1)
        logits = self.model.fc(feat)
        if return_features:
            return logits, feat
        return logits


class Generator(nn.Module):
    def __init__(self, dataset_name, z_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.name = dataset_name
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.nlabels = num_classes
        if self.name == 'CIFAR10':
            self.channels = 3
        elif self.name == 'fashion_mnist':
            self.channels = 1

        #if self.name == 'CIFAR10':
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim + num_classes, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, self.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, y=None):
        if z.dim() == 4:
            z = z.view(z.size(0), -1)
        if y is None:
            y = torch.randint(0, self.num_classes, (z.size(0),), device=z.device)
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        z_combined = torch.cat([z, y_onehot], dim=1)

        #if self.name == 'fashion_mnist':
            #h = self.fc_fm(z_combined).view(z_combined.size(0), 256, 7, 7)
            #return self.main_fm(h)

        z_combined = z_combined.view(z_combined.size(0), z_combined.size(1), 1, 1)
        return self.main(z_combined)

    
    '''def __init__(self, dataset_name, z_dim=100, n_label=10, im_size=32, im_chan=3, embed_size=256,
        nfilter=64, nfilter_max=512, actvn=nn.ReLU(),):
        super().__init__()
        self.actvn = actvn
        self.name = dataset_name
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        self.z_dim = z_dim
        self.nlabels = n_label

        if self.name == 'CIFAR10':
            self.im_chan = 3
            self.im_size = 32
        elif self.name == 'fashion_mnist':
            self.im_chan = 1
            self.im_size = 28

        # Submodules
        nlayers = int(np.log2(self.im_size / s0))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)
        self.embedding = nn.Embedding(n_label, embed_size)
        self.fc = nn.Linear(z_dim + embed_size, self.nf0 * s0 * s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2 ** (nlayers - i), nf_max)
            nf1 = min(nf * 2 ** (nlayers - i - 1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1, actvn=self.actvn, use_sn=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ]

        blocks += [ResnetBlock(nf, nf, actvn=self.actvn, use_sn=True)]

        self.resnet = nn.Sequential(*blocks)
        #self.bn_final = nn.BatchNorm2d(nfilter)
        self.conv_img = nn.Conv2d(nf, self.im_chan, 3, padding=1)

    def forward(self, z, y=None):
        if y is not None:
            assert (z.size(0) == y.size(0))
        batch_size = z.size(0)
        if y is None:
            y = torch.zeros(batch_size).type(torch.int64)
            if z.is_cuda:
                y = y.cuda()

        if y.dtype is torch.int64:
            #y = y.clamp_(None, self.nlabels - 1)
            #y = y.clamp(max=self.nlabels - 1)
            assert y.max() < self.nlabels and y.min() >= 0 #error if lables are wrong 
            yembed = self.embedding(y)
        else:
            yembed = y

        #yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)

        yz = torch.cat([z, yembed], dim=1)
        out = self.fc(yz)
        out = out.view(batch_size, self.nf0, self.s0, self.s0)
        out = self.resnet(out)
        out = self.conv_img(actvn(out))
        out = torch.tanh(out)

        if out.size(-1) == 32:
            out = out[:, :, 2:30, 2:30]

        return out'''

'''class Generator(nn.Module):
    """Generator generates 128x128. resnet generator"""

    def __init__(
        self,
        z_dim=256,
        n_label=10,
        im_size=32,
        im_chan=3,
        embed_size=256,
        nfilter=64,
        nfilter_max=512,
        actvn=F.relu,
        distribution="normal",
        bottom_width=4,
    ):
        super(Generator, self).__init__()
        self.num_features = num_features = nfilter
        self.z_dim = z_dim
        self.bottom_width = bottom_width
        self.activation = activation = actvn
        self.num_classes = num_classes = n_label
        self.distribution = distribution

        width_coe = 8
        self.l1 = nn.Linear(self.z_dim, width_coe * num_features * bottom_width ** 2)

        self.block2 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block3 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block4 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.b7 = nn.BatchNorm2d(num_features * width_coe)
        self.conv7 = nn.Conv2d(num_features * width_coe, 3, 1, 1)

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.tensor)
        init.xavier_uniform_(self.conv7.weight.tensor)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in [2, 3, 4]:
            h = getattr(self, "block{}".format(i))(h, y, **kwargs)
        h = self.activation(self.b7(h))
        return torch.tanh(self.conv7(h))'''

'''class Discriminator(nn.Module):
    def __init__(self, n_label=10):
        super().__init__()
        self.mb_std = MinibatchStdDev()
        #comment out following block
        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),   # 16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), # 8x8
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),# 4x4
            nn.LeakyReLU(0.2, inplace=True),
        )
        ''''''self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(4, 64, 4, 2, 1)),   
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), 
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )''''''

        self.feature_dim = 256 * 4 * 4 # 4096
        #self.fc = nn.Linear(self.feature_dim, 1)
        self.fc = spectral_norm(nn.Linear(self.feature_dim, 1))
        #self.embed = nn.Embedding(n_label, self.feature_dim)
        self.embed = spectral_norm(nn.Embedding(n_label, self.feature_dim))

    def forward(self, x, y=None, return_features=False):
        x = self.mb_std(x)
        h = self.conv(x)
        h = h.view(h.size(0), -1) # Shape: [Batch, 4096]

        # 1. Image Score (Unconditional)
        out = self.fc(h) # Shape: [Batch, 1]

        if y is None:
            return out.view(-1)

        # 2. Label Alignment (Conditional)
        # Ensure y is the right shape and on the right device
        target_y = y.long()
        emb = self.embed(target_y) # Shape: [Batch, 4096]

        # Use an explicit dot product for each item in the batch
        # (h * emb) results in [Batch, 4096]. Summing dim 1 results in [Batch]
        proj = torch.sum(h * emb, dim=1, keepdim=True) # Shape: [Batch, 1]

        # 3. Combine
        # Both are [Batch, 1], so they will add perfectly
        total_score = out + proj
        
        return total_score.view(-1) # Final Shape: [Batch]

        #batch_size = x.size(0)

        out = self.actvn(self.conv_img(x))
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        #out = self.fc(self.actvn(out))
        final_score = self.fc(actvn(out_flat))

        if y is None:
            return out

        index = torch.LongTensor(range(out.size(0)))
        if x.is_cuda:
            index = index.cuda()
            y = y.cuda()

        out = out[index, y]
        #return out
        if return_features:
            return out, final_score
        return final_score'''
        
class Discriminator(nn.Module):
    def __init__(
        self,
        dataset_name,
        z_dim=256,
        n_label=10,
        embed_size=256,
        nfilter=64,
        nfilter_max=512,
        actvn=nn.ReLU(),
    ):
        super().__init__()
        self.name = dataset_name
        self.actvn = actvn
        self.embed_size = embed_size
        self.nlabels = n_label
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        if self.name =='fashion_mnist':
            self.im_size = 28
            self.im_chan = 1
            self.feat_dim = 200
        elif self.name =='CIFAR10':
            self.im_size = 32
            self.im_chan = 3
            self.feat_dim = 512
        self.feat_dim = 512

        # Submodules
        nlayers = int(np.log2(self.im_size / s0))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        blocks = [ResnetBlock(nf, nf, actvn=self.actvn)]

        for i in range(nlayers):
            nf0 = min(nf * 2 ** i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1, actvn=self.actvn),
            ]

        self.conv_img = nn.Conv2d(self.im_chan, 1 * nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        #self.fc = nn.Linear(self.nf0 * s0 * s0, n_label)
        #self.fc = None
        #self.d_internal_dim = self.nf0 * s0 * s0 
        with torch.no_grad():
            dummy = torch.zeros(1, self.im_chan, self.im_size, self.im_size)
            h = self.conv_img(dummy)
            h = self.resnet(h)
            h = F.adaptive_avg_pool2d(h, 1)
            self.d_internal_dim = h.view(1, -1).size(1)
        # New Projection Head
        self.fc = nn.Linear(self.d_internal_dim + self.feat_dim, n_label)

    def forward(self, x, y, feat):
        batch_size = x.size(0)

        out = self.conv_img(x)
        out = self.resnet(out)
        #out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        #out = out.view(out.size(0), -1)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        combined_feat = torch.cat([out, feat], dim=1)
        logits = self.fc(self.actvn(combined_feat))

        # 4. Return the scalar score for the specific class y
        # This is the "conditional" part of the Discriminator
        batch_range = torch.arange(logits.size(0), device=x.device)
        return logits[batch_range, y]

        '''if (self.fc is None) or (self.fc.in_features != out.size(1)):
            self.fc = nn.Linear(out.size(1), self.nlabels).to(out.device)

        out = self.fc(self.actvn(out))

        if y is None:
            return out

        index = torch.LongTensor(range(out.size(0)))
        if x.is_cuda:
            index = index.cuda()
            y = y.cuda()
        out = out[index, y]
        return out'''



'''class Discriminator(nn.Module):
    #snresentprojection discriminator
    def __init__(
        self,
        z_dim=256,
        n_label=10,
        im_size=32,
        im_chan=3,
        embed_size=256,
        nfilter=64,
        nfilter_max=512,
        actvn=nn.ReLU(),
    ):
        super(Discriminator, self).__init__()
        self.num_features = num_features = nfilter
        self.num_classes = num_classes = n_label
        self.activation = activation = actvn

        width_coe = 8
        self.block1 = OptimizedBlock(3, num_features * width_coe)
        self.block2 = BlockD(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            downsample=True,
        )
        self.block3 = BlockD(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            downsample=True,
        )
        self.block4 = BlockD(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            downsample=True,
        )
        self.l7 = utils.spectral_norm(nn.Linear(num_features * width_coe, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Embedding(num_classes, num_features * width_coe)
            )

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l7.weight.data)
        optional_l_y = getattr(self, "l_y", None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        if y is not None:
            assert y.dim() == 1, f"Projection D expects y shape [B], got {y.shape}"
        bs = x.shape[0]
        h = x
        for i in range(1, 5):
            h = getattr(self, "block{}".format(i))(h)
        h = self.activation(h)
        # Global pooling
        #h = torch.sum(h, dim=(2, 3))
        h = h.view(h.size(0), h.size(1), -1)
        h = h.sum(dim=2)

        output = self.l7(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        else:
            output_list = []
            for i in range(self.num_classes):
                ty = torch.ones([bs,], dtype=torch.long) * i
                toutput = output + torch.sum(
                    self.l_y(ty.to(x.device)) * h, dim=1, keepdim=True
                )
                output_list.append(toutput)
            output = torch.cat(output_list, dim=1)
        return output'''

def actvn(x):
    out = torch.relu(x)
    return out

class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, actvn, fhidden=None, is_bias=True, use_bn=False, use_sn=False):
        super().__init__()
        # Attributes
        self.actvn = actvn
        self.is_bias = is_bias
        self.learned_shortcut = fin != fout
        self.fin = fin
        self.fout = fout
        self.use_bn = use_bn
        self.use_sn = use_sn

        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1, bias=is_bias)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(
                self.fin, self.fout, 1, stride=1, padding=0, bias=False
            )

        if use_sn:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)   

        if use_bn:
            self.bn0 = nn.BatchNorm2d(self.fin)
            self.bn1 = nn.BatchNorm2d(self.fhidden)

    def forward(self, x):
        x_s = self._shortcut(x)
        if self.use_bn:
            dx = self.conv_0(self.actvn(self.bn0(x)))
            dx = self.conv_1(self.actvn(self.bn1(dx)))
        else:
            dx = self.conv_0(self.actvn(x))
            dx = self.conv_1(self.actvn(dx))

        out = x_s + dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

def replace_bn_with_gn(module, num_groups=8):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # Calculate appropriate groups. 
            # If channels < num_groups, use channels/2 or 1 group.
            ng = num_groups if child.num_features % num_groups == 0 else 1
            
            gn = nn.GroupNorm(ng, child.num_features)
            # Important: Copy weights (gamma/beta) to keep initialization rough alignment
            if child.affine:
                gn.weight.data.copy_(child.weight.data)
                gn.bias.data.copy_(child.bias.data)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups)

class OptimizedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu):
        super(OptimizedBlock, self).__init__()
        self.activation = activation

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(out_ch, out_ch, ksize, 1, pad))
        self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x, 2))

    def residual(self, x):
        h = self.activation(self.c1(x))
        return F.avg_pool2d(self.c2(h), 2)

class BlockD(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        h_ch=None,
        ksize=3,
        pad=1,
        activation=F.relu,
        downsample=False,
    ):
        super(BlockD, self).__init__()

        self.activation = activation
        self.downsample = downsample

        self.learnable_sc = (in_ch != out_ch) or downsample
        if h_ch is None:
            h_ch = in_ch
        else:
            h_ch = out_ch

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, h_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(h_ch, out_ch, ksize, 1, pad))
        if self.learnable_sc:
            self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        h = self.c1(self.activation(x))
        h = self.c2(self.activation(h))
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h

class ConditionalBatchNorm2d(nn.BatchNorm2d):

    """Conditional Batch Normalization"""

    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=False,
        track_running_stats=True,
    ):
        super(ConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * output + bias


class CategoricalConditionalBatchNorm2d(ConditionalBatchNorm2d):
    def __init__(
        self,
        num_classes,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=True,
    ):
        super(CategoricalConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)

        return super(CategoricalConditionalBatchNorm2d, self).forward(
            input, weight, bias
        )


def _upsample(x):
    h, w = x.size()[2:]
    return F.interpolate(x, size=(h * 2, w * 2), mode="bilinear", align_corners=False)


class Block(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        h_ch=None,
        ksize=3,
        pad=1,
        activation=F.relu,
        upsample=False,
        num_classes=0,
    ):
        super(Block, self).__init__()

        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch or upsample
        if h_ch is None:
            h_ch = out_ch
        self.num_classes = num_classes

        # Register layrs
        self.c1 = nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
        self.c2 = nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
        if self.num_classes > 0:
            self.b1 = CategoricalConditionalBatchNorm2d(num_classes, in_ch)
            self.b2 = CategoricalConditionalBatchNorm2d(num_classes, h_ch)
        else:
            self.b1 = nn.BatchNorm2d(in_ch)
            self.b2 = nn.BatchNorm2d(h_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1)

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.tensor, gain=math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.tensor, gain=math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.tensor, gain=1)

    def forward(self, x, y=None, z=None, **kwargs):
        return self.shortcut(x) + self.residual(x, y, z)

    def shortcut(self, x, **kwargs):
        if self.learnable_sc:
            if self.upsample:
                h = _upsample(x)
            h = self.c_sc(h)
            return h
        else:
            return x

    def residual(self, x, y=None, z=None, **kwargs):
        if y is not None:
            h = self.b1(x, y, **kwargs)
        else:
            h = self.b1(x)
        h = self.activation(h)
        if self.upsample:
            h = _upsample(h)
        h = self.c1(h)
        if y is not None:
            h = self.b2(h, y, **kwargs)
        else:
            h = self.b2(h)
        return self.c2(self.activation(h))

class MinibatchStdDev(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        batch_size = x.size(0)

        if batch_size < 2:
            return x

        var = torch.var(x, dim=0, keepdim=True, unbiased=False)
        std = torch.sqrt(var + 1e-8)
        mean_std = std.mean().view(1, 1, 1, 1).detach()

        std_feature = mean_std.expand(batch_size, 1, x.size(2), x.size(3))
        return torch.cat([x, std_feature], dim=1)
