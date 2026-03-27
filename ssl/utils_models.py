''' some code adapted from https://github.com/taufikxu/Triple-GAN.git - MIT license'''
from utils_libs import *

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
        if self.name == 'fashion_mnist':
            self.channels = 1
        #if self.name == 'CIFAR10':
        else:
            self.channels = 3

        in_channels = z_dim + num_classes if num_classes > 0 else z_dim

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

        z_combined = z_combined.view(z_combined.size(0), z_combined.size(1), 1, 1)
        return self.main(z_combined)
        
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
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        combined_feat = torch.cat([out, feat], dim=1)
        logits = self.fc(self.actvn(combined_feat))

        batch_range = torch.arange(logits.size(0), device=x.device)
        return logits[batch_range, y]


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

def _upsample(x):
    h, w = x.size()[2:]
    return F.interpolate(x, size=(h * 2, w * 2), mode="bilinear", align_corners=False)




