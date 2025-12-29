import torch
import torch.nn as nn
import numpy as np

class Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        
        # Using a standard CNN backbone
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # Change 3 to 1 for grayscale
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
        
        # The head that outputs the class probabilities
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        # Note: Return logits. We apply Softmax in the loss function (CrossEntropy)
        # or use it to generate pseudo-labels for the Discriminator.
        return logits


class Generator(nn.Module):
    def __init__(self, z_dim=256, n_label=10, im_size=32, im_chan=3,embed_size=256,
        nfilter=64, nfilter_max=512, actvn=nn.ReLU(),):
        super().__init__()
        self.actvn = actvn
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        self.z_dim = z_dim
        self.nlabels = n_label
        self.im_size = im_size
        self.im_chan = im_chan

        # Submodules
        nlayers = int(np.log2(im_size / s0))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        self.embedding = nn.Embedding(n_label, embed_size)
        self.fc = nn.Linear(z_dim + embed_size, self.nf0 * s0 * s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2 ** (nlayers - i), nf_max)
            nf1 = min(nf * 2 ** (nlayers - i - 1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1, actvn=self.actvn),
                nn.Upsample(scale_factor=2),
            ]

        blocks += [ResnetBlock(nf, nf, actvn=self.actvn)]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, im_chan, 3, padding=1)

    def forward(self, z, y):
        assert z.size(0) == y.size(0)
        batch_size = z.size(0)

        yembed = self.embedding(y)
        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)
        # print(z.shape, yembed.shape)

        yz = torch.cat([z, yembed], dim=1)
        out = self.fc(yz)
        out = out.view(batch_size, self.nf0, self.s0, self.s0)

        out = self.resnet(out)

        out = self.conv_img(self.actvn(out))
        out = torch.tanh(out)

        return out

class Discriminator(nn.Module):

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

        super().__init__()
        self.actvn = actvn
        self.embed_size = embed_size
        self.nlabels = n_label
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        self.im_size = im_size
        self.im_chan = im_chan

        # Submodules
        nlayers = int(np.log2(im_size / s0))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        blocks = [ResnetBlock(nf, nf, actvn=self.actvn)]

        for i in range(nlayers):
            nf0 = min(nf * 2 ** i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1, actvn=self.actvn),
            ]


        self.conv_img = nn.Conv2d(im_chan, 1 * nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0 * s0 * s0, n_label)

    def forward(self, x, y=None):
        batch_size = x.size(0)

        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        out = self.fc(self.actvn(out))

        if y is None:
            return out

        index = torch.LongTensor(range(out.size(0)))
        if x.is_cuda:
            index = index.cuda()
            y = y.cuda()

        out = out[index, y]
        return out

class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, actvn, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.actvn = actvn
        self.is_bias = is_bias
        self.learned_shortcut = fin != fout
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(
            self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias
        )
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(
                self.fin, self.fout, 1, stride=1, padding=0, bias=False
            )

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(self.actvn(x))
        dx = self.conv_1(self.actvn(dx))
        out = x_s + 0.1 * dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s
