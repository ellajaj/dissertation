from utils_libs import *
import torchvision.models as models

class client_model(nn.Module):
    def __init__(self, name, dataset_name, args=True):
        super(client_model, self).__init__()
        self.name = name
        self.dataset_name = dataset_name
        if self.name == 'Linear':
            [self.n_dim, self.n_out] = args
            self.fc = nn.Linear(self.n_dim, self.n_out)
          
        if self.name == 'mnist_2NN':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, self.n_cls)
        
        if self.name == 'cifar10_LeNet':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, 192) 
            self.fc3 = nn.Linear(192, self.n_cls)
            
        if self.name == 'Resnet18':
            resnet18 = models.resnet18()

            if self.dataset_name == 'fashion_mnist':
                in_channels = 1
            else:
                in_channels = 3

            resnet18.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            resnet18.fc = nn.Linear(512, 10)

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

        if self.name =='Resnet50':
            resnet50 = models.resnet50()
            resnet50.fc = nn.Linear(2048, 20)
            replace_bn_with_gn(resnet50, num_groups=8)

            self.model = resnet50
        
    def forward(self, x):
        if self.name == 'Linear':
            x = self.fc(x)
            
        if self.name == 'mnist_2NN':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        
        if self.name == 'cifar10_LeNet':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            
        if self.name == 'Resnet18':
            x = self.model(x)

        if self.name == 'Resnet50':
            x = self.model(x)

        return x

def replace_bn_with_gn(model, num_groups=2):
        for name, child in model.named_children():
            if isinstance(child, nn.BatchNorm2d):
                ng = num_groups if child.num_features % num_groups == 0 else 1
                
                gn = nn.GroupNorm(ng, child.num_features)
                if child.affine:
                    gn.weight.data.copy_(child.weight.data)
                    gn.bias.data.copy_(child.bias.data)
                setattr(model, name, gn)
            else:
                replace_bn_with_gn(child, num_groups)
