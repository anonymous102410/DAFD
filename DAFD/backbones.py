import torch.nn as nn
from torchvision import models

resnet_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}

def get_backbone(name):
    if "dafd" in name.lower():
        if "resnet50" in name.lower():
            return DAFDBackbone('resnet50')
    elif "resnet" in name.lower():
        return ResNetBackbone(name)
    elif "alexnet" == name.lower():
        return AlexNetBackbone()
    elif "dann" == name.lower():
        return DaNNBackbone()
    
class DaNNBackbone(nn.Module):
    def __init__(self, n_input=224*224*3, n_hidden=256):
        super(DaNNBackbone, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self._feature_dim = n_hidden

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

    def output_num(self):
        return self._feature_dim
    
# convnet without the last layer
class AlexNetBackbone(nn.Module):
    def __init__(self):
        super(AlexNetBackbone, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), model_alexnet.classifier[i])
        self._feature_dim = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self._feature_dim

class ResNetBackbone(nn.Module):
    def __init__(self, network_type):
        super(ResNetBackbone, self).__init__()
        resnet = resnet_dict[network_type](pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  #256
        self.layer2 = resnet.layer2  #512
        self.layer3 = resnet.layer3  #1024
        self.layer4 = resnet.layer4  #2048
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        del resnet
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def output_num(self):
        return self._feature_dim



class DAFDBackbone(nn.Module):
    def __init__(self, network_type):
        super(DAFDBackbone, self).__init__()
        #netword
        resnet = resnet_dict[network_type](pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  #256
        self.layer2 = resnet.layer2  #512
        self.layer3 = resnet.layer3  #1024
        self.layer4 = resnet.layer4  #2048
        #MINE
        self.specific = Specific(2048)
        self.invariant = Invariant(2048)

        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        del resnet
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        front = x
        x = self.layer4(x)
        #print(x.shape) #torch.Size([32, 2048, 7, 7])
        #解耦
        tail = x
        x_specific = self.specific(x)
        x_invariant = self.invariant(x)
        #print(x_invariant.shape) #torch.Size([32, 2048, 7, 7])
        x = self.avgpool(x_invariant)
        x = x.view(x.size(0), -1)
        return x,x_invariant,x_specific,tail,front
    
    def output_num(self):
        return self._feature_dim

class Invariant(nn.Module):
    def __init__(self,in_channel):
        super(Invariant, self).__init__()
        self.di = nn.Sequential(
            nn.Conv2d(in_channel, 1024, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=False),
            # nn.Dropout(p=0.5),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=False),
            # nn.Dropout(p=0.5),
            nn.Conv2d(1024, in_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=False),
            )
    def forward(self, x):
        x = self.di(x)
        return x

class Specific(nn.Module):
    def __init__(self,in_channel):
        super(Specific, self).__init__()
        self.ds = nn.Sequential(
            nn.Conv2d(in_channel, 1024, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=False),
            # nn.Dropout(p=0.5),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=False),
            # nn.Dropout(p=0.5),
            nn.Conv2d(1024, in_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=False),
            )
    def forward(self, x):
        x = self.ds(x)
        return x
