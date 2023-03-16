import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision import models
from torchsummary import summary
from torchstat import stat
import torch.nn.functional as F

import pprint

# consist of (conv -- bn -- activation)
class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, stride=1, pad=0, bn=True, act_type='relu'):
        super(ConvBlock, self).__init__()
        
        layer_list = []
        layer_list += [nn.Conv2d(in_dim, out_dim, kernel_size=kernel, stride=stride, padding=pad)]
        
        # Make BatchNorm layer
        if bn == True:
            layer_list += [nn.BatchNorm2d(out_dim, affine=True)]
        
        # Make activation layer
        if act_type == 'relu':
            layer_list += [nn.ReLU(inplace=True)]
        elif act_type == 'leakyrelu':
            layer_list += [nn.LeakyReLU(negative_slope=0.01, inplace=True)]
        elif act_type == 'prelu':
            layer_list += [nn.PReLU()]
        elif act_type == None:
            pass
        
        self.conv_block = nn.Sequential(*layer_list)
        
    def forward(self, x):
        out = self.conv_block(x)
        return out

class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, stride=1, pad=1):            
        super(ResBlock, self).__init__()
        
        conv_block_1 = ConvBlock(in_dim, out_dim, kernel=kernel, stride=stride, pad=pad, 
                                 bn=True, act_type='relu')
        conv_block_2 = ConvBlock(in_dim, out_dim, kernel=kernel, stride=stride, pad=pad,
                                 bn=True, act_type=None)
        
        self.res_block = nn.Sequential(conv_block_1, conv_block_2)
    
    def forward(self, x):
        out = x + self.res_block(x)
        return out

num_classes = 7
Generator = models.wide_resnet50_2(pretrained=True)
num_ftrs = Generator.fc.in_features
Generator.fc = nn.Linear(num_ftrs, num_classes)

class Discirminator(nn.Module):
    def __init__(self, init_weights = True):
        super(Discirminator,self).__init__()
        # (-1, 512, 28, 28) -> (-1, 256, 28, 28)
        conv_1 = ConvBlock(512, 256 , kernel=3, stride=1, pad=1, bn=True, act_type='relu')
        res_1 = ResBlock(256, 256, kernel=3, stride=1, pad=1) 
        res_2 = ResBlock(256, 256, kernel=3, stride=1, pad=1) 
        # (-1, 256, 28, 28) ---> (-1, 128, 14, 14)
        conv_2 = ConvBlock(256, 128 , kernel=3, stride=2, pad=1, bn=True, act_type='relu')
        # (-1, 128, 14, 14) ---> (-1, 64, 7, 7)
        conv_3 = ConvBlock(128, 64 , kernel=3, stride=2, pad=1, bn=True, act_type='relu')
        
        fc_1 = nn.Sequential(nn.Linear(64*7*7, 1000), nn.ReLU(), nn.Dropout(0.2))  
        fc_2 = nn.Sequential(nn.Linear(1000, 100), nn.ReLU(), nn.Dropout(0.2))  
        fc_3 = nn.Linear(100, 1) 
        
        
        self.conv_blocks = nn.Sequential(conv_1, res_1, res_2, conv_2, conv_3)
        self.predictor = nn.Sequential(fc_1, fc_2, fc_3)

        self.conv_4 = ConvBlock(64, 1, kernel=7, stride=1, pad=1, bn=True, act_type='relu')
        self.fc_4 = nn.Linear(9, 1) 

        if init_weights:
            self._initialize_weights()
    
    
    def forward(self, x):
        out = self.conv_blocks(x)
        print(out.shape)
        # fairness prediction branch
        p_score = self.predictor(out.view(-1, 64*7*7))
        # discriminator output
        d_score = self.fc_4(self.conv_4(out).view(-1,9))
        return p_score, d_score
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

########## HOOK MIDDLE LAYER ###########
activations = []
def hook_acts(module, input, output):
    activations.append(output)

def get_acts(model, input):
    del activations[:]
    _ = model(input)
    # print(activations[0].shape)
    assert (len(activations) == 1)
    return activations[0]

# extract the middle level feature of network
def extract_feature(net, input):
    x = net.conv1(input)
    x = net.bn1(x)
    x = net.relu(x)
    x = net.maxpool(x)
    x = net.layer1(x)
    x = net.layer2(x)
    return x

if __name__ == '__main__':
    discirminator = Discirminator().to(torch.device('cuda:0'))
    summary(discirminator, (512,28,28))
    generator = Generator.to(torch.device('cuda:0'))
    feature = extract_feature(generator, torch.cuda.FloatTensor(1,3,224,224))
    print(feature.shape)    # (1,512,28,28)
    p_score, d_score = discirminator(feature)
    print(p_score.shape, d_score.shape)
