from torchvision.models import resnet101
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFace(nn.Module):
    
    def __init__(self, num_class):
        super(ArcFace, self).__init__()

        m = resnet101(pretrained=True)
        resnet101encoder = Resnet101Encoder(m.conv1, 
                                            m.bn1, 
                                            m.layer1, 
                                            m.layer2, 
                                            m.layer3, 
                                            m.layer4)
        self.encoder = resnet101encoder
        
        self.bn_2 = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout(0.1)
        self.fc_1 = nn.Linear(2048, 512)
        self.bn_3 = nn.BatchNorm1d(512)
        
        '''
        we explore the BN [14]-Dropout [31]-FC-BN structure to get 
        the final 512-D embedding feature.
        '''
        
        self.speaker_embedding = nn.Linear(512, num_class, bias=False)
    
    def forward(self, input_tensor):
                
        tensor = self.encoder(input_tensor)
            
        tensor = self.bn_2(tensor)
        tensor = self.dropout(tensor)
        tensor = self.fc_1(tensor)
        tensor = self.bn_3(tensor)
        
        tensor = self.speaker_embedding(tensor)

        tensor = F.log_softmax(tensor, dim=-1)
        
        return tensor

class Resnet101Encoder(nn.Module):
    def __init__(self, conv1, bn1, layer1, layer2, layer3, layer4):
        super(Resnet101Encoder, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, input_tensor):
        
        tensor = self.conv1(input_tensor)
        tensor = self.bn1(tensor)
        tensor = self.relu(tensor)
        tensor = self.maxpool(tensor)

        tensor = self.layer1(tensor)
        tensor = self.layer2(tensor)
        tensor = self.layer3(tensor)
        tensor = self.layer4(tensor)

        tensor = self.avgpool(tensor)
        tensor = torch.flatten(tensor, 1)
        
        return tensor