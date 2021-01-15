from torchvision.models import resnet101
import torch
import torch.nn as nn
import torch.nn.functional as F

def label2mask(label, h):
    B = len(label)
    H = h
    mask = torch.zeros([B, H], requires_grad=False)
    mask[torch.arange(B), label] = 1
    # for i, l in enumerate(label):
    #     mask[i, l] = 1.0
        
    return mask

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

        # self.fc_2 = nn.Linear(512, 128)
        # self.bn_4 = nn.BatchNorm1d(128)
        
        self.identity_embedding = nn.utils.weight_norm(nn.Linear(512, num_class, bias=False), dim=0)
        # self.identity_embedding = nn.utils.weight_norm(nn.Linear(128, num_class, bias=False), dim=0)
    
    def forward(self, input_tensor):
                
        tensor = self.encoder(input_tensor)
            
        tensor = self.bn_2(tensor)
        tensor = self.dropout(tensor)
        tensor = self.fc_1(tensor)
        tensor = self.bn_3(tensor)

        # Extra layers
        # tensor = self.fc_2(tensor)
        # tensor = self.bn_4(tensor)
        
        # tensor = self.speaker_embedding(tensor)

        tensor_g = torch.norm(tensor, dim=1, keepdim=True)
        normalized_tensor = tensor / tensor_g

        tensor = self.identity_embedding(normalized_tensor)
        layer_g = self.identity_embedding.weight_g
        tensor = tensor / layer_g.squeeze(1)

        # Additive Margin
        # mask = label2mask(ground_truth_tensor, self.num_speakers).to(self.device)

        # masked_embedding = tensor * mask
        # modified_angle = torch.acos(masked_embedding) + self.m * mask
        # modified_angle = torch.clamp(modified_angle, 0, torch.acos(torch.tensor(-1.0)))
        # modified_cos = torch.cos(modified_angle)
        # tensor = tensor + modified_cos - masked_embedding

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