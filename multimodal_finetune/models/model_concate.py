# -*- coding： utf-8 -*-
'''
@Time: 2022/3/21 17:35
@Author:YilanZhang
@Filename:model_concate.py
@Software:PyCharm
@Email:zhangyilan@buaa.edu.cn
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torchvision.models as models
import argparse

class MLP(nn.Module):
    '''
    MLP that is used for the subnetwork of metadata.
    '''
    def __init__(self,in_size,hidden_size,out_size,dropout=0.5):
        '''
        :param in_size: input dimension
        :param hidden_size: hidden layer dimension
        :param out_size: output dimension
        '''
        super(MLP,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size,hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size,out_size),
            nn.BatchNorm1d(out_size),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = x.view(x.size(0),-1)


        return x



class MetaSubNet(nn.Module):
    '''
    The subnetwork that is used for metadata.
    Maybe the subnetwork that is not needed in the task
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=2, dropout=0.2, bidirectional=False):
        '''
        :param in_size: input dimension
        :param hidden_size: hidden layer dimension
        :param out_size: output dimension
        :param num_layers: specify the number of layers of LSTMs.
        :param dropout: dropout probability
        :param bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(MetaSubNet, self).__init__()
        # self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.rnn2 = MLP(in_size,hidden_size,out_size,dropout=dropout)

    def forward(self, x):
        '''
        :param x: tensor of shape (batch_size, sequence_len, in_size)
        :return: tensor of shape (batch_size,out_size)
        '''
        meta_output = self.rnn2(x)
        return meta_output



class FeatureExtractor(nn.Module):
    def __init__(self,submodule,extracted_layers):
        super(FeatureExtractor,self).__init__()
        self.submodule=submodule
        self.extracted_layers=extracted_layers

    def forward(self, x):
        outputs=[]
        for name,module in self.submodule._modules.items():
            if name == "fc" : x=torch.flatten(x,1)
            x=module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


class ImgSubNet(nn.Module):
    '''
        The subnetwork that is used for image data
    '''

    def __init__(self,out_size,dropout,args,pretrained=True):
        '''
        :param dropout: dropout probability
        :param pretrained: whether use transfer learning for the task
        '''
        super(ImgSubNet, self).__init__()
        if args['model'] == 'resnet':
            self.subnet = models.resnet50(pretrained=pretrained)
            self.fc_node = 512*4

        elif args['model'] == 'regnet':
            self.subnet = models.regnet_y_400mf(pretrained=pretrained)
            self.fc_node = 440 #1.6G888

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.fc_node,out_size)
        self.model = torch.nn.Sequential(*(list(self.subnet.children())[:-1]))


    def forward(self,x):
        x = self.model(x)
        x = x.view(x.shape[0],-1)
        x = self.dropout(x)
        img_output = self.linear(x)

        return img_output

class Concate_Model(nn.Module):
    '''
    Concate Multimodal Fusion
    '''
    def __init__(self,in_size,hidden_size,out_size,dropouts,args,num_classes=5):
        '''
        :param meta_output: output of MetaSubNet for metadata
        :param img_output1: output of ImgSubNet for clinical image
        :param img_output2: output of ImgSubNet for dermoscopy image
        :param out_size: output demension of SubNets
        :param num_classes: the number of class in the task
        '''
        super(Concate_Model, self).__init__()

        # demension are secified in the order of metadata, clinical image and dermoscopy image
        self.meta_in = in_size[0]
        self.cli_img_in = in_size[1] #224 in resnet18
        self.der_img_in = in_size[2]  #224 in resnet18, maybe we will use efficientNet in the future

        self.meta_hidden = hidden_size[0]
        # self.cli_img_hidden = hidden_size[1]
        # self.der_img_hidden = hidden_size[2]

        self.meta_out = out_size[0]
        self.cli_img_out = out_size[1]
        self.der_img_out = out_size[2]

        self.meta_prob = dropouts[0]
        self.cli_img_prob = dropouts[1]
        self.der_img_prob = dropouts[2]

        #define the pre-fusion subnetworks
        self.meta_subnet = MetaSubNet(self.meta_in, self.meta_hidden, self.meta_out, num_layers=2, dropout=self.meta_prob)
        self.cli_subnet = ImgSubNet(self.cli_img_out, self.cli_img_prob, args,pretrained=args["pretrained"])
        self.der_subnet = ImgSubNet(self.der_img_out, self.der_img_prob, args,pretrained=args["pretrained"])

        self.num_classes = num_classes

        self.fc = nn.Linear(self.meta_out+self.cli_img_out+self.der_img_out, self.num_classes) # only use for three modalities
        # self.fc = nn.Linear(self.meta_out+self.der_img_out , self.num_classes)
        # self.fc = nn.Linear(self.cli_img_out+self.der_img_out, self.num_classes) #only use two image modalities
        # self.fc = nn.Linear(self.der_img_out,self.num_classes) # only use one image modality


    def forward(self, meta_x, cli_x, der_x):
        '''
        :param meta_x: tensor of shape (batch_size, meta_in)
        :param cli_x: tensor of shape (batch_size, cli_img_in)
        :param der_x: tensor of shape (batch_size, der_img_out)
        '''
        meta_h = self.meta_subnet(meta_x)
        cli_h = self.cli_subnet(cli_x)
        der_h = self.der_subnet(der_x)

        #feature fusion
        x = torch.cat([meta_h,cli_h,der_h], dim=-1)
        # print(x.size())

        x = self.fc(x)

        return x

    def criterion(self, logit, truth,weight=None):
        if weight == None:
            loss = F.cross_entropy(logit, truth)
        else:
            loss = F.cross_entropy(logit, truth,weight=weight)

        return loss


if __name__ == "__main__":
    in_size=[3,224,224]
    hidden_size=[16,64,64]
    out_size=[1,8,8]
    dropouts=[0.5,0.5,0.5]

    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument('--model', dest='model', type=str, default='regnet')
    args = vars(OPTIONS.parse_args())


    model = Concate_Model(in_size,hidden_size,out_size,dropouts,args)
    print(model)

    meta = torch.randn(4,1)
    cli = torch.randn(4,3,224,224)
    der = torch.randn(4,3,224,224)

    outputs= model(Variable(meta),Variable(cli),Variable(der))
    print(outputs)



