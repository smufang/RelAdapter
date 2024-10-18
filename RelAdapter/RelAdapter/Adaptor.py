import logging
import torch.nn as nn
import torch
from params import *
from collections import OrderedDict

class Metanet(nn.Module):
    def __init__(self,parameter):
        super(Metanet, self).__init__()
        self.params = get_params()
        self.device = parameter['device']
        self.neuron = parameter['neuron']



        # self.linear = nn.Linear(self.params['embed_dim'], 50)
        # self.bn = nn.BatchNorm2d(num_features = None, affine = False,track_running_stats = False )

        self.MLP1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(self.params['embed_dim'], self.neuron)),
            # ('bn',   nn.BatchNorm2d(num_features = None, affine = False,track_running_stats = False )),
            ('relu', nn.LeakyReLU()),
            # ('drop', nn.Dropout(p=0.1)),
        ]))

        self.output = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(self.neuron, self.params['embed_dim'])),
        ]))

        nn.init.xavier_normal_(self.MLP1.fc.weight)
        nn.init.xavier_normal_(self.output.fc.weight)

    def forward(self, rel_agg):

        size = rel_agg.shape
        # rel_agg = rel_agg.contiguous().view(size[0], size[1], -1)
        # a = self.linear(rel_agg)
        # b = self.bn(a)
        MLP1 = self.MLP1(rel_agg).cuda().to(self.device)
        oupt = self.output(MLP1).to(self.device)
        # oupt = torch.mean(oupt, 1)
        # oupt = oupt.view(size[0], 1, 1, 2*self.params['embed_dim'])
        return oupt

class WayGAN2(object):
    def __init__(self,parameter):
        # self.args = args
        logging.info("Building Metanet...")

        metanet = Metanet(parameter)
        self.metanet = metanet

    def getVariables2(self):
        return (self.metanet)

    def getWayGanInstance(self):
        return self.waygan1
