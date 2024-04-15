import numpy as np
import time
from matplotlib import pyplot as plt
import os, cv2


# pytorch
from ssta_net import SSTA_Net

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import random
from skimage.metrics import structural_similarity as compare_ssim
from VAE_model import *
from models import *
# from ssta_net import *
import sys
import gc
from tqdm import tqdm
torch.cuda.empty_cache()
gc.collect()
# import torchvision.models as models
import torch
# from torchmetrics.image import StructuralSimilarityIndexMeasure

from config import args


seed = 0
random.seed(seed)

class SSTA_predictor:
    def __init__(self, args):

        self.args = args
        
        self.vae = self.load_vae('vae.pt')
        self.models, self.connections = self.load_sstas(self.args.num_views, self.args.ssta_ckpt_dir)

        self.messages = {}
        for ssta_model in self.models.keys():
            self.messages[ssta_model] = torch.zeros((1,1,args.img_width,args.img_width,3)).to(args.device)
        self.memory = [None for _ in range(args.num_views)]
    def get_predictions(self, inputs):
        inputs = np.float32(inputs/255.00)
        inputs = torch.tensor(inputs).unsqueeze(1).to(args.device)
        # print(inputs.shape)
        outputs = []
        with torch.no_grad():
            for view, model_name in enumerate(self.models.keys()):
                message_others = self.get_relevant_msgs(model_name, self.messages, self.connections)
                output, _, self.memory[view] = self.models[model_name](inputs[view].unsqueeze(0), self.messages[model_name], message_others, self.memory[view])
                outputs.append(output)

                if args.message_type in ['vae']:
                    self.messages[model_name] = self.vae.get_message(inputs[view].unsqueeze(0))
                        # message_0 = vae.get_message(x_t[:, t:t + 1])

                elif args.message_type in ['raw_data']:
                    self.messages[model_name] = inputs[view].unsqueeze(0)

                elif args.message_type == 'zeros':
                    self.messages[model_name] = torch.zeros_like(self.messages[model_name])

                elif args.message_type == 'randn':
                    self.messages[model_name] = torch.randn_like(self.messages[model_name])



            output1 = outputs[0][:,:,:,:,0].detach().cpu().numpy().squeeze()
            border_v = np.zeros((output1.shape[0],1,3))
            output2 = outputs[1][:,:,:,:,0].detach().cpu().numpy().squeeze()

            output1 = cv2.cvtColor(output1,cv2.COLOR_GRAY2RGB)
            output2 = cv2.cvtColor(output2,cv2.COLOR_GRAY2RGB)

            output = np.hstack([output1, border_v, output2])
            # border_v = cv2.cvtColor(border_v,cv2.COLOR_GRAY2RGB)

            # print(output1.shape, border_v.shape)
            # cv2.imshow('not gonna work',output)
            # cv2.waitKey(1)
            # print(output.shape)

            input1 = inputs[0,:,:,:,:].detach().cpu().numpy().squeeze()
            input2 = inputs[1,:,:,:,:].detach().cpu().numpy().squeeze()
            input = np.hstack([input1, border_v, input2])

            border_h = np.zeros((1,output.shape[1],3))  

            final = np.vstack([input, border_h, output])
            cv2.imshow('not gonna work',final)
            cv2.waitKey(1)
        pass

    def train(self):
        pass

    def load_sstas(self, n, weights_dir):
        models = {}
        print("Loading from: {}".format(weights_dir))
        for i in range(n):
            ssta_name = 'ssta_' + str(i)
            path = os.path.join(weights_dir, ssta_name+'.pt')
            models[ssta_name] = torch.load(path)
            models[ssta_name] = models[ssta_name].to(self.args.device)
            print("Loaded model: ", ssta_name)

            #### The connections need to be stored as well, meaning this should be moved and just obtained from a save file ####
            connections = {'ssta_0': ['ssta_1'], 'ssta_1': ['ssta_0']}

        return models, connections

    def load_vae(self, file_name):
        vae_path = os.path.join(self.args.vae_ckpt_dir, file_name)
        print(vae_path)
        vae = torch.load(vae_path)
        vae = vae.to(self.args.device)
        print("Loaded model: ", 'vae')

        return vae
    
    def get_relevant_msgs(self, model_key, messages, connections):
        relevant_msgs = []
        relevant_connections = connections[model_key]
        for ssta in relevant_connections:
            relevant_msgs.append(messages[ssta])
        return relevant_msgs
    
if __name__ == '__main__':
    predictor = SSTA_predictor(args)



