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
import queue
from config import args


seed = 0
random.seed(seed)

class SSTA_predictor:
    def __init__(self, args):

        self.is_train = True

        

        self.args = args
        
        self.vae = self.load_vae('vae.pt')
        self.models, self.connections = self.load_sstas(self.args.num_views, self.args.ssta_ckpt_dir)

        if self.is_train:
            parameters = []
            for name, _ in self.models.items():
                self.models[name] = self.models[name].to(args.device)
                parameters += list(self.models[name].parameters())

            self.optimizer = optim.Adam(parameters,lr = 0.0001)

        self.messages = {}
        for ssta_model in self.models.keys():
            self.messages[ssta_model] = torch.zeros((1,1,args.img_width,args.img_width,3)).to(args.device)
        self.memory = [None for _ in range(args.num_views)]
        self.q_E = queue.Queue(args.threshold_time_step_gt)
        self.q_E_t2nd = queue.Queue(args.threshold_time_step_gt)
        self.B = np.ones((128,128))*255.0

        self.t2no = None
        self.t2nd = None

        self.vis = args.vis

        if self.vis:
            self.pred_q = queue.Queue(args.threshold_time_step_gt)
            self.inputs_q = queue.Queue(args.threshold_time_step_gt)

        self.loss = nn.MSELoss()

    def get_predictions(self, inputs):
        # print(inputs.shape)
        image = np.uint8(np.dot(inputs[...,:3], [0.200, 0.587, 0.114])) ## convert to grayscale
        
        t2no = self.compute_t2no(image)
        t2nd = self.compute_t2nd(image, t2no)
      
        inputs = np.float32(inputs/255.00)

        inputs = torch.tensor(inputs).unsqueeze(1).to(args.device)
        # print(inputs.shape)
        outputs = []
        


        with torch.no_grad():
        
            for view, model_name in enumerate(self.models.keys()):
                # print(model_name)
                message_others = self.get_relevant_msgs(model_name, self.messages, self.connections)
                print(len(self.messages))
                output, _, memory_temp = self.models[model_name](inputs[view].unsqueeze(0), self.messages[model_name], message_others, self.memory[view])
                self.memory[view] = [(mem1.detach(), mem2.detach()) for mem1,mem2 in memory_temp]
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


        # if self.is_train:
            # gt_t2ns =  np.concatenate([t2no[view][np.newaxis,:,:], t2nd[view][np.newaxis,:,:]], axis = 0)
            # gt_t2ns= torch.tensor(gt_t2ns, dtype=torch.float32).permute((1,2,0)).to(self.args.device)
            # gt_t2ns.detach().cpu()
            # self.train(output.squeeze(), gt_t2ns)

        output0 = outputs[0][:,:,:,:,:].detach().cpu().numpy().squeeze()
        output1 = outputs[1][:,:,:,:,:].detach().cpu().numpy().squeeze()
        inputs = inputs[:,:,:,:,:].detach().cpu().numpy().squeeze()
        outputs = [output0, output1]


        if self.vis:
            self.visualize(outputs, inputs, t2no, t2nd)

        return t2no, t2nd

    def train(self, pred, gt):
        

        loss = self.loss(pred, gt)
        # print('yse')
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        print(pred.shape, gt.shape)
        pass

    def load_sstas(self, n, weights_dir):
        models = {}
        print("Loading from: {}".format(weights_dir))
        for i in range(n):
            ssta_name = 'ssta_' + str(i)
            path = os.path.join(weights_dir, ssta_name+'.pt')
            models[ssta_name] = torch.load(path)
            models[ssta_name] = models[ssta_name].to(self.args.device)
            print("Loaded cv2.cvtColor(inputs,cv2.COLOR_RGB2GRAY)model: ", ssta_name)

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
    
    def compute_t2no(self, frame):

        diff_img_T = (np.abs(self.B[np.newaxis,:,:]-frame).astype(np.uint8) > 70) * 255.  # [False, True]
        self.q_E.put(diff_img_T)

        if not self.q_E.full():
            
            return np.array([np.ones((128,128))]*self.args.num_views)

        q_E_t2no_array = np.asarray(list(self.q_E.queue) + [np.ones_like(diff_img_T) * 255.])

        t2no_img = np.argmax(q_E_t2no_array, axis=0)
        self.q_E.get()

        return t2no_img

    def compute_t2nd(self, frame, t2no):
        diff_img_t2nd_T = (np.abs(self.B[np.newaxis,:,:]-frame).astype(np.uint8) < 70) * 255. 
        self.q_E_t2nd.put(diff_img_t2nd_T)

        if not self.q_E_t2nd.full():
            
            return np.array([np.ones((128,128))]*self.args.num_views)
        
        q_E_t2nd_array = np.asarray(list(self.q_E_t2nd.queue))

        t2nd_img = np.argmax(q_E_t2nd_array, axis=0)
        infty_mask = np.logical_or((np.abs(t2no - self.args.threshold_time_step_gt) < 1e-2), (np.abs(t2nd_img) < 1e-2))
        t2nd_img[infty_mask] = self.args.threshold_time_step_gt

        self.q_E_t2nd.get()

        return t2nd_img


    def visualize(self, pred, input, t2no, t2nd):

        ## obtain viridis cmap

        gradient = np.linspace(0,1,256)
        gradient = np.vstack((gradient, gradient))

        viridis_colormap = plt.cm.get_cmap('viridis')
        viridis_rgb = (viridis_colormap(gradient)[:,:3]*255).astype(np.uint8)

        self.pred_q.put(pred)
        self.inputs_q.put(input)
        
        outputs, inputs = pred, input
        if not self.pred_q.full():
            outputs = [np.ones_like(pred[0])]*2
            inputs = np.ones_like(input)
        else:
            outputs, inputs = self.pred_q.get(), self.inputs_q.get()

        outputso = []
        outputsd = []
        inputs_arr = []
        t2nos = []
        t2nds = []

        vis_t2no = np.float32(t2no/self.args.threshold_time_step_gt)
        vis_t2nd = np.float32(t2nd/self.args.threshold_time_step_gt)

        for i in range(self.args.num_views):

            outputo = outputs[i][:,:,0]
            outputd = outputs[i][:,:,1]

            threshold_ratio = self.args.threshold_time_step_gt/self.args.threshold_time_step_pd
            
            if self.args.t2n_cmap == 'viridis':
                outputo = np.float32(plt.cm.viridis(outputo)[:,:,:3])
                outputd = np.float32(plt.cm.viridis(outputd)[:,:,:3])

                outputo = cv2.cvtColor(outputo,cv2.COLOR_BGR2RGB)
                outputd = cv2.cvtColor(outputd,cv2.COLOR_BGR2RGB)
            else:
                outputo = np.clip(cv2.cvtColor(outputo,cv2.COLOR_GRAY2RGB),0,threshold_ratio)*(1/threshold_ratio)
                outputd = np.clip(cv2.cvtColor(outputd,cv2.COLOR_GRAY2RGB),0,threshold_ratio)*(1/threshold_ratio)

            border_v = np.zeros((inputs[i].shape[0],1,3))
            # print(border_v.shape)

            outputso.append(np.hstack([outputo, border_v]))
            outputsd.append(np.hstack([outputd, border_v]))

            if self.args.t2n_cmap =='viridis':
                vis_t2no_ = np.float32(plt.cm.viridis(vis_t2no[i])[:,:,:3])
                vis_t2nd_ = np.float32(plt.cm.viridis(vis_t2nd[i])[:,:,:3])

                vis_t2no_ = cv2.cvtColor(vis_t2no_,cv2.COLOR_BGR2RGB)
                vis_t2nd_ = cv2.cvtColor(vis_t2nd_,cv2.COLOR_BGR2RGB)
            else:
                vis_t2no_ = cv2.cvtColor(vis_t2no[i],cv2.COLOR_GRAY2RGB)
                # print(vis_t2no_.dtype())
                vis_t2nd_ = cv2.cvtColor(vis_t2nd[i],cv2.COLOR_GRAY2RGB)
            
            t2nos.append(np.hstack([vis_t2no_, border_v]))
            t2nds.append(np.hstack([vis_t2nd_, border_v]))

            inputs_arr.append(np.hstack([inputs[i], border_v]))

        outputs = np.hstack([border_v] + outputso + [border_v] + outputsd)

        inputs =  [border_v]+inputs_arr
        inputs = np.hstack(inputs*args.num_views)

        border_h = np.zeros((1,inputs.shape[1],3))

        t2ns = np.hstack([border_v] + t2nos + [border_v] + t2nds)
        
        final = np.vstack([border_h, border_h, inputs, border_h,  t2ns, border_h, outputs, border_h, border_h])
        border_v2 = np.zeros((final.shape[0],1,3))

        final_final = np.hstack([border_v2, final, border_v2])

        cv2.imshow('not gonna work',final_final)
        cv2.waitKey(1)

        
        # output1o = outputs[0][:,:,0]
        # output2o = outputs[1][:,:,0]
        # border_v = np.zeros((output1o.shape[0],1,3))

        # output1d = outputs[0][:,:,1]
        # output2d = outputs[1][:,:,1]

        # threshold_ratio = self.args.threshold_time_step_gt/self.args.threshold_time_step_pd
        # output1o = np.clip(cv2.cvtColor(output1o,cv2.COLOR_GRAY2RGB),0,threshold_ratio)*(1/threshold_ratio)
        # output2o = np.clip(cv2.cvtColor(output2o,cv2.COLOR_GRAY2RGB), 0,threshold_ratio)*(1/threshold_ratio)
        # outputo = np.hstack([border_v, output1o, border_v, output2o, border_v])

        # output1d = np.clip(cv2.cvtColor(output1d,cv2.COLOR_GRAY2RGB),0,threshold_ratio)*(1/threshold_ratio)
        # output2d = np.clip(cv2.cvtColor(output2d,cv2.COLOR_GRAY2RGB), 0,threshold_ratio)*(1/threshold_ratio)
        # outputd = np.hstack([border_v, output1d, border_v, output2d, border_v])

        # outputs = np.hstack([outputo, outputd])

        # vis_t2no = np.float32(t2no/self.args.threshold_time_step_gt)
        # vis_t2no1 = cv2.cvtColor(vis_t2no[0],cv2.COLOR_GRAY2RGB)
        # vis_t2no2 = cv2.cvtColor(vis_t2no[1],cv2.COLOR_GRAY2RGB)
        # t2nos = np.hstack([border_v, vis_t2no1, border_v, vis_t2no2, border_v])

        # vis_t2nd = np.float32(t2nd/self.args.threshold_time_step_gt)
        # vis_t2nd1 = cv2.cvtColor(vis_t2nd[0],cv2.COLOR_GRAY2RGB)
        # vis_t2nd2 = cv2.cvtColor(vis_t2nd[1],cv2.COLOR_GRAY2RGB)
        # t2nds = np.hstack([border_v, vis_t2nd1, border_v, vis_t2nd2, border_v])
        
        # t2ns = np.hstack([t2nos, t2nds])

        # input1 = inputs[0]
        # input2 = inputs[1]
        # input = np.hstack([border_v, input1, border_v, input2, border_v]*2)
        # # input = np.hstack([border_v, input1, border_v, input2, border_v])


        # border_h = np.zeros((1,outputs.shape[1],3))  

        # # final = np.vstack([border_h, input, border_h, t2nos, border_h, t2nds, border_h, outputo, border_h, outputd, border_h])

        # final = np.vstack([input, border_h, t2ns,border_h, outputs])
        # cv2.imshow('not gonna work',final)
        # cv2.waitKey(1)

    
if __name__ == '__main__':
    predictor = SSTA_predictor(args)



