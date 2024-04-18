#######Please modify the paramters in this file to visualise the desired characteristics
import argparse
import numpy as np

parser = argparse.ArgumentParser()

######################## SSTA model params ########################
parser.add_argument('--filter_size', type=int, default=3)
parser.add_argument('--img_width', type=int, default=128, help='img width')
parser.add_argument('--img_channel', type=int, default=3, help='img channel')
parser.add_argument('--num_hidden', type=str, default='32,32,32,64', help='64,64,64,64')
parser.add_argument('--stride', type=int, default=1)

######################## SSTA inference params ########################
parser.add_argument('--ssta_ckpt_dir', type=str, default=r'C:\Users\Welcome\Documents\Kouby\M.S.Robo- Georgia Tech\GATECH LABS\SHREYAS_LAB\Simulation_Environment\Github Simulation Network\ssta_vae_trained_models\ssta_file\51', help='checkpoint dir')
parser.add_argument('--vae_ckpt_dir', type=str, default=r'C:\Users\Welcome\Documents\Kouby\M.S.Robo- Georgia Tech\GATECH LABS\SHREYAS_LAB\Simulation_Environment\Github Simulation Network\ssta_vae_trained_models\vae_file',help='None')
parser.add_argument('--num_views', type=int, default=2)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--message_type', type=str, default='raw_data', help='normal, zeros, randn, raw_data, vae')

######################## inference/training/visualization params ########################
parser.add_argument('--threshold_time_step_gt', type=int, default=100,help="timestep of t2no/t2nd")
parser.add_argument('--threshold_time_step_pd', type=int, default=50,help="timestep of t2no/t2nd")
parser.add_argument('--vis_T2NO_D', type = bool, default = False, help='visualize results or not')
parser.add_argument('--do_inference', type = bool, default = False, help ="Predict T2no/d images or not")
parser.add_argument('--lifelong_learning', type = bool, default = False, help ="Training ssta agents on live data or not")
parser.add_argument('--t2n_cmap', type = str, default = 'viridis', help ="viridis/gray")

######################## Simulation environment params ########################
  ############# Flags ########################
parser.add_argument('--debugging', type = bool, default = True, help ="Show debugging points and view frames(ssta) or not")
parser.add_argument('--save_data', type = bool, default = False, help ="Saves the  input frames and t2no/d images  or not")
parser.add_argument('--enable_ssta_agents', type = bool, default = True, help ="Add in SSTA agents to the APF agents or not")
parser.add_argument('--generate_new_agents', type = bool, default = True, help ="generate/spawn new agents")
parser.add_argument('--display_metric', type = bool, default = True, help ="Displays the metrics when set to True")
parser.add_argument('--display_realistic', type = bool, default = False, help ="Makes the agents with a realistic image")


  ############# Simulation Display Parameters ########################
parser.add_argument('--seed', type = int, default =42 , help =" random seed ")
parser.add_argument('--window_width', type = int, default =1000 , help ="Simulation Window Width")
parser.add_argument('--window_height', type = int, default =800 , help ="Simulation Window Height")
parser.add_argument('--frame_rate', type = int, default =60 , help =" Display Frame Rate")
parser.add_argument('--agents_spawning_frequency', type = int, default =200 , help =" Time interval at which new agents are created")
parser.add_argument('--ssta_spawning_percentage', type = int, default =20 , help =" Precentage of SSTA agents in the newly spawned agents")

 ############ static definition simulation ##########
parser.add_argument('--ssta_boxes', type = list, default = np.array([[-60,300,400,300],[-60,450,50,300]]) , help =" ssta box dimensions #angle,x,y,size")
parser.add_argument('--obstacles', type = dict, default =  {'circle': np.array([]),
        'rectangle': np.array([[500,500,30,30],
                                [700,700,30,30],
                                [300,300,30,30],
                                [700,300,30,30],
                                [300,700,30,30],
                                [300,500,30,30],
                                [500,300,30,30],
                                [700,500,30,30],
                                [500,700,30,30]])}, help ="Define the circular and rectangular objects here")


args = parser.parse_args()