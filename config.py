import argparse

parser = argparse.ArgumentParser()

######################## SSTA model params ########################
parser.add_argument('--filter_size', type=int, default=3)
parser.add_argument('--img_width', type=int, default=128, help='img width')
parser.add_argument('--img_channel', type=int, default=3, help='img channel')
parser.add_argument('--num_hidden', type=str, default='32,32,32,64', help='64,64,64,64')
parser.add_argument('--stride', type=int, default=1)

######################## SSTA inference params ########################
parser.add_argument('--ssta_ckpt_dir', type=str, default='/home/knagaraj31/SSTA_2024/ssta2_sim_model_128_MSE_ssim2_only_32_32_32_64/SSTA_model/40', help='checkpoint dir')
parser.add_argument('--vae_ckpt_dir', type=str, default="/home/knagaraj31/SSTA_2024/vae_file",help='None')
parser.add_argument('--num_views', type=int, default=2)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--message_type', type=str, default='raw_data', help='normal, zeros, randn, raw_data, vae')

######################## inference/training/visualization params ########################
parser.add_argument('--threshold_time_step_gt', type=int, default=50,help="timestep of t2no/t2nd")
parser.add_argument('--threshold_time_step_pd', type=int, default=50,help="timestep of t2no/t2nd")
parser.add_argument('--vis', type = bool, default = True, help='visualize results or not')
parser.add_argument('--t2n_cmap', type = str, default = 'viridis', help ="viridis/gray")

args = parser.parse_args()
