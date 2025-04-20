import pygame
import numpy as np
from controllers_APF import *
from controllers_SSTA import *
# from test_cases import *
from loop_agents import *
from Environment import Environment
from APF_SSTA_agents import *
from Planners.path_planners import *
from predict import *
from config import args as config
import sys
import cv2
import torch
import torch
import os
import shutil
# print(torch.__version__)
# print(torch.version.cuda)  # Should return a version number, not None
# print(torch.backends.cudnn.version())  # Should return a number, not None

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available")



###REPLANNING - 10 POINTS EVERY TSECONDS REPLANNING MUST BE DONE AND FUNCTION CALLED FOR UPDATE, PATH NUMBER DOESNT NEED TO CHANGE,VARIABEL PATH LENGTH ,
#TIME BASED varied path replanning call that function

###########    Solved -----Core Problem2 :: None case for both apf and ssta agents   , code breaks if either one of the agent becomes None   ###### 

class CarSimulation(Environment):
    def __init__(self, args):
        super().__init__(args.window_height, args.window_width, args.obstacles, args)

        # Initialize Pygame necessary for initialising the simulation window and graphics
        print('Initializing Agents')
        pygame.init()
        pygame.display.set_caption("Car Simulation")#Windows heading
        self.args = args
        self.debugging = args.debugging
        self.save_data = args.save_data
        self.save_video = args.save_video
        self.save_inference_video = args.save_inference
        self.enable_ssta_agents=args.enable_ssta_agents
        self.display_mertic=args.display_metric
        self.do_inference = args.do_inference
        self.display_realistic=args.display_realistic
        self.memory_length = args.threshold_time_step_gt
        
        #running the model and visualising the T2NO results
        if self.do_inference or self.save_data:
            self.predictor = SSTA_predictor(args)
            if self.save_inference_video:
                self.inference_saver = cv2.VideoWriter('inference.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 60, (520, 390)) 
                self.inference_duration = tuple(args.inference_duration)
        if self.save_video:
            self.video_saver = cv2.VideoWriter('full_sim.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 60, (args.window_width, args.window_height)) 
            self.duration = tuple(args.video_duration)

        # Set up car and goal positions
        self.obstacles = args.obstacles
        self.clock = pygame.time.Clock()
        self.frame_rate= args.frame_rate
        self.infinity = LoopSimulation(args.window_height,args.window_width,100,args.seed)


        self.reset_index_global_path_number_ssta=args.global_path_intermediate_points

        self.apf_ssta_agents=APFSSTAAgents(args.obstacles,DoubleIntegratorAPF,DoubleIntegratorSSTA,self.frame_rate,self.infinity,self.reset_index_global_path_number_ssta)
        self.apf_ssta_agents.enable_ssta_agents=self.enable_ssta_agents


        ###
        # self.apf_ssta_agents.ssta_control.path_size=self.path_size
        self.replanning_time_interval=args.replanning_time_interval
        # self.apf_ssta_agents.ssta_control.replanning_index=self.replanning_index

        ###to be completed
        self.path_planner=Planners(self.replanning_time_interval)

        
        self.timer=0
        self.flag=True
        # setting the number of views/segment(default 2 view)
        self.ssta_boxes = args.ssta_boxes
        # each side of box/view/segment 
        self.side_length = self.ssta_boxes[:,-1]

    def run_simulation(self):
        print('running')
        # Main simulation loop
        running = True
        while running:

            #checks if window is closed and closes the loop by setting False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update/Obtain APF and SSTA car position
            self.apf_ssta_agents.generate_agents(self.timer)
            self.apf_car_pos,self.apf_goal_pos =  self.apf_ssta_agents.generate_apf_agents()
            if self.enable_ssta_agents:
                self.ssta_car_pos,self.ssta_goal_pos = self.apf_ssta_agents.generate_ssta_agents()
            #concatenating apf and ssta agents
            if self.enable_ssta_agents:
                self.car_pos=np.vstack([self.apf_car_pos,self.ssta_car_pos])
                self.goal_pos=np.vstack([self.apf_goal_pos,self.ssta_goal_pos[:,:2]])
            else:
                self.car_pos=self.apf_car_pos
                self.goal_pos=self.apf_goal_pos


            #setting the agent poses in the environment for displaying
            self.update_poses(self.car_pos, self.goal_pos)

            # setting the number of views/segment
            self.frame_angle,centers,(t_l,t_r,b_l,b_r)=self.segment_frame(self.ssta_boxes)
            ##############################
            ####       TO REMEMBER: ssta AGENTS NEED t2NO/D FOR PLANNING
            ##############################

            ##################### get predictions and visualise#######################
            if self.save_data or self.do_inference:
                inputs = self.get_frame(self.side_length,(t_l,t_r,b_l,b_r))
                self.predictor.update(np.array(inputs), self.timer)
                t2no, t2nd, prev_inputs, curr_inputs, outputs = self.predictor.get_params()

            if self.do_inference :
                vis = self.predictor.visualize(outputs, curr_inputs, t2no, t2nd)

                if self.save_inference_video and self.inference_duration[0] < self.timer and self.inference_duration[1] >= self.timer:
                    print(f"Saving {self.timer}")
                    self.inference_saver.write(np.uint8(vis*255))

                    if self.timer == self.inference_duration[1]:
                        self.inference_saver.release()
                        cv2.destroyAllWindows() 
                        print("Saved Inference")
                        quit()

            ############################################
            # print(self.apf_ssta_agents.ssta_goal_pos)
            #intersection for visualisation
            if self.enable_ssta_agents and len(self.ssta_car_pos)>0:
                self.intersections_apf=self.apf_ssta_agents.apf_control.intersection()
                self.intersections_ssta=self.apf_ssta_agents.ssta_control.intersection()
                self.intersections=np.hstack([self.intersections_apf,self.intersections_ssta])
                #takes the collision flags of apf and ssta but not considering all agents they are independant
                self.colllison_apf_ssta=np.hstack([self.apf_ssta_agents.apf_control.agent_collision,self.apf_ssta_agents.ssta_control.agent_collision])
                            
                #local points of all self.ssta.curpos
                global_cur_points_ssta=self.ssta_car_pos[:,:2]

                #first converting all the global points to local points
                local_cur_points_ssta = self.global_local_transform(global_cur_points_ssta,t_l,self.frame_angle)
                # print("local",local_cur_points_ssta.shape)

                #ssta agents local and global points
                local_cur_points, global_cur_points,global_goal_points,camera_points_indices=self.camera_agents(local_cur_points_ssta,self.side_length, self.ssta_car_pos,self.ssta_goal_pos)

                ###########Check
                self.apf_ssta_agents.ssta_goal_pos,_,self.apf_ssta_agents.ssta_control.combined_camera_indices=self.global_local_goal(self.ssta_goal_pos,camera_points_indices,local_cur_points,global_cur_points,global_goal_points,(t_l,t_r,b_l,b_r),self.frame_angle)
                
                ####setting the index of the first path 
                # print(self.apf_ssta_agents.ssta_goal_pos)
                if not args.manual_path_plan_ssta:
                    #function is not yet complete this will be the ultimate paths
                    ######Yet to complete #################
                    #determine whether astar will return local or global path
                    local_curr_pos  = self.apf_ssta_agents.ssta_goal_pos[:,4:6]
                    local_goal_pos = self.apf_ssta_agents.ssta_goal_pos[:,2:4]
                    box_idx         = self.apf_ssta_agents.ssta_goal_pos[:,8]
                    global_paths=self.apf_ssta_agents.ssta_goal_pos[:,9]

                    # local_paths = self.path_planner.a_star(local_curr_pos, local_goal_pos, box_idx, global_paths, t2no, t2nd, self.memory_length)          
                    # self.apf_ssta_agents.ssta_control.global_agent_paths=self.transform_local_to_global_path_vectorized(local_paths,camera_points_indices,t_l,self.frame_angle,n=len(self.ssta_goal_pos),k=self.path_size)

                if args.manual_path_plan_ssta:
                    #[curr_x, curr_y, vel_x, vel_y, _, _, goal_x, goal_y, box_num]
                    ###This calculates and gives the global path directly
                    curr_global_pnts=self.apf_ssta_agents.ssta_car_pos[:,0:2]
                    global_frame_goal_pnts=self.apf_ssta_agents.ssta_goal_pos[:,6:8]
                    segment_numbers=self.apf_ssta_agents.ssta_goal_pos[:,8]
                    global_paths=self.apf_ssta_agents.ssta_goal_pos[:,9]
                    path_indices=self.apf_ssta_agents.ssta_goal_pos[:,10]
                    global_start_times=self.apf_ssta_agents.ssta_goal_pos[:,11]
                    global_timer=self.timer/args.frame_rate #converting to seconds

                    # global_paths=self.path_planner.compute_global_paths(curr_global_pnts,global_frame_goal_pnts,segment_numbers,global_paths,self.ssta_boxes,self.reset_index_global_path_number_ssta)
                    
                    global_paths,path_indices,global_start_times=self.path_planner.compute_glbl_pth_rndm_lngth(curr_global_pnts,global_frame_goal_pnts,segment_numbers,global_paths,self.ssta_boxes,path_indices,global_start_times,global_timer)
                
                    ##need to do apth index here also now since replanning ##TO DO ####
                    self.apf_ssta_agents.ssta_goal_pos[:,9]=global_paths
                    self.apf_ssta_agents.ssta_goal_pos[:,10]=path_indices
                    self.apf_ssta_agents.ssta_goal_pos[:,11]=global_start_times

            else:
                #without SSTA
                self.intersections=self.apf_ssta_agents.apf_control.intersection()
                self.colllison_apf_ssta=self.apf_ssta_agents.apf_control.agent_collision

            
            # print(self.apf_ssta_agents.ssta_goal_pos)
            self.draw_map() # draws map with obstacles 
            self.draw_agents_with_goals(self.colllison_apf_ssta) # draws agents and their respective goal positions
            # plotting the segment
            self.plot_segment_frame(centers,(t_l,t_r,b_l,b_r))

            ### Plotting of global ssta paths after calculation
            if self.debugging and  len(self.ssta_car_pos)>0:
                self.plot_global_path_ssta(global_paths)


            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            ##########METRICS############################
            if self.timer%50==0:
                average_speed=self.apf_ssta_agents.traffic_speed()
                collision_rate,total_time=self.apf_ssta_agents.collison_rate()
                volume,capacity=self.apf_ssta_agents.volume_capacity()
            if self.display_mertic:
                self.display_collision_rate(collision_rate)
                self.display_total_time(total_time)
                self.display_v_c_ratio(volume,capacity)
                self.display_traffic_speed(average_speed) 
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

            ########## Saving the images on views and csv file
            ##need to complete the t2no image saving
            if self.save_data:
                self.save_camera_image(len(self.args.ssta_boxes), self.timer, 100000, 10000, 0, t2no, t2nd, prev_inputs,  500)#side_length,square dimensions,timer,train,test,val,gap(buffer)
                # saving camera csv file (TO DOOOOOOO)
                # self.save_camera_data(self.timer,camera_x_local,camera_x_global)
            if self.save_video and self.duration[0] < self.timer and self.duration[1] >= self.timer:
                frame = pygame.surfarray.array3d(self.screen)
                frame = np.transpose(frame, (1, 0, 2))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                self.video_saver.write(frame)

                if self.timer == self.duration[1]:
                    self.video_saver.release()
                    cv2.destroyAllWindows() 
                    print("Saved Video")

            ############################################
            
            pygame.display.update()
            self.clock.tick(self.frame_rate)
            self.timer+=1

        pygame.quit()

if __name__ == "__main__":
    simulation = CarSimulation(config)
    simulation.run_simulation()