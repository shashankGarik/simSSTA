import pygame
import numpy as np
from controllers_APF import *
from controllers_SSTA import *
# from test_cases import *
from loop_agents import *
from Environment import Environment
from APF_SSTA_agents import *
from Planners.path_planners import *

class CarSimulation(Environment):
    def __init__(self, obstacle_vec):
        super().__init__(800, 800, obstacle_vec)
        # Initialize Pygame necessary for initialising the simulation window and graphics
        print('Initializing Agents')
        pygame.init()
        pygame.display.set_caption("Car Simulation")#Windows heading

        self.debugging = True
        self.save_data = False
        self.enable_ssta_agents=True  
        self.display_mertic=True


        # Set up car and goal positions
        self.car_pos = None
        self.goal_pos = None
        self.obstacles = obstacle_vec
        self.clock = pygame.time.Clock()
        self.frame_rate= 60
        self.infinity = LoopSimulation(800,800,120,42)

        self.apf_ssta_agents=APFSSTAAgents(obstacle_vec,DoubleIntegratorAPF,DoubleIntegratorSSTA,self.frame_rate,self.infinity)
        self.apf_ssta_agents.enable_ssta_agents=self.enable_ssta_agents

        self.path_size=21
        self.replanning_index=5
        self.apf_ssta_agents.ssta_control.path_size=self.path_size
        self.apf_ssta_agents.ssta_control.replanning_index=self.replanning_index


        self.timer=0
        # setting the number of views/segment(default 2 view)
        self.twin_boxes = np.array([[30,450,50,250],[-30,50,400,250]])#angle,x,y,size
        # each side of box/view/segment 
        self.side_length = self.twin_boxes[:,-1]
        self.path_planner=Planners(self.path_size,self.replanning_index)
     
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

 
            self.update_poses(self.car_pos, self.goal_pos)
            # setting the number of views/segment
            self.frame_angle,centers,(t_l,t_r,b_l,b_r)=self.segment_frame(self.twin_boxes)
    
            #intersection for visulaisation
            if self.enable_ssta_agents:
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
                
                # self.ssta_path_indices=self.apf_ssta_agents.ssta_control.path_indices
                
                ##returns local path
                #return as view,n,m,2
                # local_path_test=self.path_planner.a_star( self.apf_ssta_agents.ssta_goal_pos,camera_points_indices, self.timer,self.ssta_path_indices)
                
                # self.apf_ssta_agents.ssta_control.global_agent_paths=self.transform_local_to_global_path_vectorized(local_path_test,camera_points_indices,t_l,self.frame_angle,n=len(self.ssta_goal_pos),k=self.path_size)
                # self.apf_ssta_agents.ssta_control.global_agent_paths=np.full(local_cur_points[0].shape,None)
                # print("global", self.ssta_agents.control.global_agent_paths)
                # print("camera", self.ssta_agents.control.combined_camera_indices)

            else:
                #without SSTA
                self.intersections=self.apf_ssta_agents.apf_control.intersection()
                self.colllison_apf_ssta=self.apf_ssta_agents.apf_control.agent_collision
            
            # quit()
            self.draw_map() # draws map with obstacles
            
            self.draw_agents_with_goals(self.colllison_apf_ssta) # draws agents and their respective goal positions
            # plotting the segment
            self.plot_segment_frame(centers,(t_l,t_r,b_l,b_r))
           

            #the global path is stored as a list to access
            # print(self.ssta_agents.control.global_agent_paths)
            # if self.enable_ssta_agents:
            #     # plot intersections that is local goal if self.debugging
            #     # self.test_intersection_local_goal(intersections_views_global_points)
            #     if len(self.ssta_car_pos)!=0 and self.apf_ssta_agents.ssta_control.global_agent_paths[0][0][0]!=None:
            #         for point_set in self.apf_ssta_agents.apf_control.global_agent_paths[0]:
            #             # for x, y in np.array((point_set)):
            #             #     print(x,y)
            #             pygame.draw.circle(self.screen, self.colors['lgreen'], (point_set[0], point_set[1]), 3)
         
        
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
            if self.save_data:
                #getting the agents in the frame
                # print(camera_x_local,len(camera_x_local))
                #saving camera1 dataset
                self.save_camera_image(self.side_length,(t_l,t_r,b_l,b_r),self.timer, 6000, 0, 0, 0)#side_length,square dimensions,timer,train,test,val,gap(buffer)
                # saving camera csv file (TO DOOOOOOO)
                # self.save_camera_data(self.timer,camera_x_local,camera_x_global)

            ############################################
            
            pygame.display.update()
            self.clock.tick(self.frame_rate)
            self.timer+=1
        pygame.quit()

if __name__ == "__main__":
    obstacles = {'circle': np.array([]),
        'rectangle': np.array([[500,500,30,30],
                                [700,700,30,30],
                                [300,300,30,30],
                                [700,300,30,30],
                                [300,700,30,30],
                                [300,500,30,30],
                                [500,300,30,30],
                                [700,500,30,30],
                                [500,700,30,30]])}
    simulation = CarSimulation(obstacles)
    simulation.run_simulation()