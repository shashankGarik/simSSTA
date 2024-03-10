import pygame
import numpy as np
from controllers_APF import *
from controllers_SSTA import *
# from test_cases import *
from loop_agents import *
from Environment import Environment
from APF_agents import *
from SSTA_agents import *
from path_planners import *

class CarSimulation(Environment):
    def __init__(self, obstacle_vec):
        super().__init__(800, 800, obstacle_vec)
        # Initialize Pygame necessary for initialising the simulation window and graphics
        print('Initializing Agents')
        pygame.init()
        pygame.display.set_caption("Car Simulation")#Windows heading

        self.debugging = True
        self.save_data = False

        # Set up car and goal positions
        self.car_pos = None
        self.goal_pos = None
        self.obstacles = obstacle_vec
        self.clock = pygame.time.Clock()
        self.frame_rate= 60
        self.infinity = LoopSimulation(800,800,120,1,1,42)
        self.apf_agents=APFAgents(obstacle_vec,DoubleIntegratorAPF,self.frame_rate,self.infinity)
        self.ssta_agents=SSTAAgents(obstacle_vec,DoubleIntegratorSSTA,self.frame_rate,self.infinity)
        self.path_size=5
        self.ssta_agents.control.path_size=self.path_size

        self.timer=0
        # setting the number of views/segment(default 2 view)
        self.twin_boxes = np.array([[-60,450,50,300]])
        # each side of box/view/segment 
        self.side_length = self.twin_boxes[:,-1]
        self.path_planner=Planners()

        
    

    def run_simulation(self):
        print('running')
        # Main simulation loop
        running = True
        while running:
            #checks if window is closed and closes the loop by setting False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update/Obtain APF car position
            self.apf_car_pos,self.apf_goal_pos = self.apf_agents.generate_agents(self.timer)
            self.ssta_car_pos,self.ssta_goal_pos = self.ssta_agents.generate_agents(self.timer)

            #concatenating apf and ssta agents
            self.car_pos=np.vstack([self.apf_car_pos,self.ssta_car_pos])
            self.goal_pos=np.vstack([self.apf_goal_pos,self.ssta_goal_pos[:,:2]])

            self.update_poses(self.car_pos, self.goal_pos)
            

            #intersection for visulaisation
            self.intersections_apf=self.apf_agents.control.intersection()
            self.intersections_ssta=self.ssta_agents.control.intersection()
            self.intersections=np.hstack([self.intersections_apf,self.intersections_ssta])


            self.draw_map() # draws map with obstacles
            #takes the collision flags of apf and ssta but not considering all agents they are independant
            self.colllison_apf_ssta=np.hstack([self.apf_agents.control.agent_collision,self.ssta_agents.control.agent_collision])
            

            self.draw_agents_with_goals(self.colllison_apf_ssta) # draws agents and their respective goal positions
            
            # setting the number of views/segment
            self.frame_angle,centers,(t_l,t_r,b_l,b_r)=self.segment_frame(self.twin_boxes)
            # plotting the segment
            self.plot_segment_frame(centers,(t_l,t_r,b_l,b_r))
            
            
            #local points of all self.ssta.curpos
            global_cur_points_ssta=self.ssta_car_pos[:,:2]


            #first converting all the global points to local points
   
            local_cur_points_ssta = self.global_local_transform(global_cur_points_ssta,t_l,self.frame_angle)
            # print(local_cur_points_ssta)
            #ssta agents local and global points
            local_cur_points, global_cur_points,global_goal_points,camera_points_indices=self.camera_agents(local_cur_points_ssta,self.side_length, self.ssta_car_pos,self.ssta_goal_pos)
            
            self.ssta_agents.goal_pos,intersections_views_global_points=self.global_local_goal(self.ssta_goal_pos,camera_points_indices,local_cur_points,global_cur_points,global_goal_points,(t_l,t_r,b_l,b_r),self.frame_angle)
            
            # plot intersections that is local goal if self.debugging
            self.test_intersection_local_goal(intersections_views_global_points)

            

            # print(self.ssta_car_pos)
            # print("-------")
            # print( self.ssta_agents.goal_pos)
            # print(camera_points_indices)
            # print("-------")

            # put in planner functin here
            #takes in ssta agents and returns the path
            self.ssta_path_indices=self.ssta_agents.control.path_indices
            print(self.ssta_path_indices)
            local_path_test=self.path_planner.a_star( self.ssta_agents.goal_pos,self.path_size)
            
            

            #assume you get a path from T2no file
            #path shape is n,(m,2)  -n -no of views,m-no.of agents
            self.ssta_agents.control.global_agent_paths=self.transform_local_to_global_path_vectorized(local_path_test,t_l,self.frame_angle)[0]
            # print(self.ssta_agents.control.global_agent_paths)
            # #the global path is stored as a list to access
            # print(self.ssta_agents.control.global_agent_paths[0].shape)
            # for point_set in self.ssta_agents.control.global_agent_paths[1]:
            #     # print(point_set)
            #     # for x, y in np.array((point_set)):
            #     #     print(x,y)
            #     pygame.draw.circle(self.screen, self.colors['lgreen'], (point_set[0], point_set[1]), 3)
         
            
            ### now with the lists mapped use the indices to have a global path variable
            # creating path using T2NO (give local start and local goal# index 2,3-goalpoint,4,5-startcurr point-self.ssta_agents.goal_pos )
            # print(camera_points_indices)
            self.ssta_agents.control.camera_points_indices=camera_points_indices
            # convert the local_path_points to global_path_points:(create a function to convert local to global)
            # self.ssta_car_pos,self.ssta_goal_pos = self.ssta_agents.switch_controller()

            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            ##########metrics function need to be modified 

            # speed=self.ssta_agents.control.traffic_speed()
            # collision_rate,total_time=self.ssta_agents.control.collison_rate()
            # capacity,volume=self.ssta_agents.control.volume_capacity()
            # #metrics
            # if self.debugging == True:
            #     self.display_collision_rate(collision_rate)
            #     self.display_total_time(total_time)
            #     self.display_v_c_ratio(volume,capacity)
            #     self.display_traffic_speed(speed)
            ##########metrics function need to be modified 
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

            ## Saving the images on views
            if self.save_data:
                #getting the agents in the frame
                # print(camera_x_local,len(camera_x_local))
                #saving camera1 dataset
                self.save_camera_image(self.side_length,(t_l,t_r,b_l,b_r),self.timer, 10, 5, 5, 5)#side_length,square dimensions,timer,train,test,val,gap(buffer)
                # saving camera csv file (TO DOOOOOOO)
                # self.save_camera_data(self.timer,camera_x_local,camera_x_global)
          
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