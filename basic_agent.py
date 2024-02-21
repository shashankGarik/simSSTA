import pygame
import numpy as np
from controllers import *
# from test_cases import *
from loop_agents import *
from Environment import Environment
from APF_agents import *
from SSTA_agents import *

class CarSimulation(Environment):
    def __init__(self, obstacle_vec,controller):
        super().__init__(800, 800, obstacle_vec)
        # Initialize Pygame necessary for initialising the simulation window and graphics
        print('Initializing Agents')
        pygame.init()
        pygame.display.set_caption("Car Simulation")#Windows heading

        self.debugging = False
        self.save_data = False

        # Set up car and goal positions
        self.car_pos = None
        self.goal_pos = None
        self.obstacles = obstacle_vec
        self.controller = controller
        self.clock = pygame.time.Clock()
        self.frame_rate= 60
        self.infinity = LoopSimulation(800,800,120,1,1,42)
        self.apf_agents=APFAgents(obstacle_vec,controller,self.frame_rate,self.infinity)
        self.ssta_agents=SSTAAgents(obstacle_vec,controller,self.frame_rate,self.infinity)
        self.timer=0
    

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
            # self.intersections_ssta=self.ssta_agents.control.intersection()
            self.draw_map() # draws map with obstacles
            self.colllison_apf_ssta=np.hstack([self.apf_agents.control.agent_collision,self.ssta_agents.control.agent_collision])
            self.draw_agents_with_goals(self.colllison_apf_ssta) # draws agents and their respective goal positions
            
            # speed=self.ssta_agents.control.traffic_speed()
            # collision_rate,total_time=self.ssta_agents.control.collison_rate()
            # capacity,volume=self.ssta_agents.control.volume_capacity()
           
            # #metrics
            # if self.debugging == True:
            #     self.display_collision_rate(collision_rate)
            #     self.display_total_time(total_time)
            #     self.display_v_c_ratio(volume,capacity)
            #     self.display_traffic_speed(speed)

            # # setting the segment
            # twin_boxes = np.array([[-60,300,400,300],[-60,450,50,300]])
            # self.frame_angle,centers,(t_l,t_r,b_l,b_r)=self.segment_frame(twin_boxes)
            # side_length = twin_boxes[:,-1]
            # # plotting the segment
            # #local points of all self.x
            # global_points=self.apf_agents.control.x[:,:2]
            # local_points = self.global_local_transform(global_points,t_l,self.frame_angle)
            # camera_x_local,camera_x_global=self.camera_agents(local_points,side_length, self.ssta_agents.control.x) #####uncomment

            
            # # if self.debugging:
            # self.plot_segment_frame(centers,(t_l,t_r,b_l,b_r))

            # if self.save_data:
            #     #getting the agents in the frame
            #     # print(camera_x_local,len(camera_x_local))
            #     #saving camera1 dataset
            #     self.save_camera_image(side_length,(t_l,t_r,b_l,b_r),self.timer, 6000, 1500, 1500, 500)
            #     # saving camera csv file
            #     self.save_camera_data(self.timer,camera_x_local,camera_x_global)

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
    simulation = CarSimulation(obstacles, DoubleIntegrator)
    simulation.run_simulation()