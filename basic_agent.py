import pygame
import numpy as np
from controllers import *
# from test_cases import *
from loop_agents import *
from Environment import Environment

class CarSimulation(Environment):
    def __init__(self, start_vec, goal_vec, obstacle_vec, controller):
        super().__init__(800, 800, obstacle_vec)
        # Initialize Pygame necessary for initialising the simulation window and graphics
        print('Initializing Agents')
        pygame.init()

        pygame.display.set_caption("Car Simulation")#Windows heading

        self.debugging = False
        self.save_data = False

        # Set up car and goal positions
        self.car_pos = start_vec
        self.goal_pos = goal_vec
        # self.update_poses(start_vec, goal_vec)
        self.obstacles = obstacle_vec
        self.controller = controller
        self.clock = pygame.time.Clock()
        self.frame_rate= 60

        self.control = self.controller(self.car_pos, self.goal_pos, self.obstacles)
        self.control.dt = 1/self.frame_rate
        self.infinity = LoopSimulation(800,800,120,1,1,42)
        
        # self.control.create_agents(new_agents)
        self.timer=0
        # self.car_pos = self.control.x
        # self.goal_pos = self.control.goal_pos
    

    def run_simulation(self):
        print('running')
        # Main simulation loop
        running = True
        while running:
            #checks if window is closed and closes the loop by setting False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

           # Update car position
            if self.timer%100==0:
                new_agents=self.infinity.run_simulation(self.car_pos,(self.goal_pos[:,:2]).astype(np.int32))# important that goal points passed in must be global and of int 32
                self.control.create_agents(new_agents)
                self.car_pos = self.control.x
                self.goal_pos = self.control.goal_pos
            

            # print(self.car_pos)
            self.car_pos,self.goal_pos = self.control.car_pos()
            self.update_poses(self.car_pos, self.goal_pos)
            self.intersections=self.control.intersection()
            self.draw_map() # draws map with obstacles
            
            self.draw_agents_with_goals(self.control.agent_collision) # draws agents and their respective goal positions
            speed=self.control.traffic_speed()
            collision_rate,total_time=self.control.collison_rate()
            capacity,volume=self.control.volume_capacity()
           
            #metrics
            if self.debugging == True:
                self.display_collision_rate(collision_rate)
                self.display_total_time(total_time)
                self.display_v_c_ratio(volume,capacity)
                self.display_traffic_speed(speed)

            # setting the segment
            twin_boxes = np.array([[-60,300,400,300],[-60,450,50,300]])
            self.frame_angle,centers,(t_l,t_r,b_l,b_r)=self.segment_frame(twin_boxes)
            side_length = twin_boxes[:,-1]
            # plotting the segment
            #local points of all self.x
            global_points=self.control.x[:,:2]
            local_points = self.global_local_transform(global_points,t_l,self.frame_angle)
            camera_x_local,camera_x_global=self.camera_agents(local_points,side_length, self.control.x) #####uncomment

            
            # if self.debugging:
            self.plot_segment_frame(centers,(t_l,t_r,b_l,b_r))

            if self.save_data:
                
                #getting the agents in the frame
                # print(camera_x_local,len(camera_x_local))
                #saving camera1 dataset
                self.save_camera_image(side_length,(t_l,t_r,b_l,b_r),self.timer, 6000, 1500, 1500, 500)
                #saving camera csv file
                # self.save_camera_data(self.timer,camera_x_local,camera_x_global)

            pygame.display.update()
            self.clock.tick(self.frame_rate)
            self.timer+=1

            
        pygame.quit()

if __name__ == "__main__":
   
    start = np.array([[-50.0, 300.0, 0.0, 0.0, 1, 15, -1],[-30.0, 50.0, 0.0, 0.0,1, 15, -1]])# startx,starty,vx,vy,colour,radius,type(agent)
    #the agent type 0 indicates APF and 1 indicates SSTA
    # goal = np.array([[800, 1500],[700, 1600]])  # global_goal_x,global_goal,y,local_goal_x(intersection_x),local_goal_y(intersection_y)
    goal = np.array([[800, 1500,None,None],[700, 1600,None,None]]) # the new one is an object
  
    obs = {'circle': np.array([]),
       'rectangle': np.array([[500,500,30,30],
                              [700,700,30,30],
                              [300,300,30,30],
                              [700,300,30,30],
                              [300,700,30,30],
                              [300,500,30,30],
                              [500,300,30,30],
                              [700,500,30,30],
                              [500,700,30,30]])}


    simulation = CarSimulation(start, goal, obs, DoubleIntegrator)
    simulation.run_simulation()
