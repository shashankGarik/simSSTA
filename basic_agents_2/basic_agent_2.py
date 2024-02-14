
import pygame
import numpy as np
from controllers_2 import *
from loop_agents_2 import *
from Environment_2 import Environment


class CarSimulation(Environment):
    def __init__(self, start_vec, goal_vec, obstacle_vec, controller):
        super().__init__(800, 800, obstacle_vec)
        # Initialize Pygame necessary for initialising the simulation window and graphics
        print('Initializing Agents')
        pygame.init()

        pygame.display.set_caption("Car Simulation")#Windows heading

        self.debugging = True
        self.save_data = False
        self.spawn_random_agents=True

        # Set up car and goal positions
        self.car_pos = start_vec
        self.goal_pos = goal_vec
        self.update_poses(start_vec, goal_vec)
        self.obstacles = obstacle_vec
        self.controller = controller
        self.clock = pygame.time.Clock()
        self.frame_rate= 80

        self.control = self.controller(self.car_pos, self.goal_pos, self.obstacles)
        self.control.dt = 1/self.frame_rate
        self.infinity = LoopSimulation(800,800,120,1,1)
        
        # self.control.create_agents(new_agents)
        self.timer=0
        self.car_pos = self.control.x
        self.goal_pos = self.control.goal_pos
    

    def run_simulation(self):
        print('running')
        # Main simulation loop
        running = True
        while running:
            #checks if window is closed and closes the loop by setting False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # print(len(self.car_pos))
            self.car_pos,self.goal_pos = self.control.car_pos()
            self.update_poses(self.car_pos, self.goal_pos)
            self.draw_map('white') # draws map with obstacles
            
            self.draw_agents_with_goals(self.control.agent_collision) # draws agents and their respective goal positions
            if self.timer%100==0 and self.spawn_random_agents:
                new_agents=self.infinity.run_simulation(self.car_pos,self.goal_pos)
                self.control.create_agents(new_agents)
                self.car_pos = self.control.x
                self.goal_pos = self.control.goal_pos

           
           

            # setting the segment
            twin_boxes = np.array([[0,50,100,600]])
            self.frame_angle,centers,(t_l,t_r,b_l,b_r)=self.segment_frame(twin_boxes)
            side_length = twin_boxes[:,-1]

            side_length = twin_boxes[:,-1]
            # plotting the segment
            #local points of all self.x
            global_points=self.control.x[:,:2]
            local_points = self.global_local_transform(global_points,t_l,self.frame_angle)


            local_agent_points, global_agent_points,global_goal_points=self.camera_agents(local_points,side_length, self.control.x,self.goal_pos)
          
           
            ######## Obtain local_goal from global_goal
            local_goal_points=self.global_local_goal(global_agent_points,global_goal_points,(t_l,t_r,b_l,b_r))
            print("ppppppppppppppppppppppppppppppppppppppp---",b_l,b_r)
            print("--------------------------------------local_goal_points",local_goal_points)
            self.test_intersection(local_goal_points)
            ###############
            
            if self.debugging:
                self.plot_segment_frame(centers,(t_l,t_r,b_l,b_r))


            pygame.display.update()
            self.clock.tick(self.frame_rate)
            self.timer+=1

            
        pygame.quit()

if __name__ == "__main__":
    np.random.seed(42)
   
    start = np.array([[100.0, 400.0, 0.0, 0.0,1],[120.0, 400.0, 0.0, 0.0,1],[-50.0, 400.0, 0.0, 0.0,1],[0.0, 700.0, 0.0, 0.0,2]])
    goal = np.array([[700, 100],[700,500],[700,400],[700,700]]) 
    # start = np.array([[100.0, 400.0, 0.0, 0.0,1]])
    # goal = np.array([[700, 100]])   
    obs = {'circle': np.array([]),
       'rectangle': np.array([])}


    simulation = CarSimulation(start, goal, obs, DoubleIntegrator)
    simulation.run_simulation()