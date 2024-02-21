import pygame
import numpy as np
from controllers import *
# from test_cases import *
from loop_agents import *
# from loop_agents import LoopSimulation
from Environment import Environment

class APFAgents():
    def __init__(self,obstacles,controller,frame_rate,infinity):
        self.car_pos = np.array([[-50.0, 300.0, 0.0, 0.0, 0, 15, -1],[-30.0, 50.0, 0.0, 0.0,0, 15, -1]])# startx,starty,vx,vy,colour,radius,shape(agent)
        # goal = np.array([[800, 1500],[700, 1600]])  # global_goal_x,global_goal,y,local_goal_x(intersection_x),local_goal_y(intersection_y)
        self.goal_pos = np.array([[800, 1500],[700, 1600]]) # the new one is an object
        
        self.obstacles = obstacles
        self.controller = controller
        self.infinity = infinity
        self.control = self.controller(self.car_pos, self.goal_pos, self.obstacles)
        self.control.dt = 1/frame_rate
        self.random_agent_generate_time=100

    def generate_agents(self,timer):
        # Update car position
        if timer%self.random_agent_generate_time==0:
            new_agents=self.infinity.run_simulation(self.car_pos,(self.goal_pos[:,:2]).astype(np.int32))# important that goal points passed in must be global and of int 32
            self.control.create_agents(new_agents)
            self.car_pos = self.control.x
            self.goal_pos = self.control.goal_pos
        self.car_pos,self.goal_pos = self.control.car_pos()
        return  self.car_pos,self.goal_pos

    def intersections(self):
        return self.control.intersection()


            

            

