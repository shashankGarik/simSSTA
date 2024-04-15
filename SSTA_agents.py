import pygame
import numpy as np
from Environment import Environment

class SSTAAgents():
    def __init__(self,obstacles,controller,frame_rate,infinity):
        # self.car_pos = np.array([[-20.0, 300.0, 0.0, 0.0, 6, 15, -1],[-10.0, 50.0, 0.0, 0.0,6, 15, -1],[-10.0, 80.0, 0.0, 0.0,6, 15, -1]])# startx,starty,vx,vy,colour,radius,shape(agent)
        # # goal = np.array([[800, 1500],[700, 1600]])  # global_goal_x,global_goal_y,local_goal_x(intersection_x),local_goal_y(intersection_y)
        # self.goal_pos = np.array([[1500, 600,None,None,None,None,None],[1500, 400,None,None,None,None,None],[1500, 500,None,None,None,None,None]]) # globalgx,globalgy,goalviewlocalgx,goalviewlocalgy,currlocalviewx,currlocalviewy,view/segment
        self.car_pos = np.array([[-10.0, 100.0, 0.0, 0.0,6, 15, -1]])# startx,starty,vx,vy,colour,radius,shape(agent)
        # goal = np.array([[800, 1500],[700, 1600]])  # global_goal_x,global_goal,y,local_goal_x(intersection_x),local_goal_y(intersection_y)
        self.goal_pos = np.array([[1000, 150,np.nan,np.nan,np.nan,np.nan,np.nan]]) 
        # self.car_pos = np.array([[-10.0, 300.0, 0.0, 0.0,6, 15, -1],[-50.0, 0.0, 0.0, 0.0,6, 15, -1]])# startx,starty,vx,vy,colour,radius,shape(agent)
        # # goal = np.array([[800, 1500],[700, 1600]])  # global_goal_x,global_goal,y,local_goal_x(intersection_x),local_goal_y(intersection_y)
        # self.goal_pos = np.array([[1000, 100,None,None,None,None,None],[200, 800,None,None,None,None,None]]) 
        self.obstacles = obstacles
        self.controller = controller
        self.infinity = infinity
        self.control = self.controller(self.car_pos, self.goal_pos, self.obstacles)
        self.control.dt = 1/frame_rate
        self.random_agent_generate_time=1000
        self.car_pos = self.control.x
        self.goal_pos = self.control.goal_pos

    def generate_agents(self,timer):
        # Update car position
        #uncomment if you want to generate random SSTA agents# ((((function needs to be complete))))
        # if timer%self.random_agent_generate_time==0:
        #     new_agents=self.infinity.run_simulation(self.car_pos,(self.goal_pos[:,:2]).astype(np.int32))# important that goal points passed in must be global and of int 32
        #     self.control.create_agents(new_agents)
        #     self.car_pos = self.control.x
        #     self.goal_pos = self.control.goal_pos

        #################
        # control switch check here to switch  between controller
        #################
        self.car_pos,self.goal_pos = self.control.car_pos()
        return  self.car_pos,self.goal_pos

    def intersections(self):
        return self.control.intersection()


            

            

