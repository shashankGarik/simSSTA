# This file is the planner file and the return format must always be a set number of m values for the path,
#If n agents are there then the path shape must be view * (n,m,2)- numpy arrays in a list where list contains agents of views seperated
import numpy as np
from Planners.astar import *
from Planners.utils import *
import os
import cv2

class Planners():
    def __init__(self,path_size,replanning_index):
        #testing
        self.replanning_index=replanning_index
        self.path_size=path_size
        self.path=[np.full((path_size,2),-1)]
      

    def a_star(self,ssta_agents_goal_poses,ssta_camera_indices,time_step,ssta_path_indices):
  
        ssta_agents_start_poses=ssta_agents_goal_poses[:,4:6]

        # return as view,n,m,2 - this is as a list
        
        for idx, each in enumerate(ssta_agents_goal_poses):
            
            if each[-1] == None :
                continue
            if  ssta_path_indices[0]==self.replanning_index or self.path[0][0][0]==-1:
                count = str(time_step+1)
                count_filled = count.zfill(8)
                t2no_path = os.path.join(r'C:/Users/Welcome/Documents/Kouby/M.S.Robo- Georgia Tech/GATECH LABS/SHREYAS_LAB/Simulation_Environment/Github Simulation Network/dataset/train/_MOG_t2no_50','camera_'+str(each[-1]),'t2no_' + count_filled + '.png')
                t2nd_path = os.path.join(r'C:/Users/Welcome/Documents/Kouby/M.S.Robo- Georgia Tech/GATECH LABS/SHREYAS_LAB/Simulation_Environment/Github Simulation Network/dataset/train/_MOG_t2no_50','camera_'+str(each[-1]),'t2nd_' + count_filled + '.png')
                t2no = cv2.imread(t2no_path, cv2.IMREAD_GRAYSCALE)
                t2nd = cv2.imread(t2nd_path, cv2.IMREAD_GRAYSCALE)
                check = Astar_T2nod(t2no.T, t2nd.T)
                start = tuple(np.int16(ssta_agents_start_poses[0]*(128/300)))
                goal = tuple(np.int16(each[2:4]*(128/300)))
                path = check.run_search(start,goal, euclidean_dist)
                # print(start, goal, each[-1])
                if path is not None:
                    # print(path, each[-1])
                    # print(np.array(path))
                    self.path=(np.array([(path[:self.path_size])])/128)*300
                    # print(self.path)
                    print("inside",self.path)
                    return self.path
        
        return self.path




