import pygame
import random
import numpy as np
from controllers_APF import *
from basic_agent import *
from itertools import permutations

class LoopSimulation:
    def __init__(self, frame_h, frame_w, dist2GE_min,seed = None):
        """
        Initializes simulation that ends after some time period

        Args:
            frame_h (int): height of visible frame. Agents will spawn beyond this.
            frame_w (int): width of visible frame. Agents will spawn beyond this.
            dist2GE_min (int or float): min distance from start/goal pos to edge of frame.
            dist2GE_max (int or float):  maxdistance from start/goal pos to edge of frame.
            obs (dict): obstacle vector with format {"circle": [[x1,y1]], [...]],
                                                     "rectangle": [[x1,y1,w1,h1], [...]]}
        """

        self.frame_h = frame_h
        self.frame_w = frame_w
        self.dist2GE_min = 500
        self.spawing_radius=20
        self.total_random_agents=100
        
        self.agents = {"start" : [], "goal" : []}
        if not seed == None:
            np.random.seed(seed)
            random.seed(seed)


    def run_simulation(self,old_agents_start=None,old_agents_goal=None):#(takes in only global goal not local goal)
        self.old_agents_start=old_agents_start
        self.old_agents_goal=old_agents_goal
        start_points,goal_points=self.create_agents(1, self.total_random_agents)
        new_agents = {"start": start_points, "goal" :goal_points}
        return new_agents
    
    def probability_agent_spawning(self,probability, size=1):
        """
    Generate random binary values (1 or 0) based on a given probability.

    Parameters:
    - probability: Probability of generating 1 (should be between 0 and 1).
    - size: Number of random values to generate (default is 1).

    Returns:
    - numpy array of generated binary values.
    """
        random_values = np.random.rand(size)  # Generate random values between 0 and 1
        binary_values = (random_values < probability).astype(int)  # Convert to 1 or 0 based on the probability
        indices_ones = np.where(binary_values == 1)
        return len(indices_ones[0])
    
    def bool_check_distance_spawn_point(self,distance_matrix,distance_threshold=25):
        #This function takes the matrix and returns the bool of the points within the threshold
        a_g_dist = np.squeeze(distance_matrix, axis=-1)
        a_g_dist = np.transpose(a_g_dist, ( 1, 0))
        a_g_dist_bool= np.any(a_g_dist < distance_threshold, axis=0)
        return a_g_dist_bool
    
    def check_spawning_overlap(self,start_points,goal_points):
        #This core function checks if the new points overlap with any points of start and goal
        if  self.old_agents_start is not None   and self.old_agents_goal is not None:
            start_start_diff = start_points[:,:2][:,np.newaxis] - self.old_agents_start[:,:2][np.newaxis,:,:]
            start_start_dist = np.linalg.norm(start_start_diff, axis = 2)[:,:,np.newaxis] + (1e-6)
            start_goal_diff = start_points[:,:2][:,np.newaxis] - self.old_agents_goal[np.newaxis,:,:]
            start_goal_dist = np.linalg.norm(start_goal_diff, axis = 2)[:,:,np.newaxis] + (1e-6)
            new_start_new_start_diff=start_points[:,:2][:,np.newaxis] - start_points[:,:2][np.newaxis,:,:]
            new_start_new_start_dist = np.linalg.norm(new_start_new_start_diff, axis = 2)[:,:,np.newaxis] + (1e-6)

            new_start_new_start_dist = np.squeeze(new_start_new_start_dist, axis=-1)
            np.fill_diagonal(new_start_new_start_dist, np.inf)
            new_start_new_start_dist=new_start_new_start_dist[:,:,np.newaxis]

            new_start_new_goal_diff=start_points[:,:2][:,np.newaxis] - goal_points[np.newaxis,:,:]
            new_start_new_goal_dist = np.linalg.norm(new_start_new_goal_diff, axis = 2)[:,:,np.newaxis] + (1e-6)
            new_goal_new_goal_diff=goal_points[:,np.newaxis] - goal_points[np.newaxis,:,:]
            new_goal_new_goal_dist = np.linalg.norm(new_goal_new_goal_diff, axis = 2)[:,:,np.newaxis] + (1e-6)
            new_goal_new_goal_dist = np.squeeze(new_goal_new_goal_dist, axis=-1)
            np.fill_diagonal(new_goal_new_goal_dist, np.inf)
            new_goal_new_goal_dist=new_goal_new_goal_dist[:,:,np.newaxis]

            new_goal_new_start_diff=goal_points[np.newaxis,:,:]-start_points[:,:2][:,np.newaxis] 
            new_goal_new_start_dist = np.linalg.norm(new_goal_new_start_diff, axis = 2)[:,:,np.newaxis] + (1e-6)
            goal_start_diff =  goal_points[:,np.newaxis] -self.old_agents_start[:,:2][np.newaxis,:,:]
            goal_start_dist = np.linalg.norm(goal_start_diff, axis = 2)[:,:,np.newaxis] + (1e-6)
            goal_goal_diff =   goal_points[:,np.newaxis]-self.old_agents_goal[np.newaxis,:,:]
            goal_goal_dist = np.linalg.norm(goal_goal_diff, axis = 2)[:,:,np.newaxis] + (1e-6)

            s_s_bool=self.bool_check_distance_spawn_point(start_start_dist)
            s_g_bool=self.bool_check_distance_spawn_point(start_goal_dist)
            g_s_bool=self.bool_check_distance_spawn_point(goal_start_dist)
            g_g_bool=self.bool_check_distance_spawn_point(goal_goal_dist)
            new_s_s_bool=self.bool_check_distance_spawn_point(new_start_new_start_dist)
            new_s_g_bool=self.bool_check_distance_spawn_point(new_start_new_goal_dist)
            new_g_g_bool=self.bool_check_distance_spawn_point(new_goal_new_goal_dist)
            new_g_s_bool=self.bool_check_distance_spawn_point(new_goal_new_start_dist)

            validate_points= np.logical_or.reduce([s_s_bool,s_g_bool,g_s_bool,g_g_bool,new_s_g_bool,new_s_s_bool,new_g_g_bool,new_g_s_bool])
            validate_points_collide_indices=np.where(validate_points)
            mask = np.ones(start_points.shape[0], dtype=bool)
            mask[validate_points_collide_indices] = False
            start_points=start_points[mask]
            goal_points=goal_points[mask]

        return start_points,goal_points,len(start_points)

    def create_agents(self, p, n):
        """
        creates start and goal points for agents to spawn in

        Args:
            p (float): probability with which to spawn agent(s).
            n (int): number of agents to spawn with probability p (See max_agents_flag).
            max_agents_flag (bool): if True n is the max number of agents to spawn. else
                                    n in the total number of agents spawned with the 
                                    probability p 

        Returns:
            start (np.ndarray): start vector for agents.
            goal (np.ndarray)
        """
        n=self.probability_agent_spawning(p,n)
        sample_n=n*2
        # start_points_res,goal_points_res=np.empty((0, 4)),np.empty((0, 2))
        while(n!=0):
            
            AgentR = ["R","L","U","D"]
            pairs_permutations = list(permutations(AgentR, 2))
            start_goal_random= np.array(random.choices(pairs_permutations, k=sample_n))

            start_random_var = start_goal_random[:, 0]
            goal_random_var = start_goal_random[:, 1]

            start_points = np.zeros((sample_n,4), dtype=float)
            goal_points= np.zeros((sample_n,2), dtype=float)

            for case in AgentR:
                condition_start = np.where(start_random_var == case)[0]
                condition_goal = np.where(goal_random_var == case)[0]

                if case=="L":
                    x_start = np.random.sample(len(condition_start))*(self.dist2GE_min - self.spawing_radius) - self.dist2GE_min
                    y_start=np.random.sample(len(condition_start))*(self.frame_h+(self.dist2GE_min))- self.dist2GE_min
                    x_goal = np.random.sample(len(condition_goal))*(self.dist2GE_min - self.spawing_radius) - self.dist2GE_min
                    y_goal=np.random.sample(len(condition_goal))*(self.frame_h+(self.dist2GE_min))- self.dist2GE_min
                
                elif case=="U":
                    y_start = np.random.sample(len(condition_start))*(self.dist2GE_min - self.spawing_radius) - self.dist2GE_min
                    x_start=np.random.sample(len(condition_start))*(self.frame_h+(self.dist2GE_min))- self.dist2GE_min
                    y_goal = np.random.sample(len(condition_goal))*(self.dist2GE_min - self.spawing_radius) - self.dist2GE_min
                    x_goal=np.random.sample(len(condition_goal))*(self.frame_h+(self.dist2GE_min))- self.dist2GE_min
                
                elif case=="R":
                    x_start = np.random.sample(len(condition_start))*(self.dist2GE_min )+(self.frame_w+self.spawing_radius ) 
                    y_start= np.random.sample(len(condition_start))*(self.frame_h+(self.dist2GE_min))- self.dist2GE_min
                    x_goal = np.random.sample(len(condition_goal))*(self.dist2GE_min )+(self.frame_w+self.spawing_radius ) 
                    y_goal= np.random.sample(len(condition_goal))*(self.frame_h+(self.dist2GE_min))- self.dist2GE_min

                else:
                # case=="D":
                    y_start = np.random.sample(len(condition_start))*(self.dist2GE_min )+(self.frame_h+self.spawing_radius ) 
                    x_start = np.random.sample(len(condition_start))*(self.frame_w+(self.dist2GE_min))- self.dist2GE_min
                    y_goal = np.random.sample(len(condition_goal))*(self.dist2GE_min )+(self.frame_h+self.spawing_radius ) 
                    x_goal = np.random.sample(len(condition_goal))*(self.frame_w+(self.dist2GE_min))- self.dist2GE_min
                    
                start_points[condition_start,0]=x_start
                start_points[condition_start,1]=y_start
                goal_points[condition_goal,0]=x_goal
                goal_points[condition_goal,1]=y_goal

            color_vec = np.random.randint(0,6,size=(start_points.shape[0],1))

            radius_choice = [10, 13, 15, 17, 20]
            radius_prob = [0.5, 0.175, 0.125, 0.10, 0.10]         

            radius_vec = np.random.choice(radius_choice, p = radius_prob, size=(start_points.shape[0],1))

            shape_choice = [3,5,6,7,8]
            polygon_vec = np.random.choice(shape_choice, size=(start_points.shape[0],1))

            start_points = np.hstack([start_points,color_vec, radius_vec, polygon_vec])

            print("ahm",start_points.shape,goal_points.shape,"sample",sample_n,"n",n)
        
            start_points,goal_points,sample_n=self.check_spawning_overlap(start_points,goal_points)
            
            print("ahm",start_points.shape,goal_points.shape,"sample",sample_n,"n",n)
            if sample_n>n:
                start_points=start_points[:n]
                goal_points=goal_points[:n]
            n=0
            # else:
            #     n=sample_n=n-sample_n
            #     start_points_res,goal_points_res=np.vstack([start_points_res,start_points]),np.vstack([goal_points_res,goal_points])
            #     self.old_agents_start,self.old_agents_goal=np.vstack([self.old_agents_start,start_points]),np.vstack([self.old_agents_goal,goal_points])
            # print(n,sample_n)
        return start_points,goal_points
    