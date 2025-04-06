from Planners.utils import *
from Planners.occupancy import OccupancyHelper
import numpy as np
import cv2
from Planners.localize import adust_local_t2nod

class Astar_T2nod_agentic:
    def __init__(self, curr_pos, t2no, t2nd, r = 15, g_f = 3, max_time_step = 50):
        """
        args
        - t2no
        - t2nd
        - r: radius of the agent
        - g_f: grid factor used in combination with the radius to determine grid size
        """
        self.curr_pos = curr_pos
        t2no, t2nd = adust_local_t2nod(curr_pos, t2no, t2nd, max_time_step)
        self.occupancy = OccupancyHelper(t2no, t2nd, r, g_f, max_time_step)
        
    def is_terminal(self, state, goal):
        """
        checks if state is terminal
        """
        # if  (goal[0] - self.w < state[0] < goal[0] + self.w) and (goal[1] - self.w < state[1] < goal[1] + self.w):
        #     return True
        if state == goal:
            return True
        return False

    def scale_points(self, points, dir = "in"):
        """
        scale in or out based on t2no scaling
        """
        points = np.array(points)

        if dir == "in":
            points = points//self.occupancy.scale
        elif dir == "out":
            points = points*self.occupancy.scale
        else: raise AttributeError("define scaling properly")

        return points.tolist()
    
    
    def run_search(self, goal, heuristic = manhattan_dist):
        start = self.curr_pos

        start, goal = self.scale_points([start, goal], dir = "in")
        start, goal = tuple(start), tuple(goal)
        
        frontier = PriorityQueue()
        visited = set()

        frontier.insert((start,[start],0,0),0) #(start_state, path, cost, time_step), priority

        while frontier.elements:
            (curr_state, curr_path, cost, t), _ = frontier.pop()

            if curr_state not in visited:
                if self.is_terminal(curr_state, goal): 
                    path = curr_path.copy()
                    path = self.scale_points(path, dir = "out")
                    return path
                
                visited.add(curr_state)
                neighbors = self.occupancy.get_grid_neighbors(curr_state)
                for n in neighbors:
                    temp_path = curr_path.copy()
                    temp_path.append(n)

                    state_cost = np.round(cost + self.occupancy.get_state_cost(n),4)
                    heuristic_cost = np.round(heuristic(n, goal), 3)

                    if n not in visited:
                        frontier.insert((n,temp_path, state_cost, t+1),state_cost + heuristic_cost)

                self.occupancy.iterate()
        return None            

class Astar_T2nod_general:
    def __init__(self, t2no, t2nd, args, max_time_step = 50, d = 5):
        self.t2no = t2no
        self.t2nd = t2nd
        self.max_time_step = max_time_step
        self.args = args
        self.w = d # how many pixels each neighbor is apart

    def compute_cost(self, neighbor, t):

        cost = 0

        if self.t2no[neighbor[0],neighbor[1]]-t <= 0:
            if self.t2nd[neighbor[0],neighbor[1]]-t <= 0:
                cost = 0
            else:
                cost = 1e3
        elif self.t2no[neighbor[0],neighbor[1]]-t == self.max_time_step-t:
            cost = 0
        else:
            cost = 1/(self.t2no[neighbor[0],neighbor[1]]-t)    

        return cost

    def get_neighbors(self, state, max_boundary = 128):
        w = self.w
        x,y = state
        p_neighbors = [(x-w,y+w),(x,y+w),(x+w,y+w),
                       (x-w,y),          (x+w,y),
                       (x-w,y-w),(x,y-w),(x+w,y-w)]
        
        r_c = 1
        d_c = r_c*(2**0.5)
        p_cost_multiplier = [d_c,r_c,d_c,
                             r_c,    r_c,
                             d_c,r_c,d_c]

        cost_multiplier = []
        neighbors = []  

        for idx, (x_p,y_p) in enumerate(p_neighbors):

            if (0 <= x_p <= max_boundary-1 and 0 <= y_p <= max_boundary-1): 
                # if map.T[x_p,y_p] != 1:
                #     neighbors.append((x_p,y_p))
                neighbors.append((x_p,y_p))
                cost_multiplier.append(p_cost_multiplier[idx])
                
        return neighbors, cost_multiplier
    
    def is_terminal(self, state, goal):
        """
        checks if state is terminal
        """
        if  (goal[0] - self.w < state[0] < goal[0] + self.w) and (goal[1] - self.w < state[1] < goal[1] + self.w):
            return True
        return False

    def run_search(self, start_state, goal_state, heuristic_cost = manhattan_dist):

        frontier = PriorityQueue()
        visited = set()
        frontier.insert((start_state,[start_state],0,0),0) #(start_state, path, cost, time_step), priority

        while frontier.elements:
            # print(frontier.elements)
            (curr_state, curr_path, cost, t), _ = frontier.pop()

            if curr_state not in visited:

                if self.is_terminal(curr_state, goal_state):
                    
                    path = curr_path.copy()
                    if curr_state != goal_state:
                        path.append(goal_state)
                    return path
                
                visited.add(curr_state)
                neighbors, cost_multiplier = self.get_neighbors(curr_state, self.args.img_width)
                # print(neighbors)
                for (n,c_m) in zip(neighbors, cost_multiplier):
                    temp_path = curr_path.copy()
                    temp_path.append(n)

                    ind_cost = self.compute_cost(n, t)
                    # print(ind_cost)
                    # ind_cost = (1 if self.map.T[n[0],n[1]] == 0 else self.map.T[n[0],n[1]])

                    temp_cost = cost + ind_cost*c_m
                    h_cost = heuristic_cost(n, goal_state) 

                    if n not in visited:
                        frontier.insert((n,temp_path, temp_cost, t + self.w),temp_cost + h_cost)

        return None
    
# if __name__ == '__main__':
    
#     ## sample implementation
#     t2no_path = "/home/sgarikipati7/packages/simSSTA/dataset/train/_MOG_t2no_50/camera_0/t2no_00001662.png"

#     t2nd_path = "/home/sgarikipati7/packages/simSSTA/dataset/train/_MOG_t2no_50/camera_0/t2nd_00001662.png"
    
#     t2no = cv2.imread(t2no_path, cv2.IMREAD_GRAYSCALE)
#     t2nd = cv2.imread(t2nd_path, cv2.IMREAD_GRAYSCALE)
#     check = Astar_T2nod(t2no.T, t2nd.T)
#     start,goal = (0,0), (127,127)
#     path = check.run_search(start,goal, euclidean_dist)
#     print(path)