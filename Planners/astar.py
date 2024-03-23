from Planners.utils import *
import numpy as np
import cv2

class Astar_T2nod:
    def __init__(self, t2no, t2nd, max_time_step = 50, d = 1):
        self.t2no = t2no
        self.t2nd = t2nd
        self.max_time_step = max_time_step
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

        # print(cost)

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
                neighbors, cost_multiplier = self.get_neighbors(curr_state)
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
#     t2no_path = 'C:/Users/shash/OneDrive/Desktop/SSTA_2/simSSTA/dataset/train/_MOG_t2no_120/camera_0/t2no_00002075.png'
#     t2nd_path = 'C:/Users/shash/OneDrive/Desktop/SSTA_2/simSSTA/dataset/train/_MOG_t2no_120/camera_0/t2nd_00002075.png'
    
#     t2no = cv2.imread(t2no_path, cv2.IMREAD_GRAYSCALE)
#     t2nd = cv2.imread(t2nd_path, cv2.IMREAD_GRAYSCALE)
#     check = Astar_T2nod(t2no.T, t2nd.T)
#     start,goal = (0,0), (127,127)
#     path = check.run_search(start,goal, euclidean_dist)
#     print(path)