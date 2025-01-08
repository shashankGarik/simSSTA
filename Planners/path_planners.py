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
      
        ######Yet to complete                    ######Yet to complete                    ######Yet to complete
    def a_star(self,ssta_agents_goal_poses,time_step,ssta_path_indices, t2no, t2nd):
        ######Yet to complete                    ######Yet to complete                    ######Yet to complete
        
        ssta_agents_start_poses=ssta_agents_goal_poses[:,4:6]
        # print(ssta_agents_goal_poses)
        # print(ssta_agents_start_poses)

        # return as view,n,m,2 - this is as a list
        
        for idx, each in enumerate(ssta_agents_goal_poses):
            
            if each[-1] == None :
                continue
            elif  ssta_path_indices[0]==self.replanning_index or self.path[0][0][0]==-1:
                # count = str(time_step+1)
                # count_filled = count.zfill(8)
                # t2no_path = os.path.join(r'C:/Users/Welcome/Documents/Kouby/M.S.Robo- Georgia Tech/GATECH LABS/SHREYAS_LAB/Simulation_Environment/Github Simulation Network/dataset/train/_MOG_t2no_50','camera_'+str(each[-1]),'t2no_' + count_filled + '.png')
                # t2nd_path = os.path.join(r'C:/Users/Welcome/Documents/Kouby/M.S.Robo- Georgia Tech/GATECH LABS/SHREYAS_LAB/Simulation_Environment/Github Simulation Network/dataset/train/_MOG_t2no_50','camera_'+str(each[-1]),'t2nd_' + count_filled + '.png')
                # t2no = cv2.imread(t2no_path, cv2.IMREAD_GRAYSCALE)
                # t2nd = cv2.imread(t2nd_path, cv2.IMREAD_GRAYSCALE)
                check = Astar_T2nod(t2no[each[-1]].T, t2nd[each[-1]].T)
                start = tuple(np.int16(each[4:6]*(128/300)))
                goal = tuple(np.int16(each[2:4]*(128/300)))
                print(start, goal, each[-1])
                path = check.run_search(start,goal, euclidean_dist)
                if path is not None:
                    # print(path, each[-1])
                    # print(np.array(path))
                    self.path=(np.array([(path[:self.path_size])])/128)*300
                    # print(self.path)
                    print("inside",self.path)
                    return self.path
        
        return self.path

    def straigh_path_w_noise(self,curr_global_pnts, global_frame_goal_pnts, segment_numbers, global_paths, ssta_boxes, num_points=3):
        n_agents = len(curr_global_pnts)
        updated_paths = list(global_paths)  # Copy to avoid modifying the input directly
        for i in range(n_agents):
            # Skip computation if a global path already exists or if no segment is assigned
            if updated_paths[i] is not None or segment_numbers[i] is None:
                continue


        return updated_paths

    def compute_global_paths(self,curr_global_pnts, global_frame_goal_pnts, segment_numbers, global_paths, ssta_boxes, num_points=3):
        """
        Compute global paths for agents based on their current and goal positions,
        segment assignments, and segment-specific boxes.

        Args:
            curr_global_pnts (np.ndarray): Current positions of agents (n, 2).
            global_frame_goal_pnts (np.ndarray): Goal positions of agents (n, 2).
            segment_numbers (list or np.ndarray): Segment box index for each agent (n,).
            global_paths (list): Existing paths for agents (n,).
            ssta_boxes (np.ndarray): Array of segment-specific boxes (m, 4) [angle, x, y, size].
            num_points (int): Number of points in the path (including start and goal).

        Returns:
            list: Updated global paths for agents.
        """
        n_agents = len(curr_global_pnts)
        updated_paths = list(global_paths)  # Copy to avoid modifying the input directly

        for i in range(n_agents):
            # Skip computation if a global path already exists or if no segment is assigned
            if updated_paths[i] is not None or segment_numbers[i] is None:
                continue

            segment_idx = segment_numbers[i]

            # Ensure the segment index is valid
            if not (0 <= segment_idx < len(ssta_boxes)):
                updated_paths[i] = None
                continue

            # Extract box properties
            angle, top_left_x, top_left_y, size = ssta_boxes[segment_idx]
            half_size = size / 2
            x_center = top_left_x + half_size
            y_center = top_left_y + half_size
            rotation_matrix = np.array([
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
            ])

            # Generate random points inside the rotated box
            def generate_random_point():
                while True:
                    # Generate a random point in the local frame of the box
                    local_point = np.random.uniform(
                        low=[-half_size, -half_size],
                        high=[half_size, half_size]
                    )
                    # Transform the point to the global frame
                    global_point = (rotation_matrix @ local_point) + np.array([x_center, y_center])
                    # Check if the point lies inside the box
                    if (
                        top_left_x <= global_point[0] <= top_left_x + size and
                        top_left_y <= global_point[1] <= top_left_y + size
                    ):
                        return global_point

            # Generate random intermediate points
            random_points = np.array([generate_random_point() for _ in range(num_points - 2)])

            # Construct the path with start, random, and goal points
            path = np.vstack([curr_global_pnts[i], random_points, global_frame_goal_pnts[i]])

            # Assign the computed path to the global paths
            updated_paths[i] = path

        return updated_paths
