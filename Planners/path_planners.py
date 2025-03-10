# This file is the planner file and the return format must always be a set number of m values for the path,
#If n agents are there then the path shape must be view * (n,m,2)- numpy arrays in a list where list contains agents of views seperated
import numpy as np
from Planners.astar import *
from Planners.utils import *
import os
import cv2
from multiprocessing import Pool

class Planners():
    def __init__(self,replanning_time_interval):
        self.replanning_time_interval=replanning_time_interval
      
    def a_star(self,curr_poses, goal_poses, box_num, global_paths, t2no, t2nd, max_timestep):
        n_agents = len(curr_poses)
        updated_paths = list(global_paths)

        for i in range(n_agents):
            if updated_paths[i] is not None or box_num[i] is None:
                continue
            
            segment_idx = box_num[i]

            # Ensure the segment index is valid
            if not (0 <= segment_idx < len(box_num)):
                updated_paths[i] = None
                continue

            t2no, t2nd = t2no[box_num[i]], t2nd[box_num[i]]

            check = Astar_T2nod_agentic(t2no, t2nd, r=1,g_f=1, max_time_step=max_timestep)

            start, goal = curr_poses[i], goal_poses[i]
            print(check.run_search(start, goal))

            updated_paths[i] = check.run_search(start, goal)

        return updated_paths


    def straigh_path_w_noise(self,curr_global_pnts, global_frame_goal_pnts, segment_numbers, global_paths, ssta_boxes, num_points=20):
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

            path = np.linspace(curr_global_pnts[i], global_frame_goal_pnts[i], num_points)
            path += np.random.randn(*path.shape)*3
            updated_paths[i] = path
        return updated_paths
    


    # Generate random points inside the rotated box
    def generate_random_point(self,half_size,x_center,y_center,rotation_matrix,top_left_x,top_left_y,size):
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
                    
    def compute_global_paths(self,curr_global_pnts, global_frame_goal_pnts, segment_numbers, global_paths, ssta_boxes, num_points=20):
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
            # Generate random intermediate points
            random_points = np.array([self.generate_random_point(half_size,x_center,y_center,rotation_matrix,top_left_x,top_left_y,size) for _ in range(num_points - 2)])

            # Construct the path with start, random, and goal points
            path = np.vstack([curr_global_pnts[i], random_points, global_frame_goal_pnts[i]])

            # Assign the computed path to the global paths
            updated_paths[i] = path

        return updated_paths
    
####integrate replanning here : countdowntime and path index 

        ###countdown timer for replanning 
        ####make path index of that to None so that every time it replans the controller changes path index to 0 
        ###so return path index total and global paths total
        ###replanning should hold start point and goal point 
        ##everytimwe replanning happens change starttime to current time from self.timer
        
    def compute_glbl_pth_rndm_lngth(self,curr_global_pnts, global_frame_goal_pnts, segment_numbers, global_paths, ssta_boxes, path_indices,global_start_times,global_timer):
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
        updated_global_start_times=np.copy(global_start_times)
        updated_path_indices=np.copy(path_indices)

        for i in range(n_agents):
            
            
            # Skip computation if a global path already exists or if no segment is assigned
            if segment_numbers[i] is None:
                continue

            segment_idx = segment_numbers[i]
            # Ensure the segment index is valid
            if not (0 <= segment_idx < len(ssta_boxes)):
                updated_paths[i] = None
                continue
            ###############################

            ###if start time is None then time is recorded
            if updated_global_start_times[i] is None :
                updated_global_start_times[i]=global_timer

            if updated_global_start_times[i] is not None and updated_paths[i] is not None: 
                # print(global_timer-updated_global_start_times[i])
                if (global_timer-updated_global_start_times[i]) < self.replanning_time_interval:
                    continue
                else:
                    updated_path_indices[i]=None
                    updated_global_start_times[i]=global_timer
            ###############################


            # Extract box properties
            angle, top_left_x, top_left_y, size = ssta_boxes[segment_idx]
            half_size = size / 2
            x_center = top_left_x + half_size
            y_center = top_left_y + half_size
            rotation_matrix = np.array([
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
            ])

            num_points= np.random.randint(1, 6)
            # Generate random intermediate points

            #####This is the portion the astar function must return 
            random_points = np.array([self.generate_random_point(half_size,x_center,y_center,rotation_matrix,top_left_x,top_left_y,size) for _ in range(num_points - 2)])
            # Construct the path with start, random, and goal points
            if len(random_points)==0:path = np.vstack([curr_global_pnts[i], global_frame_goal_pnts[i]])
            else:path = np.vstack([curr_global_pnts[i], random_points, global_frame_goal_pnts[i]])
            # Assign the computed path to the global paths
            updated_paths[i] = path

        # print(updated_global_start_times)
        return updated_paths,updated_path_indices,updated_global_start_times
 