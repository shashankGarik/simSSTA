import pygame
import numpy as np
import pandas as pd
import math,cv2
# from PIL import Image, ImageDraw
from controllers_2 import *



class Environment():
    def __init__(self, height, width, obs_vec):
        # Initialize Pygame necessary for initialising the simulation window and graphics
        print('Initializing Environment')
        pygame.init()
        self.debugging = False
        self.save_data = False

        # Define screen dimensions
        self.width, self.height = width,height
        self.screen = pygame.display.set_mode((self.width, self.height))
        
        #df for csv files
        self.dataframe_columns = ["Index","TotalAgents","LocalPoints", "GlobalPoints"]
        self.df = pd.DataFrame(columns=self.dataframe_columns)

        self.font = pygame.font.Font(None, 25)  # Smaller font size
        # Define colors
        self.colors = {
            'white': (255,255,255),
            'black': (0,0,0),
            'red':   (255,0,0),
            'green': (0,180,0),
            'purple':(145,30,180),
            'teal':(0,128,128),
            'yellow': (255,225,25),
            'blue':  (0,0,255),
            'lgreen': (0,255,0),
            'lblue': (0,191,255),
            'glaucous':(96, 130, 182),
            'indigo': (63, 0, 255),
            'blueGray':(115, 147, 179),

        }

        self.agent_colors = ['glaucous','teal','indigo','blue','blueGray','lblue']

        # Set up car and goal positions
        self.cur_pos = None
        self.goal_pos = None
        self.obstacles = obs_vec
    
    def update_poses(self, cur_pos, goal_pos):
        self.cur_pos = cur_pos
        self.goal_pos = goal_pos


    def segment_frame(self, boxes):
        """
        This function below segments the frame to smaller portions with specific angle

        Parameters:
        - boxes: numpy array with angle, center_x, center_y, box_width/height (vectorized)

        """

        f_angle = boxes[:,0]
        x_tl, y_tl = boxes[:,1], boxes[:,2]
        w, h = boxes[:,3], boxes[:,3]
        x_c, y_c = x_tl + w // 2, y_tl + h // 2
        xtr, ytr = x_tl + w, y_tl
        xbl, ybl = x_tl, y_tl + h
        xbr, ybr = x_tl + w, y_tl + h
        rotated_rectangle = self.rotate_points(np.array([[x_tl,y_tl],[xtr, ytr],[xbl, ybl],[xbr, ybr]]), f_angle, np.array([x_c,y_c]))

        t_l,t_r,b_l,b_r= rotated_rectangle[:,0],rotated_rectangle[:,1],rotated_rectangle[:,2],rotated_rectangle[:,3]


        return f_angle, np.array([x_c, y_c]), (t_l,t_r,b_l,b_r)
        
    
    def rotate_points(self,points, angle, center):
        """
        Rotate a set of points around a center by a given angle.

        Parameters:
        - points: List of (x, y) coordinates representing the points of the rectangle.
        - angle: Rotation angle in degrees.
        - center: (x, y) coordinates of the rotation center.

        Returns:
        List of rotated (x, y) coordinates.
        """
        angle_rad = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        rotation_matrix = rotation_matrix.transpose((2,0,1))
        points = points.transpose((2,0,1))
        translated_point = points - center.T[:,np.newaxis,:]
        rotated_point = np.matmul(rotation_matrix, translated_point.transpose((0,2,1))) + center.T[:,:,np.newaxis]
        return rotated_point.transpose(0,2,1)

    def global_local_transform(self,global_position,local_origin,frame_angle):
        """
        This function takes in the global points and converts to the local points
        """
        x_g, y_g = global_position[:, 0], global_position[:, 1]
        x_origin, y_origin = local_origin[:,0], local_origin[:,1]
        theta = np.radians(frame_angle)
        some = x_origin * np.cos(theta) - y_origin * np.sin(theta)
        transformation_matrix = np.array([
            [np.cos(theta), np.sin(theta), -x_origin * np.cos(theta) - y_origin * np.sin(theta)],
            [-np.sin(theta), np.cos(theta), x_origin * np.sin(theta) - y_origin * np.cos(theta)],
            [np.zeros(theta.shape), np.zeros(theta.shape), np.ones(theta.shape)]])
        
        homogeneous_global_points = np.column_stack((x_g, y_g, np.ones_like(x_g)))
        homogeneous_local_points = np.dot(homogeneous_global_points, transformation_matrix.T)
        homogeneous_local_points = homogeneous_local_points.transpose((1,0,2))
        local_points = homogeneous_local_points[:,:, :2]
        return local_points
    
    def camera_agents(self,local_points_camera_1,box_limits, agent_pos,goal_pos):
        self.camera_x_local, self.camera_x_global = 0,0
        min_x_1,max_x_1,min_y_1,max_y_1=(np.zeros(box_limits.shape)[:,np.newaxis], box_limits[:,np.newaxis], np.zeros(box_limits.shape)[:,np.newaxis], box_limits[:,np.newaxis])
        x_row_1,y_row_1=local_points_camera_1[:,:,0],local_points_camera_1[:,:,1]
        inside_camera_x = np.logical_and(x_row_1 >= min_x_1, x_row_1 <= max_x_1) & np.logical_and(y_row_1 >= min_y_1, y_row_1 <= max_y_1)
        
        global_agent_points = []
        local_points = []
        global_goal_points=[]

        for i in range(len(box_limits)):
            camera_x_indices=np.where(inside_camera_x[i])
            global_agent_points.append(agent_pos[camera_x_indices])
            global_goal_points.append(goal_pos[camera_x_indices])
            local_points.append(local_points_camera_1[i][camera_x_indices])

        return  local_points,global_agent_points,global_goal_points
        

    
    #plotting on the screen
    def draw_map(self, color_BG = 'black', color_obs = 'red'):
        self.screen.fill(self.colors[color_BG])
        for obs in self.obstacles['circle']:
            pygame.draw.circle(self.screen, self.colors[color_obs], obs, 20)
        for x,y,w,h in self.obstacles['rectangle']:
            pygame.draw.rect(self.screen, self.colors[color_obs], pygame.Rect(x - w/2, y - h/2,w,h))

    def test_intersection(self,intersections):
            print(intersections.shape)
            if intersections.shape[1]==1:
                pygame.draw.circle(self.screen, self.colors["blue"], (intersections[0], intersections[1]), 10)
            else:
                for point_set in intersections:
                    print(point_set)
                    x, y =point_set[0],point_set[1]
                    print("hi",x,y)
                    pygame.draw.circle(self.screen, self.colors["blue"], (x, y), 10)

    def plot_segment_frame(self,center,box_points):
        # print(center)
        t_l,t_r,b_l,b_r = box_points
        box_points = np.array([t_l,t_r,b_r,b_l]).transpose(1,0,2)   
        # frames = np.array([[center, box_points]])

        for i, row in enumerate(center.T):
            pygame.draw.polygon(self.screen, self.colors['black'], box_points[i,:],3)
            if self.debugging == True:
                pygame.draw.circle(self.screen, self.colors['red'], row, 3)

    def draw_agents_with_goals(self, collision_flag):
        for start, goal, collided in zip(self.cur_pos, self.goal_pos, collision_flag):
            if collided == False:
                agent_color = self.colors[self.agent_colors[int(start[4])]]
            else:
                agent_color = self.colors["red"]
            if self.debugging:
                pygame.draw.circle(self.screen, self.colors['black'], goal, 2)
            pygame.draw.circle(self.screen, agent_color, start[:2], 10)

    ################### finding local goal direct projection ##########
    def line_intersection(self,p1, p2, square_sides):
        np.seterr(divide='ignore')
        p3_0,p3_1,p4_0,p4_1=square_sides[:,0,0],square_sides[:,0,1],square_sides[:,1,0],square_sides[:,1,1] 
        denominator = np.dot((p2[:,0] - p1[:,0])[:, np.newaxis] , ((p4_1- p3_1)[:, np.newaxis]).T) - np.dot(((p4_0 - p3_0)[:, np.newaxis]),((p2[:,1] - p1[:,1])[:, np.newaxis].T)).T
        a = p1[:,1][:, np.newaxis]  - p3_1[:, np.newaxis].T
        b = p1[:,0][:, np.newaxis]  - p3_0[:, np.newaxis] .T
        numerator1 = (np.tile(((p4_0 - p3_0)[:, np.newaxis]),(1,p2.shape[0])) *a.T) - (np.tile(((p4_1 - p3_1)[:, np.newaxis]),(1,p2.shape[0])) *b.T)
        numerator2 = ((np.tile(((p2[:,0] - p1[:,0])[:, np.newaxis]),(1,4)) *a)).T - ((np.tile(((p2[:,1] - p1[:,1])[:, np.newaxis]),(1,4)) *b)).T
        ua = numerator1.T / denominator
        ub = numerator2.T / denominator
        valid = (0 <= ua) & (ua <= 1) & (0 <= ub) & (ub <= 1)
        idx = np.where(valid)
        new_ua,new_ub=ua[idx],ub[idx]
        intersection_matrix=np.full((p2.shape[0],2),None)
        # print(result_matrix)
        print( new_ua,new_ub)

        # intersections = np.where(valid[:, None], p1 + ua[:, None] * (p2 - p1), None)
        intersection_x,intersection_y=[p1[:,0] + new_ua * (p2[:,0] - p1[:,0]), p1[:,1] + new_ua * (p2[:,1] - p1[:,1])]
        intersection_stack= np.vstack([intersection_x, intersection_y])
        # intersection=np.array(intersection_stack)
        print(intersection_stack.T)
        return intersection_stack.T


 


    def find_segment_square_intersection(self,segment, square):
        """Find intersections between a line segment and a square."""
        intersections = []
        # Define square sides as segments
        square_sides = np.array([[square[0], square[1]], [square[1], square[2]],
                        [square[2], square[3]], [square[3], square[0]]])
        print(square_sides.shape)
        # Check intersection with each side of the square
        print("ppppppppppppppppppppp")
        intersections = self.line_intersection(segment[0], segment[1],square_sides )
        print(intersections.shape)
        intersections_with_nan = np.where(intersections == None, np.nan, intersections)
        print(intersections)
        # filtered_values = intersections_with_nan[~np.isnan(intersections_with_nan).any(axis=1)]

        # print("inter",filtered_values)
        #return an array of nX1X2 where each n is  achent and 1X2 is the intersection
  
        return intersections

   

    
    def global_local_goal(self,agents_global_points,goal_global_points,frame_edges):
    
        for view in range(len(agents_global_points)):
            (t_l,t_r,b_l,b_r)=frame_edges
            print(agents_global_points[view][:,:2])
            print(goal_global_points[view])


            segment = [agents_global_points[view][:,:2],goal_global_points[view] ]  # Line segment defined by its two end points
            # segment=[(100,400),(700,100)]
            square = [(t_l[view][0],t_l[view][1]),(t_r[view][0],t_r[view][1]),(b_r[view][0],b_r[view][1]),(b_l[view][0],b_l[view][1])]  # Square defined by its four corners
            # Find intersections
            intersections = self.find_segment_square_intersection(segment, square)
            print("Intersection Points:", intersections)
            return intersections
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx -X- xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx






 
    
    #save camera images
    def save_camera_image(self, frame_sizes ,frame_corners, index, train_len, val_len, test_len, buffer = 500):

        train_buffer = buffer
        val_buffer = buffer + train_buffer + train_len
        test_buffer = buffer + val_buffer + val_len

        if index >= train_buffer and index  < train_buffer + train_len:
            data_type = 'train'
            reset_counter = buffer
        elif index >= val_buffer and index  < val_buffer + val_len:
            data_type = 'val'
            reset_counter = val_buffer
        elif index >= test_buffer and index  < test_buffer + test_len:
            data_type = 'test'
            reset_counter = test_buffer
        else:
            data_type = None # stop saving

        t_l,t_r,b_l,b_r = frame_corners # unpacking corners for all frames

        main_path = "C:/Users/shash/OneDrive/Desktop/SSTA_2/simSSTA/dataset/"

        if data_type != None:
            branched_path = main_path + data_type
            for idx in range(len(frame_sizes)): #iterating for each box
                import os
                path = branched_path + "/camera_" + str(idx) 
                if not os.path.exists(path):
                    os.makedirs(path)
        
        if index >= buffer and data_type != None:
            
            index = index - reset_counter
            if index%100 == 0:
                print("current image ("+data_type+"): ", index)

            for idx in range(len(frame_sizes)): #iterating for each box
                width, height = frame_sizes[idx], frame_sizes[idx] 
                tl,tr,bl,br = t_l[idx],t_r[idx],b_l[idx],b_r[idx]

                screen_array = pygame.surfarray.array3d(self.screen)
                transposed_array = np.transpose(screen_array, (1, 0, 2))
                pt1=np.float32([tl,tr,bl,br])
                pt2=np.float32([[0,0],[width,0],[0,height],[width,height]])
                matrix = cv2.getPerspectiveTransform(pt1,pt2)
                output = cv2.warpPerspective(transposed_array,matrix,(width, height))
                # print(idx)
                
                save_image_name = branched_path + "/camera_" + str(idx)  + "/image_" + str(index) + ".jpg"
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                # print(save_image_name)
                # cv2.imshow('image',output)
                # cv2.waitKey(50000)
                if not cv2.imwrite(save_image_name, output):
                    raise Exception("Could not write image")
            

    #save camera data in csv file
    def save_camera_data(self,index,local_points,global_points):
        current_step_df = pd.DataFrame([[index,len(local_points),local_points, global_points]], columns=self.dataframe_columns)
        self.df = self.df.append(current_step_df, ignore_index=True)
        if index%100==0:
            csv_file_name = "output_data.csv"
            # self.df.to_csv(csv_file_name, index=False)
        
    ###################### Helper Functions####################
    def crop_image(self,image, polygon_points):
        """
        This function takes in the entire image and the rotated rectangle points and crops it 
        """
        mask = pygame.Surface(image.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(mask, (255, 255, 255, 255), polygon_points)
        masked_image = pygame.Surface(image.get_size(), pygame.SRCALPHA)
        masked_image.blit(image, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        masked_image.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        return masked_image
    
       
       



