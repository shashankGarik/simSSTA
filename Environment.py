import pygame
import numpy as np
import pandas as pd
import math,cv2
# from PIL import Image, ImageDraw
from controllers_APF import *
from test_cases import *



class Environment():
    def __init__(self, height, width, obs_vec):
        # Initialize Pygame necessary for initialising the simulation window and graphics
        print('Initializing Environment')
        pygame.init()

        # Define screen dimensions
        self.height,self.width = height,width
        self.screen = pygame.display.set_mode((self.width, self.height))
        
        #df for csv files
        self.dataframe_columns = ["Index","TotalAgents","LocalPoints", "GlobalPoints"]
        # self.df = pd.DataFrame(columns=self.dataframe_columns)

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
            'blueGray':(115, 147, 179),}

        self.agent_colors = ['glaucous','teal','indigo','blue','blueGray','lblue','lgreen']

        # Set up car and goal positions
        self.cur_pos = None
        self.goal_pos = None
        self.obstacles = obs_vec
        
        # temporary functionality - realistic objects:
        self.obstacle_img = pygame.transform.scale(pygame.image.load(r'C:\Users\Welcome\Documents\Kouby\M.S.Robo- Georgia Tech\GATECH LABS\SHREYAS_LAB\Simulation_Environment\Github Simulation Network\sim_vis_images\noentry.jpg'),(65, 65))
        self.car_img = pygame.transform.scale(pygame.image.load(r'C:\Users\Welcome\Documents\Kouby\M.S.Robo- Georgia Tech\GATECH LABS\SHREYAS_LAB\Simulation_Environment\Github Simulation Network\sim_vis_images\carssta.jpg'),(60, 60))
        self.pedestrian_img = pygame.transform.scale(pygame.image.load(r'C:\Users\Welcome\Documents\Kouby\M.S.Robo- Georgia Tech\GATECH LABS\SHREYAS_LAB\Simulation_Environment\Github Simulation Network\sim_vis_images\pedestrian.PNG'),(30, 30))
    
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
        transformation_matrix = np.array([
            [np.cos(theta), np.sin(theta), -x_origin * np.cos(theta) - y_origin * np.sin(theta)],
            [-np.sin(theta), np.cos(theta), x_origin * np.sin(theta) - y_origin * np.cos(theta)],
            [np.zeros(theta.shape), np.zeros(theta.shape), np.ones(theta.shape)]])
        
        homogeneous_global_points = np.column_stack((x_g, y_g, np.ones_like(x_g)))
        homogeneous_local_points = np.dot(homogeneous_global_points, transformation_matrix.T)
        homogeneous_local_points = homogeneous_local_points.transpose((1,0,2))
        local_points = homogeneous_local_points[:,:, :2]
        return local_points
    

    # #3##########################l-g
    ## for one frame /view any number of points
    def transform_local_to_global_path_vectorized(self,all_local_points,ssta_camera_indices, all_local_origin, all_frame_angle,n,k):
        """
        Vectorized conversion of local points to global points.
        
        :param local_points: A view* (m, k, 2) numpy array of local points for m agents and k points each.
        :param local_origin: A (2,) numpy array representing the x, y coordinates of the local origin.
        :param frame_angle: The rotation angle of the local frame in degrees.
        :return: A (n, k, 2) numpy array of global points. n is total ssta agents
        """
        # Convert angle to radians
        global_path_matrix=np.full((n,k,2),None)
        for view in range(len(all_local_points)):
            frame_angle,local_origin,local_points=all_frame_angle[view],all_local_origin[view],all_local_points[view]
            theta = np.radians(frame_angle)
            # Create the rotation matrix
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            # Create the translation vector (expanded to match the shape for broadcasting)
            translation_vector = local_origin.reshape(1, 1, 2)
            # Apply rotation
            rotated_points = np.dot(local_points, rotation_matrix.T)
            # Apply translation
            global_path_view_points = rotated_points + translation_vector
            # print(global_path_view_points.shape,global_path_matrix.shape)
            # remove this below condition later
            if len(ssta_camera_indices[view][0])!=0:
                global_path_matrix[ssta_camera_indices[view][0]]=global_path_view_points
        
        # print(global_path_matrix.shape)
        
        return global_path_matrix

    ################ Display Metrics ########################
    def display_collision_rate(self,collision_value):
        collison_text ="CR="+str(np.round(collision_value*60,0))+"coll/min"
        collison_render = self.font.render(collison_text, True, (0, 0, 0))
        col_rect = collison_render.get_rect()
        col_rect.topright = ( self.width-100, 30) 
        self.screen.blit(collison_render, col_rect)

    def display_total_time  (self,time):
        time_text ="t="+str(np.round(time,0))+"s"
        time_render = self.font.render(time_text, True, (0, 0, 0))
        self.col_rect = time_render.get_rect()
        self.col_rect.topright = ( self.width-100, 10) 
        self.screen.blit(time_render, self.col_rect)
    
    def display_v_c_ratio(self,volume,capacity):
        v_c_text ="v/c="+str(  np.round(volume/capacity,2))
        v_c_render = self.font.render(v_c_text, True, (0, 0, 0))
        self.v_c_rect = v_c_render.get_rect()
        self.v_c_rect.topright = ( self.width-100, 50) 
        self.screen.blit(v_c_render, self.v_c_rect )

    def display_traffic_speed(self,speed):
        speed_text ="v="+str(  np.round(speed,0))+"m/s"
        speed_render = self.font.render(speed_text, True, (0, 0, 0))
        speed_rect = speed_render.get_rect()
        speed_rect.topright = (self.width-100, 70)
        self.screen.blit(speed_render, speed_rect)
    ################ Metrics ########################
    
    #plotting on the screen
    
    def draw_map(self, color_BG = 'white', color_obs = 'red'):
        self.screen.fill(self.colors[color_BG])
        for obs in self.obstacles['circle']:
            pygame.draw.circle(self.screen, self.colors[color_obs], obs, 20)
            # img_rect = self.obstacle_img.get_rect(center=obs)
            # self.screen.blit(self.obstacle_img, img_rect)
        for x,y,w,h in self.obstacles['rectangle']:
            if self.display_realistic:
                img_rect = self.obstacle_img.get_rect(center=(x, y))
                self.screen.blit(self.obstacle_img, img_rect)  
            else:
                pygame.draw.rect(self.screen, self.colors[color_obs], pygame.Rect(x - w/2, y - h/2,w,h))

        #the intersection closest point
        # if self.debugging:
        #     for point_set in self.intersections:
        #         for x, y in point_set:
        #             pygame.draw.circle(self.screen, self.colors['black'], (x, y), 3)

    def plot_segment_frame(self,center,box_points, frame_color = 'black'):
        # print("inside")
        t_l,t_r,b_l,b_r = box_points
        box_points = np.array([t_l,t_r,b_r,b_l]).transpose(1,0,2)   
        # frames = np.array([[center, box_points]])
        if self.debugging == True:
            for i, row in enumerate(center.T):
                pygame.draw.polygon(self.screen, self.colors[frame_color], box_points[i,:],3)
            
                # pygame.draw.circle(self.screen, self.colors['red'], row, 3)

    def test_intersection_local_goal(self,intersections):
        # print(intersections.shape)
        if self.debugging:
            for intersection in intersections:
                if intersection.shape[1]==1:
                    pygame.draw.circle(self.screen, self.colors["blue"], (intersection[0], intersection[1]), 5)
                else:
                    for point_set in intersection:
                        x, y =point_set[0],point_set[1]
                        pygame.draw.circle(self.screen, self.colors["blue"], (x, y), 5)      

    
    def draw_agents_with_goals(self,collision_flags):
       
        for start, goal, collided in zip(self.cur_pos, self.goal_pos, collision_flags):
            if collided == False:
                agent_color = self.colors[self.agent_colors[int(start[4])]]
            else:
                agent_color = self.colors["red"]
            
            if self.debugging:
                pygame.draw.circle(self.screen, self.colors['black'], goal, 2)
            if start[6] == -1:
                if self.display_realistic:
                    img_rect = self.pedestrian_img.get_rect(center=start[:2])
                    self.screen.blit(self.pedestrian_img, img_rect)
                else:
                    pygame.draw.circle(self.screen, agent_color, start[:2], start[5])
                
            else:
                vertices = self.calculate_equilateral_polygon_vertices(start[:2],int(start[5]), int(start[6]))
                pygame.draw.polygon(self.screen, agent_color, vertices)

    def calculate_equilateral_polygon_vertices(self, center, radius, num_sides):
        angles = np.linspace(0, 2*np.pi, num_sides, endpoint=False)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        vertices = [(round(x[i],2), round(y[i],2)) for i in range(num_sides)]
        vertices.append(vertices[0])  # Append the first vertex again to close the polygon
        return vertices

    def camera_agents(self,local_points_camera_1,box_limits, agent_pos,goal_pos):
        self.camera_x_local, self.camera_x_global = 0,0
        min_x_1,max_x_1,min_y_1,max_y_1=(np.zeros(box_limits.shape)[:,np.newaxis], box_limits[:,np.newaxis], np.zeros(box_limits.shape)[:,np.newaxis], box_limits[:,np.newaxis])
        x_row_1,y_row_1=local_points_camera_1[:,:,0],local_points_camera_1[:,:,1]
        inside_camera_x = np.logical_and(x_row_1 >= min_x_1, x_row_1 <= max_x_1) & np.logical_and(y_row_1 >= min_y_1, y_row_1 <= max_y_1)
        
        global_agent_points = []
        local_points = []
        global_goal_points=[]
        camera_points_indices=[]
        for i in range(len(box_limits)):
            camera_x_indices=np.where(inside_camera_x[i])
            global_agent_points.append(agent_pos[camera_x_indices])
            global_goal_points.append(goal_pos[camera_x_indices])
            local_points.append(local_points_camera_1[i][camera_x_indices])
            camera_points_indices.append(camera_x_indices)
            
        return  local_points,global_agent_points,global_goal_points,camera_points_indices


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
        intersection_x,intersection_y=[p1[:,0] + new_ua * (p2[:,0] - p1[:,0]), p1[:,1] + new_ua * (p2[:,1] - p1[:,1])]
        intersection_stack= np.vstack([intersection_x, intersection_y])
        # print(intersection_stack.T)
        return intersection_stack.T

    def find_segment_square_intersection(self,segment, square):
        """Find intersections between a line segment and a square."""
        intersections = []
        # Define square sides as segments
        square_sides = np.array([[square[0], square[1]], [square[1], square[2]],
                        [square[2], square[3]], [square[3], square[0]]])
        intersections = self.line_intersection(segment[0], segment[1],square_sides )   
        return intersections

    def global_local_goal(self,ssta_goal_pos,camera_points_indices,local_cur_points,agents_global_points,goal_global_points,frame_edges,frame_angle):
        #finds the new local global goal in the box and replaces in the ssta goal pose local goal points first as None
        intersections_all=[]
        #mask where all indices are not in the camera indices to make anything outside indices as None
        combined_camera_indices = np.concatenate([arr[0] for arr in camera_points_indices])
        mask = np.ones(len(ssta_goal_pos), dtype=bool)
        mask[combined_camera_indices] = False
        ssta_goal_pos[mask,2:]=np.full(((abs(len(ssta_goal_pos)-len(combined_camera_indices))),5), None)#if agent exits box everything will become none



        for view in range(len(agents_global_points)):
            local_goal_view=ssta_goal_pos[camera_points_indices[view],-1].flatten()
            
            # local_mask=np.full(len(camera_points_indices[view]),False ,dtype=bool)
            
            none_indices=np.argwhere(local_goal_view==None)
            
            camera_temp_indices=camera_points_indices[view][0].flatten()
            camera_temp_indices=camera_temp_indices[none_indices]
            # print(camera_temp)
            (t_l,t_r,b_l,b_r)=frame_edges
            segment = [agents_global_points[view][:,:2],goal_global_points[view] ]  # Line segment defined by its two end points
            square = [(t_l[view][0],t_l[view][1]),(t_r[view][0],t_r[view][1]),(b_r[view][0],b_r[view][1]),(b_l[view][0],b_l[view][1])]  # Square defined by its four corners
            # Find intersections
            intersection_view = self.find_segment_square_intersection(segment, square)
            # print(intersection_view.shape)
            # print("Intersection Points:", intersections)
            intersections_all.append(intersection_view)
            # intersections_global_frame[none_indices_camera_view]=intersection_view
            # intersections_all.append(intersection_view)
            #converting view_goal_global_point to view_goal_local_Point
            goal_view_points_ssta = self.global_local_transform(intersection_view,t_l,frame_angle)
            ssta_goal_pos[camera_temp_indices,2:4]=np.round(np.float64(goal_view_points_ssta[view][none_indices]),2) #adding global local
            ssta_goal_pos[camera_points_indices[view],4:6]=np.round(local_cur_points[view],2) # adding local curr point
            ssta_goal_pos[camera_points_indices[view],6]=np.full((len(intersection_view),), view)
   
        
        return ssta_goal_pos,intersections_all,combined_camera_indices
   

        
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx -X- xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def get_frame(self, frame_sizes ,frame_corners):
        t_l,t_r,b_l,b_r = frame_corners # unpacking corners for all frames
        outputs = []
        for idx in range(len(frame_sizes)): #iterating for each box
                width, height = frame_sizes[idx], frame_sizes[idx] 
                tl,tr,bl,br = t_l[idx],t_r[idx],b_l[idx],b_r[idx]

                screen_array = pygame.surfarray.array3d(self.screen)
                transposed_array = np.transpose(screen_array, (1, 0, 2))
                pt1=np.float32([tl,tr,bl,br])
                pt2=np.float32([[0,0],[width,0],[0,height],[width,height]])
                matrix = cv2.getPerspectiveTransform(pt1,pt2)
                output = cv2.warpPerspective(transposed_array,matrix,(width, height))
                output = cv2.resize(output, (128, 128))
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                # cv2.imshow('not gonna work',output)
                # cv2.waitKey(1)
                outputs.append(output)

        return outputs



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

        main_path = "C:/Users/Welcome/Documents/Kouby/M.S.Robo- Georgia Tech/GATECH LABS/SHREYAS_LAB/Simulation_Environment/Github Simulation Network/dataset/"

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

            outputs = self.get_frame(frame_sizes ,frame_corners)
            
            for idx in range(len(frame_sizes)): #iterating for each box
                save_image_name = branched_path + "/camera_" + str(idx)  + "/image_" + str(index) + ".jpg"
                output = outputs[idx]
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

    
       
       
