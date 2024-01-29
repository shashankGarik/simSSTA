import pygame
import numpy as np
import pandas as pd
import math,cv2,imageio
from PIL import Image, ImageDraw
from controllers import *
from test_cases import *


class Environment():
    def __init__(self, height, width, obs_vec):
        # Initialize Pygame necessary for initialising the simulation window and graphics
        print('Initializing Environment')
        pygame.init()

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
            'lgreen': (0,255,0),
            'blue':  (0,0,255),
            'lblue': (0,191,255)
        }

        # Set up car and goal positions
        self.cur_pos = None
        self.goal_pos = None
        self.obstacles = obs_vec
    
    def update_poses(self, cur_pos, goal_pos):
        self.cur_pos = cur_pos
        self.goal_pos = goal_pos

    
    
    # Ths function below segments the frame to smaller portions with specific angle
    def segment_frame(self,frame_angle,translation_point,segment_size):
        xtl,ytl=translation_point
        width, height = segment_size, segment_size 
        xc,yc = xtl + width // 2, ytl + height // 2 
        x_tr, y_tr = xtl + width, ytl
        x_bl, y_bl = xtl, ytl + height
        x_br, y_br = xtl + width, ytl + height
        rotated_rectangle =self.rotate_points([(xtl,ytl),(x_tr, y_tr),(x_bl, y_bl),(x_br, y_br)], frame_angle, (xc,yc))
        tl,tr,bl,br=rotated_rectangle[0],rotated_rectangle[1],rotated_rectangle[2],rotated_rectangle[3]
        #returns angle,center points and 4 corners of the rectangle(tl/translation points,tr,bl,br)
        return frame_angle,(xc,yc),(tl,tr,bl,br)

    #This function takes in the global points and converts to the local points
    def global_local_transform(self,global_position,local_origin,frame_angle):
        x_g, y_g = global_position[:, 0], global_position[:, 1]
        x_origin, y_origin = local_origin
        theta = np.radians(frame_angle)
        transformation_matrix = np.array([
            [np.cos(theta), np.sin(theta), -x_origin * np.cos(theta) - y_origin * np.sin(theta)],
            [-np.sin(theta), np.cos(theta), x_origin * np.sin(theta) - y_origin * np.cos(theta)],
            [0, 0, 1]])
        homogeneous_global_points = np.column_stack((x_g, y_g, np.ones_like(x_g)))
        homogeneous_local_points = np.dot(homogeneous_global_points, transformation_matrix.T)
        local_points = homogeneous_local_points[:, :2]
        return local_points
    
    ################ Display Metrics ########################
    def display_collision_rate(self,collision_value):
        collison_text ="CR="+str(np.round(collision_value*60,0))+"coll/min"
        collison_render = self.font.render(collison_text, True, (255, 255, 255))
        col_rect = collison_render.get_rect()
        col_rect.topright = ( self.width-100, 30) 
        self.screen.blit(collison_render, col_rect)

    def display_total_time  (self,time):
        time_text ="t="+str(np.round(time,0))+"s"
        time_render = self.font.render(time_text, True, (255, 255, 255))
        self.col_rect = time_render.get_rect()
        self.col_rect.topright = ( self.width-100, 10) 
        self.screen.blit(time_render, self.col_rect)
    
    def display_v_c_ratio(self,volume,capacity):
        v_c_text ="v/c="+str(  np.round(volume/capacity,2))
        v_c_render = self.font.render(v_c_text, True, (255, 255, 255))
        self.v_c_rect = v_c_render.get_rect()
        self.v_c_rect.topright = ( self.width-100, 50) 
        self.screen.blit(v_c_render, self.v_c_rect )

    def display_traffic_speed(self,speed):
        speed_text ="v="+str(  np.round(speed,0))+"m/s"
        speed_render = self.font.render(speed_text, True, (255, 255, 255))
        speed_rect = speed_render.get_rect()
        speed_rect.topright = ( self.width-100, 70) 
        self.screen.blit(speed_render, speed_rect )
    ################ Metrics ########################
    
    #plotting on the screen
    def draw_map(self):
        self.screen.fill(self.colors['black'])
        for obs in self.obstacles['circle']:
            pygame.draw.circle(self.screen, self.colors['red'], obs, 20)
        for x,y,w,h in self.obstacles['rectangle']:
            pygame.draw.rect(self.screen, self.colors['red'], pygame.Rect(x - w/2, y - h/2,w,h))

        # for point_set in self.intersections:
        #     for x, y in point_set:
        #         pygame.draw.circle(self.screen, self.colors['white'], (x, y), 3)

    def plot_segment_frame(self,center,tl,tr,bl,br):
        pygame.draw.polygon(self.screen, self.colors['white'], [tl,tr,br,bl],3 )
        pygame.draw.circle(self.screen, self.colors['red'], tl, 3)

    def draw_agents_with_goals(self, collision_flag):
        for start, goal, collided in zip(self.cur_pos, self.goal_pos, collision_flag):
            if collided == False:
                agent_color = self.colors["blue"]
            else:
                agent_color = self.colors["lgreen"]
            pygame.draw.circle(self.screen, self.colors['white'], goal, 2)
            pygame.draw.circle(self.screen, agent_color, start[:2], 10)


 
    #save camera images
    def save_camera_image(self,frame_size ,frame_corners,index):
        width, height = frame_size
        tl,tr,bl,br=frame_corners
        screen_array = pygame.surfarray.array3d(self.screen)
        transposed_array = np.transpose(screen_array, (1, 0, 2)) 
        pt1=np.float32([tl,tr,bl,br])
        pt2=np.float32([[0,0],[width,0],[0,height],[width,height]])
        matrix=cv2.getPerspectiveTransform(pt1,pt2)
        output=cv2.warpPerspective(transposed_array,matrix,(width, height))
        save_image_name="test_dataset_1"+"/captured_image"+str(np.round(index,3))+".jpg"
        # imageio.imwrite(save_image_name, output)
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
    #The rotate point function is used for the rotation of segment in segment_frame function
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
        angle_rad = math.radians(angle)
        rotation_matrix = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)]
        ])
        rotated_points = []
        for point in points:
            # Translate the point to the origin, rotate, and then translate it back
            translated_point = np.array(point) - np.array(center)
            rotated_point = np.dot(rotation_matrix, translated_point)
            final_point = rotated_point + np.array(center)
            rotated_points.append(tuple(final_point.astype(float)))
        return rotated_points
       
       



