import numpy as np


class APFSSTAAgents():
    def __init__(self,obstacles,controller_apf,controller_ssta,frame_rate,infinity):
        self.obstacles = obstacles
        ##APF agent initialise
        self.apf_car_pos = np.array([[-50.0, 300.0, 0.0, 0.0, 0, 15, 3]])# startx,starty,vx,vy,colour,radius,shape(agent)
        # goal = np.array([[800, 1500],[700, 1600]])  # global_goal_x,global_goal,y,local_goal_x(intersection_x),local_goal_y(intersection_y)
        self.apf_goal_pos = np.array([[800, 1500]]) # the new one is an object

        ###SSTA agent initialise
        # self.car_pos = np.array([[-20.0, 300.0, 0.0, 0.0, 6, 15, -1],[-10.0, 50.0, 0.0, 0.0,6, 15, -1],[-10.0, 80.0, 0.0, 0.0,6, 15, -1]])# startx,starty,vx,vy,colour,radius,shape(agent)
        # # goal = np.array([[800, 1500],[700, 1600]])  # global_goal_x,global_goal_y,local_goal_x(intersection_x),local_goal_y(intersection_y)
        # self.goal_pos = np.array([[1500, 600,None,None,None,None,None],[1500, 400,None,None,None,None,None],[1500, 500,None,None,None,None,None]]) # globalgx,globalgy,goalviewlocalgx,goalviewlocalgy,currlocalviewx,currlocalviewy,view/segment
        self.ssta_car_pos = np.array([[-10.0, 100.0, 0.0, 0.0,6, 15, -1]])# startx,starty,vx,vy,colour,radius,shape(agent)
        # goal = np.array([[800, 1500],[700, 1600]])  # global_goal_x,global_goal,y,local_goal_x(intersection_x),local_goal_y(intersection_y)
        self.ssta_goal_pos = np.array([[1000, 150,None,None,None,None,None]]) 
        # self.car_pos = np.array([[-10.0, 300.0, 0.0, 0.0,6, 15, -1],[-50.0, 0.0, 0.0, 0.0,6, 15, -1]])# startx,starty,vx,vy,colour,radius,shape(agent)
        # # goal = np.array([[800, 1500],[700, 1600]])  # global_goal_x,global_goal,y,local_goal_x(intersection_x),local_goal_y(intersection_y)
        # self.goal_pos = np.array([[1000, 100,None,None,None,None,None],[200, 800,None,None,None,None,None]]) 
        

        self.infinity = infinity
        self.apf_control = controller_apf(self.apf_car_pos, self.apf_goal_pos, self.obstacles)#apf controller
        self.ssta_control = controller_ssta(self.ssta_car_pos, self.ssta_goal_pos, self.obstacles)#ssta controller
        self.apf_control.dt = 1/frame_rate
        self.ssta_control.dt = 1/frame_rate
        self.apf_control.frame_h,self.apf_control.frame_w =800,800  
        self.ssta_control.frame_h ,self.ssta_control.frame_w = 800,800

        self.random_agent_generate_time=100 #change to generate varied agents at shorter or longer time
        self.ssta_agent_percentage=10 #percentage of total generated agents as ssta from all agents
        self.generate_flag=True


    def generate_agents(self,timer):
        ### The section takes the existinf ssta and apf agents and generates new agents 
        if timer%self.random_agent_generate_time==0 and self.generate_flag:
            self.car_pos=np.vstack([self.apf_car_pos,self.ssta_car_pos])
            self.goal_pos=np.vstack([self.apf_goal_pos,self.ssta_goal_pos[:,:2]])
            new_agents=self.infinity.run_simulation(self.car_pos,(self.goal_pos[:,:2]).astype(np.int32))# important that goal points passed in must be global and of int 32

            ###This function splits the new generated points as ssta and apf based on percentage
            apf_new_agents,ssta_new_agents=self.split_agents(new_agents)
            ###
            ###combines the apf newly generated points
            self.apf_control.create_agents(apf_new_agents)
            self.apf_car_pos = self.apf_control.x
            self.apf_goal_pos = self.apf_control.goal_pos

            ###combines the ssta newly generated points
            self.ssta_control.create_agents(ssta_new_agents)
            self.ssta_car_pos = self.ssta_control.x
            self.ssta_goal_pos = self.ssta_control.goal_pos  
        ###

        #combine all of apf and ssta agents

        all_agents=self.apf_car_pos
        if  self.enable_ssta_agents:all_agents=np.vstack([self.apf_car_pos,self.ssta_car_pos])
        ##finding the agent-agent replsuion for all agents split later
        all_agents_potential,all_agents_distance=self.apf_control.avoid_agents(9, 60000, all_agents)#100000  
        
        #Splitting agent potential based on apf and ssta
        self.apf_control.apf_agent_potential,self.apf_control.apf_agent_distance = all_agents_potential[:self.apf_car_pos.shape[0]],all_agents_distance[:self.apf_car_pos.shape[0]] 
        if  self.enable_ssta_agents:self.ssta_control.ssta_agent_potential,self.ssta_control.ssta_agent_distance = all_agents_potential[self.apf_car_pos.shape[0]:],all_agents_distance[self.apf_car_pos.shape[0]:] 

        #updating the pos for ssta and apd seperately      
        self.apf_car_pos,self.apf_goal_pos = self.apf_control.car_pos()
        if  self.enable_ssta_agents:self.ssta_car_pos,self.ssta_goal_pos=self.ssta_control.car_pos()  

        if timer%100==0:
            if  self.enable_ssta_agents:print("Total APF agents:",self.apf_car_pos.shape[0],"Total SSTA_apf agents:",self.ssta_control.apf_agents.shape[0],"Total SSTA_SSTA agents:",self.ssta_control.ssta_agents.shape[0])



    def generate_apf_agents(self):
        return  self.apf_car_pos,self.apf_goal_pos

    def generate_ssta_agents(self):
        return  self.ssta_car_pos,self.ssta_goal_pos
    
    def split_agents(self,new_agents):
        # This function splits the ssta and apf newly generated points based on percentage

        if self.enable_ssta_agents==False: self.ssta_agent_percentage = 0

        apf_new_agents,ssta_new_agents={},{}

        start_points=new_agents["start"]
        goal_points=new_agents["goal"]

        ssta_percent_rows = int((self.ssta_agent_percentage/100) * start_points.shape[0])
        ssta_start_points=start_points[:ssta_percent_rows]
        #ssta set the shape and color
        ssta_start_points[:,4:7]= 6, 15, -1 #color,radius,shape
        ssta_goal_points=goal_points[:ssta_percent_rows]

        apf_start_points=start_points[ssta_percent_rows:]
        apf_goal_points=goal_points[ssta_percent_rows:]

        ssta_goal_array = np.full((ssta_goal_points.shape[0],self.ssta_goal_pos.shape[1]), None)
        ssta_goal_array[:, :2] = ssta_goal_points

        apf_new_agents["start"]=apf_start_points
        apf_new_agents["goal"]=apf_goal_points 
        ssta_new_agents["start"]=ssta_start_points
        ssta_new_agents["goal"]=ssta_goal_array
        # print(apf_start_points.shape,apf_goal_points.shape,ssta_start_points.shape,ssta_goal_array.shape)
        return apf_new_agents,ssta_new_agents


    def apf_intersections(self):
        return self.apf_control.intersection()
    def ssta_intersections(self):
        return self.ssta_control.intersection()
    
    def traffic_speed(self):
        average_speed=self.apf_control.traffic_speed()
        if  self.enable_ssta_agents:average_speed=(self.ssta_control.traffic_speed()+self.apf_control.traffic_speed())/2.0
        if np.isnan(self.ssta_control.traffic_speed()):average_speed=self.apf_control.traffic_speed()
        if np.isnan(self.apf_control.traffic_speed()):average_speed=self.ssta_control.traffic_speed()
        return average_speed
    
    def volume_capacity(self):
        vol,cap=self.apf_control.volume_capacity()
        if  self.enable_ssta_agents:vol=self.ssta_control.volume_capacity()[0]+self.apf_control.volume_capacity()[0]
        return vol,cap
    def collison_rate(self):
        col_rate,total_time=self.apf_control.collison_rate()
        if  self.enable_ssta_agents:col_rate=(self.apf_control.collison_rate()[0]+self.ssta_control.collison_rate()[0])/2.0
        return col_rate,total_time
    