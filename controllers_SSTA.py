import numpy as np
import utils

class DoubleIntegratorSSTA:
    """
    acceleration-based PD control using double integrator mode. Multi agent-multi obstacle, 
    Avoids obstacles, and avoids other agent.
    """
    def __init__(self, x_init, goal_pos, obstacles):
        self.x = x_init
        self.camera_x=None
        self.goal_pos = goal_pos
        self.obs_circle = obstacles['circle']
        self.obs_rectangle = obstacles['rectangle']
        self.dt = 0.05
        self.total_time=0.0
        self.total_collision=0.0
        # self.agent_collision=np.array([False]*self.x.shape[0])
        self.frame_h=800
        self.frame_w=800
        self.combined_camera_indices=[(np.array([], dtype=np.int64),)]
        self.global_agent_paths=None
        self.path_indices= np.array(self.x.shape[0]*[0])

        

        self.A = np.array([[0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        
        self.B = np.array([[0, 0],
                           [0, 0],
                           [1, 0],
                           [0, 1]])
        
        # proportional gains
        self.apf_Kp= np.array([[1.5, 0],
                             [0, 1.5]])
        
        self.ssta_Kp= np.array([[5, 0],
                             [0, 5]])
        
        # derivative gains
        self.apf_Kd = np.array([[10, 0],
                              [0, 10]])
        
        self.ssta_Kd = np.array([[12, 0],
                              [0, 12]])

    def step_apf(self): 
        self.frame_agents()
        
        self.total_time+=self.dt
        obstacle_potential = 0
        rectangle_distance,circle_distance,agent_distance=None,None,None

        if len(self.obs_circle) != 0:
            c_obstacle_potential,circle_distance = DoubleIntegratorSSTA.avoid_circle(200, 500000, self.obs_circle, self.apf_agents)     
            obstacle_potential=c_obstacle_potential+obstacle_potential

        if len(self.obs_rectangle) != 0:
            r_obstacle_potential,rectangle_distance,self.intersections = DoubleIntegratorSSTA.avoid_rectangle(80, 80000, self.obs_rectangle, self.apf_agents)
            obstacle_potential=r_obstacle_potential+obstacle_potential

        
        error = (( self.apf_agents_goal_pos[:,:2]).astype(np.int32)) - self.apf_agents[:,:2]
        dist2goal = np.linalg.norm(error, axis = 1)
    
        goal_close_idx = np.argwhere(dist2goal <= 100)
        goal_reached_idx=np.argwhere(dist2goal <= 5.0)

        v_error = 5-self.apf_agents[:,2:4]

        prop_potential = np.zeros((self.apf_agents[:,:4].shape[0],2))
        diff_potential = np.zeros((self.apf_agents[:,:4].shape[0],2))

        agent_potential,agent_distance = DoubleIntegratorSSTA.avoid_agents(9, 60000, self.apf_agents)#100000
        prop_potential = np.squeeze(np.dot(self.apf_Kp[np.newaxis, :,:], error[:,:,np.newaxis])).T
        prop_potential = self.apf_desired_force(1000)

        diff_potential = np.squeeze(np.dot(self.apf_Kd[np.newaxis,:,:], v_error[:,:,np.newaxis])).T

        if len(goal_close_idx) > 0:
            agent_potential[goal_close_idx] = 0.0
            if type(obstacle_potential) == np.ndarray:
                obstacle_potential[goal_close_idx] = np.array([0.0,0.0])

        control_input = prop_potential + diff_potential + obstacle_potential + agent_potential
        A_x = np.squeeze(np.dot(self.A[np.newaxis,:,:], self.apf_agents[:,:4,np.newaxis]))
        B_u = np.squeeze(np.dot(self.B[np.newaxis,:,:], control_input[:,:, np.newaxis]))
        v = (A_x + B_u).T

        #XXXX collision detetctionXXX
        #this line to collide show and go
        
        self.apf_agent_collision=np.array([False]*self.apf_agents.shape[0])
        v=self.collision_detection(v,agent_distance,rectangle_distance,circle_distance)
        v=self.terminate_agent_movement(v,goal_reached_idx)

        if self.apf_agents.shape[1] > 4:
            self.apf_agents = np.hstack([self.apf_agents[:,:4] + self.dt * v,self.apf_agents[:,4:]])
        else:
            self.apf_agents = self.apf_agents[:,:4] + self.dt * v
        

        

    def step_ssta(self): 
        
        print("XXXXpathindicesXXXX",self.path_indices)
        
        reset_path_indices=np.argwhere(self.path_indices==self.replanning_index)
        self.path_indices[reset_path_indices]=0

        self.global_agent_paths[:,-1]=self.ssta_agents_goal_pos[:,:2]

        self.camera_indices_global_path=self.global_agent_paths[self.combined_camera_indices]
        
        self.dummy=self.path_indices[self.combined_camera_indices]
        self.camera_indices_global_path= self.camera_indices_global_path[np.arange(self.camera_indices_global_path.shape[0]),self.dummy].reshape(-1,2)
        error = ((self.camera_indices_global_path).astype(np.int32)) -self.ssta_agents[:,:2]

        dist2goal = np.linalg.norm(error, axis = 1)
        v_error = 5-self.ssta_agents[:,2:4]
        prop_potential = np.zeros((self.ssta_agents[:,:4].shape[0],2))
        diff_potential = np.zeros((self.ssta_agents[:,:4].shape[0],2))
        prop_potential = np.squeeze(np.dot(self.ssta_Kp[np.newaxis, :,:], error[:,:,np.newaxis])).T
        prop_potential = self.ssta_desired_force(1000,self.camera_indices_global_path)
        diff_potential = np.squeeze(np.dot(self.ssta_Kd[np.newaxis,:,:], v_error[:,:,np.newaxis])).T
        control_input = prop_potential + diff_potential
        A_x = np.squeeze(np.dot(self.A[np.newaxis,:,:], self.ssta_agents[:,:4,np.newaxis]))
        B_u = np.squeeze(np.dot(self.B[np.newaxis,:,:], control_input[:,:, np.newaxis]))
        v = (A_x + B_u).T
  
        if self.ssta_agents.shape[1] > 4:
            self.ssta_agents = np.hstack([self.ssta_agents[:,:4] + self.dt * v,self.ssta_agents[:,4:]])
        else:
            self.ssta_agents = self.ssta_agents[:,:4] + self.dt * v
        
        # switching between goals
        mask_dist2goal=np.full((self.x.shape[0],2),np.inf)
        mask_dist2goal[self.ssta_indices]=dist2goal
        increase_goal=np.where(mask_dist2goal<=4)
        self.path_indices[increase_goal[0]]+=1
        # print(  "inside",self.path_indices)
        # print(self.camera_indices_global_path)

    def car_pos(self):
        ##split self.x as apf and ssta
        # split self.x as 
        self.frame_agents()
        self.apf_indices=np.argwhere(self.goal_pos[:,4]==None).flatten()
        self.ssta_indices=np.argwhere(self.goal_pos[:,4]!=None).flatten()
        
        self.apf_agents=self.x[self.apf_indices]
        self.apf_agents_goal_pos=self.goal_pos[self.apf_indices]

        self.ssta_agents=self.x[self.ssta_indices]
        self.ssta_agents_goal_pos=self.goal_pos[self.ssta_indices]

        self.step_apf()
        if self.ssta_agents.shape[0]!=0:
            self.step_ssta()
        self.agent_collision=np.array([False]*self.x.shape[0])

        self.agent_collision[self.apf_indices]=self.apf_agent_collision
        self.x[self.apf_indices]=self.apf_agents
        self.x[self.ssta_indices]=self.ssta_agents
        # print("XXXXXXXXXXXXXXXXXXXXXXXX",len(self.x),len(self.ssta_agents),len(self.apf_agents))
        self.frame_agents()
        self.remove_agent_goal()
        return  self.x,self.goal_pos
    
    def create_agents(self, new_agents):
        self.x = np.concatenate([self.x, new_agents["start"]])
        self.goal_pos = np.concatenate([self.goal_pos, new_agents["goal"]])
        self.agent_collision=np.concatenate([self.agent_collision,[False]*len(new_agents["start"])])


    # This function takes the local points and checks the points inside the bounding box
    def remove_agent_goal(self):
        self.x_error = (( self.goal_pos[:,:2]).astype(np.int32)) - self.x[:,:2]
        self.dist2goal = np.linalg.norm(self.x_error, axis = 1)
        indices_less_than_5 = np.where(self.dist2goal < 5.5)[0]
        mask_to_keep = np.isin(np.arange(len(self.x)),indices_less_than_5 , invert=True)
        # Filter the array to keep only the desired elements
        self.x = self.x[mask_to_keep]
        self.goal_pos = self.goal_pos[mask_to_keep]
        self.agent_collision=self.agent_collision[mask_to_keep]
        self.path_indices=self.path_indices[mask_to_keep]
    
    def collison_rate(self):
        collision_in_frame=self.agent_collision[self.frame_x_indices]
        # print(collision_in_frame)
        indices_agent_collision = np.where(collision_in_frame)
        count_collisons = np.count_nonzero( indices_agent_collision)
        self.total_collision+=count_collisons
        return (self.total_collision/self.total_time),self.total_time
   
    def intersection(self):
        return self.intersections
    
    def volume_capacity(self):
        self.capacity=min(self.frame_h/3,self.frame_w/3)
        #volume inside the frame
        x_row ,y_row = self.x[:, 0], self.x[:, 1]
        volume_condition = np.logical_and(x_row >= 0, x_row <= self.frame_w) & np.logical_and(y_row >= 0, y_row <= self.frame_h)
        self.volume = np.sum(volume_condition)
     
        return self.capacity,self.volume
    
    def collision_status(self):
        return self.agent_collision
    
    def frame_agents(self):
        x_row ,y_row = self.x[:, 0], self.x[:, 1]
        volume_condition = np.logical_and(x_row >= 0, x_row <= self.frame_w) & np.logical_and(y_row >= 0, y_row <= self.frame_h)
        self.frame_x_indices=np.where(volume_condition)
        self.frame_x=self.x[self.frame_x_indices]
        self.outside_frame_x_indices=np.where(volume_condition==False)
        self.outside_frame_x=self.x[self.outside_frame_x_indices]


    def traffic_speed(self):
        v_x ,v_y = self.frame_x[:, 2], self.frame_x[:, 3]
        velocity_magnitude = np.sqrt(v_x**2 + v_y**2)
        speed=np.average(velocity_magnitude)
        return speed
    
    def collision_detection(self,v,agent_distance,rectangle_distance,circle_distance):
        self.apf_agent_collision=DoubleIntegratorSSTA.collision_analysis(self.apf_agent_collision,agent_distance,rectangle_distance,circle_distance)
        # print( self.apf_agent_collision,self.outside_frame_x_indices)
        # self.apf_agent_collision[self.outside_frame_x_indices]=False
        # print(self.agent_collision.shape)
        # self.true_indices_agent_collision = np.where(self.apf_agent_collision)
        if len(v.shape)<2:
            v=v.reshape(1,v.shape[0])
        #collision stop-uncomment below line if you want collided agents to stop
        # v[self.true_indices_agent_collision]=0.0,0.0,0.0,0.0
        #stop when goal is reached
        return v
    
    #stop when goal is reached 
    def terminate_agent_movement(self,v,goal_reached_idx):
        v[goal_reached_idx]=0.0,0.0,0.0,0.0
        return v
    
    def apf_desired_force(self, strength):
        error = (self.apf_agents_goal_pos[:,:2].astype(np.int32)) - self.apf_agents[:,:2]
        dist2goal = np.linalg.norm(error, axis = 1)
        dir = error/dist2goal[:, np.newaxis]
        return dir*strength
    
    def ssta_desired_force(self, strength,path_point):
        error = (path_point.astype(np.int32)) - self.ssta_agents[:,:2]
        dist2goal = np.linalg.norm(error, axis = 1)
        dir = error/dist2goal[:, np.newaxis]
        return dir*strength
 
    @staticmethod
    def avoid_circle(radii, strength, obs_c, x_c):
        """
        Caluculates the potential for all obstacles with respect to the agent

        Args:
            radii (float or int or np.ndarray): vector/number representing radius 
                of each obstacle. if ndarray, shape must be (m,1).
            strength (float or int or np.ndarray): vector/number representing repulsive
                strength of each obstacle. if ndarray, shape must be (m,1).
            obs_c (np.ndarray): numpy array of obstacle centroids of shape (m,2).
            x_c (np.ndarray): numpy array of obstacle centroids of shape (n,4).

        Returns:
            np.ndarray: combined resulting potential of obstacles for each agent of shape (n,2)
        """
        diff = x_c[:,:2][:,np.newaxis] - obs_c[np.newaxis,:,:]
        dist = np.linalg.norm(diff, axis = 2)[:,:,np.newaxis] + (1e-6)
        dir = (diff)/dist
        mag = (1/dist) - (1/radii)
        mag[mag < 0] = 0
        obstacle_potential = np.sum(dir*mag*strength, axis = 1)
        return obstacle_potential,dist
    
    def avoid_rectangle(radii, strength, rectangle, x_c):
        x,y,w,h = rectangle[:,0], rectangle[:,1], rectangle[:,2], rectangle[:,3]

        bl = np.array([x - w/2, y - h/2]).T
        tr = np.array([x + w/2, y + w/2]).T
        max_pass = np.maximum(bl[:,np.newaxis,:], x_c[:,:2][np.newaxis,:,:])
        intersections = np.minimum(max_pass,tr[:,np.newaxis,:])
        diff = x_c[:,:2][np.newaxis,:,:] - intersections
        dist = np.linalg.norm(diff, axis = 2)[:,:,np.newaxis] + (1e-6)
        dir = diff/dist
        mag = (1/dist) - (1/radii)
        mag[mag < 0] = 0
        obstacle_potential = np.sum(dir*mag*strength, axis = 0)
        return obstacle_potential,dist,intersections
    
    def avoid_agents(m_factor, strength, x_c):
        """
        Calculates the potential for all agents with respect to other agents

        Args:
            radii (float or int or np.ndarray): vector/number representing radius 
                of each obstacle. if ndarray, shape must be (n,1).
            strength (float or int or np.ndarray): vector/number representing repulsive
                strength of each agent. if ndarray, shape must be (n,1).
            x_c (np.ndarray): numpy array of obstacle centroids of shape (n,4).

        Returns:
            np.ndarray: combined resulting potential of obstacles for each agent of shape (n,2)
            dist:distance between each and every agent
        """
        radii = x_c[:,5]
        strength_factor = radii/10
        radii = radii*m_factor

        if len(x_c) == 1:
            return np.array([[0.0]]),None
        diff = x_c[:,:2][:,np.newaxis,:] - x_c[:,:2][np.newaxis,:,:]
        dist = np.squeeze(np.linalg.norm(diff, axis = 2)[:,:,np.newaxis])
        np.fill_diagonal(dist, np.inf)
        dir = diff/dist[:,:,np.newaxis]
        potential = (1.0 / dist - 1.0 / radii)
        potential[potential < 0] = 0
        collision_potentials = strength * (potential) * strength_factor
        agent_potential = np.sum(collision_potentials[:,:,np.newaxis]*dir, axis = 1)

        return agent_potential,dist

    def collision_analysis(agent_collision,agent_dist,rect_dist,circ_distance):
        threshold_rect_obstacle =10
        threshold_circle_obstacle=30
        threshold_agent=20
        obstacle_logic=[agent_collision]
        if rect_dist is not None: 
            rectan_dist = np.squeeze(rect_dist, axis=-1)
            rectan_dist = np.transpose(rectan_dist, ( 1, 0))
            rectangle_collision= np.any(rectan_dist < threshold_rect_obstacle, axis=1)
            obstacle_logic.append(rectangle_collision)
        if circ_distance is not None:
            circ_dist=np.transpose(circ_distance, (1, 0, 2))
            circ_dist=np.squeeze(circ_dist, axis=-1)
            circle_collision= np.any(circ_dist < threshold_circle_obstacle, axis=0)
            obstacle_logic.append(circle_collision)
        # Check if any element in each row is less than the threshold
        if agent_dist is not None:
            a_collision= np.any(agent_dist < threshold_agent, axis=1)
            obstacle_logic.append(a_collision)
        collision= np.logical_or.reduce(obstacle_logic)
        return collision
    

    class controlSSTA:
        def __init__(self, T2NO_dirs, ):
            pass
        def _compute_paths(self, t2nod, start_pos, goal_pos):
            pass
        def path2global(self, paths, tranforms):
            pass
        
