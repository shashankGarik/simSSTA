import numpy as np

class DoubleIntegrator:
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
        self.agent_collision=np.array([False]*self.x.shape[0])
        self.frame_h=800
        self.frame_w=800
        

        self.A = np.array([[0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        
        self.B = np.array([[0, 0],
                           [0, 0],
                           [1, 0],
                           [0, 1]])
        
        # proportional gains
        self.Kp_1= np.array([[2.0, 0],
                             [0, 2.0]])
        
        # derivative gains
        self.Kd_1 = np.array([[10, 0],
                              [0, 10]])

    def step(self):
        
        self.frame_agents()
        self.total_time+=self.dt
        obstacle_potential = 0
        rectangle_distance,circle_distance,agent_distance=None,None,None

        if len(self.obs_circle) != 0:
            c_obstacle_potential,circle_distance = DoubleIntegrator.avoid_circle(200, 500000, self.obs_circle, self.x)     
            obstacle_potential=c_obstacle_potential+obstacle_potential

        if len(self.obs_rectangle) != 0:
            r_obstacle_potential,rectangle_distance,self.intersections = DoubleIntegrator.avoid_rectangle(80, 80000, self.obs_rectangle, self.x)
            obstacle_potential=r_obstacle_potential+obstacle_potential

        error = self.goal_pos - self.x[:,:2]
        self.dist2goal = np.linalg.norm(error, axis = 1)
    
        goal_close_idx = np.argwhere(self.dist2goal <= 100)
        goal_reached_idx=np.argwhere(self.dist2goal <= 5.0)

        v_error = 5-self.x[:,2:4]

        prop_potential = np.zeros((self.x[:,:4].shape[0],2))
        diff_potential = np.zeros((self.x[:,:4].shape[0],2))

        agent_potential,agent_distance = DoubleIntegrator.avoid_agents(90, 60000, self.x)#100000
        prop_potential = np.squeeze(np.dot(self.Kp_1[np.newaxis, :,:], error[:,:,np.newaxis])).T
        prop_potential = self.desired_force(1000)

        diff_potential = np.squeeze(np.dot(self.Kd_1[np.newaxis,:,:], v_error[:,:,np.newaxis])).T

        if len(goal_close_idx) > 0:
            agent_potential[goal_close_idx] = 0.0
            if type(obstacle_potential) == np.ndarray:
                obstacle_potential[goal_close_idx] = np.array([0.0,0.0])

        control_input = prop_potential + diff_potential + obstacle_potential + agent_potential

        A_x = np.squeeze(np.dot(self.A[np.newaxis,:,:], self.x[:,:4,np.newaxis]))
        B_u = np.squeeze(np.dot(self.B[np.newaxis,:,:], control_input[:,:, np.newaxis]))
        v = (A_x + B_u).T

        #XXXX collision detetctionXXX
        #this line to collide show and go
        
        self.agent_collision=np.array([False]*self.x.shape[0])

        v=self.collision_detection(v,agent_distance,rectangle_distance,circle_distance,goal_reached_idx)
        if self.x.shape[1] > 4:
            self.x = np.hstack([self.x[:,:4] + self.dt * v,self.x[:,4:]])
        else:
            self.x = self.x[:,:4] + self.dt * v

        self.remove_agent_goal()
        self.frame_agents()
    
    def create_agents(self, new_agents):
        self.x = np.concatenate([self.x, new_agents["start"]])
        self.goal_pos = np.concatenate([self.goal_pos, new_agents["goal"]])
        self.agent_collision=np.concatenate([self.agent_collision,[False]*len(new_agents["start"])])


    # This function takes the local points and checks the points inside the bounding box
    def remove_agent_goal(self):
        # print(self.dist2goal)
        indices_less_than_5 = np.where(self.dist2goal < 5.5)[0]
        mask_to_keep = np.isin(np.arange(len(self.x)),indices_less_than_5 , invert=True)
        # Filter the array to keep only the desired elements
        self.x = self.x[mask_to_keep]
        self.goal_pos = self.goal_pos[mask_to_keep]
        self.agent_collision=self.agent_collision[mask_to_keep]
    
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
    
    def car_pos(self):
        # print(len(self.x))
        if len(self.x)!=None:
            self.step()
        return  self.x,self.goal_pos
    
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
    
    def collision_detection(self,v,agent_distance,rectangle_distance,circle_distance,goal_reached_idx):
        self.agent_collision=DoubleIntegrator.collision_analysis(self.agent_collision,agent_distance,rectangle_distance,circle_distance)
        self.agent_collision[self.outside_frame_x_indices]=False
        # print(self.agent_collision.shape)
        self.true_indices_agent_collision = np.where(self.agent_collision)
        if len(v.shape)<2:
            v=v.reshape(1,v.shape[0])
        #collision stop-uncomment below line if you want collided agents to stop
        # v[self.true_indices_agent_collision]=0.0,0.0,0.0,0.0
        #stop when goal is reached
        v[goal_reached_idx]=0.0,0.0,0.0,0.0
        return v
    
    def desired_force(self, strength):
        error = self.goal_pos - self.x[:,:2]
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
    
    def avoid_agents(radii, strength, x_c):
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
        if len(x_c) == 1:
            return np.array([[0.0]]),None
        diff = x_c[:,:2][:,np.newaxis,:] - x_c[:,:2][np.newaxis,:,:]
        dist = np.squeeze(np.linalg.norm(diff, axis = 2)[:,:,np.newaxis])
        np.fill_diagonal(dist, np.inf)
        dir = diff/dist[:,:,np.newaxis]
        potential = (1.0 / dist - 1.0 / radii)
        potential[potential < 0] = 0
        collision_potentials = strength * (potential)
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
