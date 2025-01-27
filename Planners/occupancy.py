import numpy as np

class OccupancyHelper:
    def __init__(self, t2no, t2nd, r, g_f, t_max_occ = 100):
        """
        args
        - t2no
        - t2nd
        - r: radius of the agent
        - g_f: grid factor used in combination with the radius to determine grid size
        - t_max_occ: max occupancy duration tracked by t2no or t2nd
        """
        self.scale = 1  # init the scaling to gridify
        self.t2no, self.t2nd = self.gridify(t2no, t2nd, r//g_f)
        self.t_max_occupancy = t_max_occ
        with np.errstate(divide='ignore'):
            self.cost_map = np.round(1/(self.t2no),3)
        

    def gridify(self, t2no, t2nd, grid_size):
        """
        args:
        - t2no: 128x128 image
        - t2nd: 1288x128 image
        - gridsize: int is taken in an t2no and t2nd are shrunk height/gridize, width/gridsize
        """
        if not t2no.shape == t2nd.shape:
            raise AssertionError("T2NO and T2ND must be the same size")
        
        h, w = t2no.shape
        pooled_h = h // grid_size
        pooled_w  = w // grid_size

        reshaped_t2no = t2no[:pooled_h*grid_size, :pooled_w*grid_size].reshape(pooled_h, grid_size, pooled_w, grid_size)
        reshaped_t2nd = t2nd[:pooled_h*grid_size, :pooled_w*grid_size].reshape(pooled_h, grid_size, pooled_w, grid_size)

        pooled_t2no = np.min(reshaped_t2no, axis=(1, 3))
        pooled_t2nd = np.max(reshaped_t2nd, axis=(1, 3))
        
        self.scale = t2no.shape[0]//pooled_t2no.shape[0]

        return pooled_t2no, pooled_t2nd
    
    def iterate(self, keep_max_occupany = False):
        """
        updates occupancy after timestamp
        """
        if keep_max_occupany:
            t2no_modify_logic = np.logical_and(self.t2no > 0, self.t2no < self.t_max_occupancy)
            self.t2no[t2no_modify_logic] -= 1
        else:
            self.t2no[self.t2no > 0] -= 1
            self.t_max_occupancy -= 1
            
        self.t2nd[self.t2nd > 0] -= 1

        self.t2no[self.t2nd == 0] == self.t_max_occupancy
        with np.errstate(divide='ignore'):
            self.cost_map = np.round(1/self.t2no,3)
    
    def get_state_cost(self, state):
        return self.cost_map[state[0], state[1]]
    
    def get_grid_neighbors(self, state):
        """
        args:
        - state: current pos (x,y)
        """
        max_boundary = self.t2no.shape[0]
        x,y = state
        p_neighbors = [(x-1,y+1),(x,y+1),(x+1,y+1),
                       (x-1,y),          (x+1,y),
                       (x-1,y-1),(x,y-1),(x+1,y-1)]
        
        neighbors = []
        for x_p,y_p in p_neighbors:
            if (0 <= x_p <= max_boundary-1 and 0 <= y_p <= max_boundary-1): 
                neighbors.append((x_p,y_p))
            
        return neighbors