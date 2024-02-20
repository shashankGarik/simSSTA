import itertools
import heapq

class Astar:
    def __init__(self,map):
        self.map = map
    def get_neighbors(self, state, max_boundary = 300):
        x,y = state
        p_neighbors = [(x-1,y+1),(x,y+1),(x+1,y+1),
                       (x-1,y),          (x+1,y),
                       (x-1,y-1),(x,y-1),(x+1,y-1)]
        neighbors = []    

        for x_p,y_p in p_neighbors:

            if (0 <= x_p <= max_boundary-1 and 0 <= y_p <= max_boundary-1): 
                if map.T[x_p,y_p] != 1:
                    neighbors.append((x_p,y_p))
        return neighbors
    
    def run_search(self, start_state, goal_state):

        frontier = PriorityQueue()
        visited = set()
        frontier.insert((start_state,[start_state],0),0)

        while frontier.elements:
            # print(frontier.elements)
            (curr_state, curr_path, cost), _ = frontier.pop()

            if curr_state not in visited:

                if curr_state == goal_state:
                    path = curr_path.copy()
                    path.append(curr_state)
                    return path
                visited.add(curr_state)

                neighbors = self.get_neighbors(curr_state)
                for n in neighbors:
                    temp_path = curr_path.copy()
                    temp_path.append(n)
                    ind_cost = (1 if self.map.T[n[0],n[1]] == 0 else self.map.T[n[0],n[1]])
                    temp_cost = cost + ind_cost
                    h_cost = manhattan_dist(n, goal_state) 

                    if n not in visited:
                        frontier.insert((n,temp_path, temp_cost),temp_cost + h_cost)

        return None


class PriorityQueue:
    def __init__(self):
        self.elements = []
        self.counter = itertools.count()

    def empty(self):
        return len(self.elements) == 0

    def insert(self, item, priority):
        count = next(self.counter)
        heapq.heappush(self.elements, (priority, count, item))

    def pop(self):
        out = heapq.heappop(self.elements)
        return out[2],out[0]
    
def manhattan_dist(p1,p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def add_cost(binary_map, cost = 2 , radius = 15):
    kernel = np.ones((radius, radius), np.uint8) 
    img_dilation = cv2.dilate(map, kernel, iterations=1) 
    wheres = binary_map - img_dilation
    binary_map[wheres != 0] = cost
    return binary_map
