import heapq
import itertools

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

def euclidean_dist(p1,p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

