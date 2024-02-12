import numpy as np

# Case 1:
start = np.array([[50.0, 300.0, 0.0, 0.0],[300.0, 50.0, 0.0, 0.0]])
goal = np.array([[800, 800],[700, 700]])
obs = {'circle': np.array([[200,500]]),
       'rectangle': np.array([[600.0, 500.0,80.0,80.0],[500.0, 350.0,160.0,160.0]])}

# # Case 2:
start_left = [np.array([50.0,i,0.0,0.0]) for i in range(50,950,200)]
start_right = [np.array([950,i,0.0,0.0]) for i in range(150,950,200)]
start_top = [np.array([i,50.0,0.0,0.0]) for i in range(150,950,200)]
start_bottom = [np.array([i,950.0,0.0,0.0]) for i in range(150,950,200)]

goal_left = np.array([np.array([950.0,i]) for i in range(50,950,200)])
goal_right = np.array([np.array([50.0,i]) for i in range(150,950,200)])
goal_top = np.array([np.array([i,950.0]) for i in range(150,950,200)])
goal_bottom = np.array([np.array([i,50.0]) for i in range(150,950,200)])
# print("hi")
start = np.concatenate([start_left, start_right, start_top, start_bottom])

goal = np.concatenate([goal_left,goal_right, goal_top, goal_bottom])
obs = {'circle': np.array([]),
       'rectangle': np.array([[500,500,30,30],
                              [700,700,30,30],
                              [300,300,30,30],
                              [700,300,30,30],
                              [300,700,30,30],
                              [300,500,30,30],
                              [500,300,30,30],
                              [700,500,30,30],
                              [500,700,30,30]])}

# Case 3

start = np.array([])
goal = np.array([])
obs = {'circle': np.array([]),
       'rectangle': np.array([])}

points = []

for x in range(50,975,120):
    for y in range(50,975,120):
        points.append(np.array([x,y]))

points = np.array(points)

while len(points) > 1:

    choice = np.random.choice([0,1,2], p = [0.50,0.05,0.45]) # 0->agents, 1->empty space, 2->obs
    if choice == 0:
        start_point, goal_point = np.random.choice(len(points), 2, replace = False)
        # print(start_point)
        # print(start)
        if len(start) != 0:
            start = np.concatenate([start, np.array([[points[start_point][0], points[start_point][1],0,0]])])
            goal = np.concatenate([goal, np.array([points[goal_point]])])
        else:
            start = np.array([[points[start_point][0],points[start_point][1],0.0,0.0]])
            goal = np.array([points[goal_point]])

        points = points[np.setdiff1d(range(len(points)),[start_point, goal_point])]
        

    elif choice == 1:
        skip_point = np.random.choice(len(points), 1, replace = False)
        points = points[np.setdiff1d(range(len(points)),skip_point)]

    elif choice == 2:
        obs_point = np.random.choice(len(points), 1, replace = False)

        if len(obs['rectangle']) != 0:
            obs['rectangle'] = np.concatenate([obs['rectangle'],np.array([[points[obs_point][0][0], points[obs_point][0][1],30.0,30.0]])])
            
        else:
            # print(points[obs_point][0])
            obs['rectangle'] = np.array([np.array([points[obs_point][0][0], points[obs_point][0][1],30.0,30.0])])
        
        points = points[np.setdiff1d(range(len(points)),obs_point)]


# print(start)
# print(goal)
# print(len(obs['rectangle']))
# print(obs['rectangle'])



# goal = np.array([[800, 800],[700, 700]])
# obs = {'circle': np.array([[200,500]]),
#        'rectangle': np.array([[600.0, 500.0,80.0,80.0],[500.0, 350.0,160.0,160.0]])}
# start = np.array([[1080.76 ,1042.98  ,  0.  ,    0.  ],
#  [1064.54 ,1074.67 ,   0.  ,    0.  ]])
 