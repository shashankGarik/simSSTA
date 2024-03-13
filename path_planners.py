# This file is the planner file and the return format must always be a set number of m values for the path,
#If n agents are there then the path shape must be view * (n,m,2)- numpy arrays in a list where list contains agents of views seperated
import numpy as np

class Planners():
    def __init__(self):
        #testing
        self.ssta_replanning_factor=5
        self.testarray=[np.array([
                
                        [[57.01,0.63],
                        [100.16444444,10.24],
                        [50.01888889,90.96],
                        [200.87333333,160.60],
                        [70.72777778,190.50],
                        [124.58222222,196.18],
                        [149.43666667,219.48],
                        [174.29111111,249.82],
                        [199.14555556,274.26],
                        [224.,300.]],
                       ]),
                        np.array([
                        [[57.01,0.63],
                        [100.16444444,10.24],
                        [50.01888889,90.96],
                        [200.87333333,160.60],
                        [70.72777778,190.50],
                        [124.58222222,196.18],
                        [149.43666667,219.48],
                        [174.29111111,249.82],
                        [199.14555556,274.26],
                        [224.,300.]]
                        ])]
        # self.testarray=[np.array([
        #                  [[176.86443,3.72743243],
        #                  [256.24, 100.09],
        #                  [176.89, 200.45],
        #                  [50.45, 100.35],
        #                  [189.73, 200.08],
        #                  [200.02, 200.03],
        #                  [227.59, 150.91],
        #                  [50.10, 250.39],
        #                  [200.17, 100.34],
        #                  [300,193]]
        #                 ])]
        # self.testarray=[np.array([[[0., 0.], [50.84487633, 50.50066381], [60.94161902, 60.37797936], [70.03836171, 70.25529491], [82.1351044 , 80.13261046]]])]
                            # [[3.99750614, 8.40244838], [3.70858772, 6.58400636], [3.41966931, 4.76556435], [3.13075089, 2.94712233], [2.84183248, 1.12868032]],
                            # [[9.8946591 , 8.96850668], [9.20652376, 7.90496581], [8.51838841, 6.84142493], [7.83025307, 5.77788406], [7.14211773, 4.71434319]],
                            # [[4.53858008, 5.23847224], [4.70827311, 500.65541641], [4.87796614, 6.07236059], [5.04765917, 6.48930477], [5.2173522 , 6.90624895]]]
                            

    def a_star(self, ssta_agents_poses,ssta_camera_indices,path_size=10):
        # print(self.testarray[0].shape)
        #return as view,n,m,2 - this is as a list
        # print(len(self.testarray))
        # print(self.testarray[0].shape,self.testarray[1].shape)
        return self.testarray




