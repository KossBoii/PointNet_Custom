import h5py
import os
import numpy as np
import glob
import sys

def readPCD(file_name):
    pcd_file = open(file_name, 'r')
    for line in pcd_file:
        if line.startswith('DATA'):
            break
    
    cloud_pts = []
    for line in pcd_file:
        temp = line.split()
        assert(len(temp) <= 9)
        x = float(temp[0])
        y = float(temp[1])
        z = float(temp[2])
        r = int(temp[3])
        g = int(temp[4])
        b = int(temp[5])
        intensity = int(temp[6])
        label = int(temp[7])
        object = int(temp[8])

        cloud_pts.append([x,y,z])
    pcd_file.close()
    return np.array(cloud_pts)

def normalize(cloud_pts):
    centroid = cloud_pts.mean(axis=0)
    cloud_pts -= centroid
    R = np.sum(cloud_pts**2, axis=1)
    cloud_pts /= np.sqrt(np.max(R))
    return cloud_pts

if __name__ == '__main__':
    data = []
    label = []

    

# def write_pyh5_dataset(path):
#     for file in glob.iglob(path + '*.pcd'):
#         pass

# class MyCustomDataset():
#     def __init__(self, path, num_pts, train) -> None:
#         self.path = path
#         self.num_pts = num_pts

#         if train:
#             data_file = 'data/test_project/train_files.txt'
#         else:
#             data_file = 'data/test_project/test_files.txt'

#     def __len__(self):
#         return self.label.shape[0]
    
#     def __getitem__(self, index):
#         return self.pointcloud[index], self.label[index]







