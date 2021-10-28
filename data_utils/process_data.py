import os 
import sys
import glob
import open3d as o3d
import numpy as np
from PIL import ImageColor
from data_utils import *

# change these 4 variables fitting to the point cloud data loading from CloudCompare
# data_utils.x_shift = 6052570
# data_utils.y_shift = 2179450
# data_utils.z_shift = 0
# data_utils.z_max = 106.68

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

if __name__ == '__main__':
    SAVED_DIR = './data/original_pcd'
    PER_SEC_DATA = 20                   # group point cloud data per 40s into 1 set of point cloud data
    K_POINTS = 480      # 240
    # NB_NEIGHBORS = 5
    # STD_RATIO = 4.0
    VOXEL_SIZE = 0.3

    # print('Reading point cloud from Velodyne file')
    # count = 1
    # pcd_size = 0
    # for velo_file in glob.glob('./data/Velodyne_NP500*.txt'):
    #     datas = {}
    #     read_velodyne(velo_file, datas, filter=True, type='pts')
        
    #     # datas are now organized in term of second
    #     data = []
    #     for id, key in enumerate(datas.keys()):
    #         pcd_size += len(datas[key])
    #         data = data + datas[key]
        
    #     print(f'Writing point cloud data into pts format for set {count}')
    #     print(f'POINT COUNTS: {pcd_size}')
    #     write_pcd(SAVED_DIR, 'pcd_' + str(count) + '.txt', data, type='pts')
    #     count += 1
    #     pcd_size = 0       

    # print('Finished parsing data!!!')

    # read the newly-parsed point cloud data
    for file in glob.glob(os.path.join(SAVED_DIR, 'pcd_*.txt')):  
        pcd = o3d.io.read_point_cloud(file, format='pts')
        # o3d.visualization.draw_geometries([pcd])

        file_name = os.path.basename(file)
        print(f'Down-sampling {file_name}')

        # ------------------------------ Down Sampling Point Cloud Data ------------------------------
        print(f'Performing Voxel DownSampling with voxel_size = {VOXEL_SIZE}')
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        print(len(voxel_down_pcd.points))
        # o3d.visualization.draw_geometries([voxel_down_pcd])

        write_xyzrgb(SAVED_DIR, file_name[:-4] + '_ds.txt', voxel_down_pcd.points, voxel_down_pcd.colors)
        write_pcd_xyzrgb(SAVED_DIR, file_name[:-4] + '_ds.pcd', voxel_down_pcd.points, voxel_down_pcd.colors)

        # print(f'Performing Uniform DownSampling every {K_POINTS}-th points')
        # uni_down_pcd = pcd.uniform_down_sample(every_k_points=K_POINTS)
        # print(len(uni_down_pcd.points))

        # print(f'Performing Voxel DownSampling with voxel_size = {VOXEL_SIZE}')
        # voxel_down_pcd = uni_down_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        # print(len(voxel_down_pcd.points))
        # o3d.visualization.draw_geometries([voxel_down_pcd])

        # # The final Down-sample cloud point data
        # final_down_pcd = voxel_down_pcd
        # print(len(final_down_pcd.points))
        # o3d.visualization.draw_geometries([final_down_pcd])

        # exit(0)
        # write_pcd_xyzrgb(SAVED_DIR, file_name[:-4] + '_ds.pcd', final_down_pcd.points, final_down_pcd.colors)

    # # read the new point cloud data
    # pcd = o3d.io.read_point_cloud("./data/original_pcd/Velodyne_NP5001_ori.txt", format='pts')

    # # ----------------------- Down Sampling Methods -----------------------
    # print("Downsample the point cloud with a voxel of 1.0")
    # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=1.5)
    # print(len(voxel_down_pcd.points))
    # # o3d.visualization.draw_geometries([voxel_down_pcd])
    # write_pcd_xyzrgb('./data/original_pcd', 'Velodyne_NP5001_ds.pcd', voxel_down_pcd.points, voxel_down_pcd.colors)

    # Uniform Down sample every n-th points
    # print("Every 30th points are selected")
    # uni_down_pcd = pcd.uniform_down_sample(every_k_points=30)
    # print(len(uni_down_pcd.points))
    # o3d.visualization.draw_geometries([uni_down_pcd])
    # o3d.visualization.draw_geometries([uni_down_pcd])
    # o3d.io.write_point_cloud("./data/original_pcd/Velodyne_NP5001_30thds.pcd", uni_down_pcd, write_ascii=True)
    # write_pcd_xyzrgb('./data/original_pcd', 'Velodyne_NP5001_30thds.pcd', uni_down_pcd.points, uni_down_pcd.colors)

    # # print("Radius oulier removal")
    # # cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    # # display_inlier_outlier(voxel_down_pcd, ind)

    







    # read_velodyne('./data/Velodyne_NP5001.txt', datas, filter=True, type='pts')
    # read_velodyne('./data/Velodyne_NP5002.txt', datas, filter=True, type='pts')
    # read_velodyne('./data/Velodyne_NP5003.txt', datas, filter=True, type='pts')
    # read_velodyne('./data/Velodyne_NP5004.txt', datas, filter=True, type='pts')
    # read_velodyne('./data/Velodyne_NP5005.txt', datas, filter=True, type='pts')
    # read_velodyne('./data/Velodyne_NP5006.txt', datas, filter=True, type='pts')

    # # datas are now organized in term of second
    # pcd_size = 0
    # count = 0
    # data = []
    # for id, key in enumerate(datas.keys()):
    #     if id % PER_SEC_DATA == 0 and id != 0:
    #         print(f'Writing point cloud data into pts format for set {count}')
    #         print(f'POINT COUNTS: {pcd_size}')
    #         write_pcd(SAVED_DIR, 'pcd_' + str(count) + '.txt', data, type='pts')

    #         # reset data for the next set
    #         count += 1
    #         pcd_size = 0
    #         data = []
        
    #     pcd_size += len(datas[key])
    #     data = data + datas[key]
