from cv2 import IMREAD_ANYDEPTH
import open3d as o3d
import numpy as np
import cv2
from math import sqrt

K = np.array([[481.20, 0., 319.50, 0.],
                [0., -480.00, 239.50, 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]])

Kinv = np.linalg.inv(K)

def uvd_xyz(u, v, d):
    # f = sqrt(K[0][0]**2 + K[1][1]**2)
    # z = f*sqrt(d**2/
            #    (u**2 + v**2 + f**2))
    z = d/5000
    vector = Kinv@np.array([u, v, 1., 1/z]).transpose()*z
    return vector[:3]
    

color_raw = np.asarray(cv2.imread('annot.png'))
depth_raw = np.asarray(cv2.imread('depth.png', IMREAD_ANYDEPTH))

height = color_raw.shape[0]
width = color_raw.shape[1]

points = np.array([[0.]*3]*height*width)
colors = np.array([[0.]*3]*height*width)

for i in range(height):
    for j in range(width):
        points[width*i + j] = uvd_xyz(i, j, depth_raw[i][j])
        colors[width*i + j] = color_raw[i][j][::-1]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors/256)
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])