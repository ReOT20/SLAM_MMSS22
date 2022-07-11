from cv2 import IMREAD_ANYDEPTH
import open3d as o3d
import numpy as np
import cv2

ImageScaleFactor = 5000

CalibrationMatrix = np.array([[481.20, 0., 319.50, 0.],
                [0., -480.00, 239.50, 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]])

InvCalibrationMatrix = np.linalg.inv(CalibrationMatrix)

def uvd_xyz_array(data):
    height = data.shape[0]
    width = data.shape[1]
    data = data.reshape(height, width, 1)
    IndiciesArray = np.indices((height, width))
    IndiciesArray = np.append((IndiciesArray[0].reshape(height, width, 1)),
                             (IndiciesArray[1].reshape(height, width, 1)),
                             axis=2)
    data = np.append(IndiciesArray, data, axis=2)
    data = data*[1, 1, 1/ImageScaleFactor]
    data = np.append(data, np.array([[[1]]*width]*height), axis = 2).reshape(height*width, 4)
    data = data@[[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]]
    ones = np.array([[1,1,1]]*height*width)
    raw = np.array([data[:,3]]).T
    data = data/np.append(ones, raw, axis=1)**2
    data = (InvCalibrationMatrix@(data.T)).T*raw
    return data[:, :3]


color_raw = np.asarray(cv2.imread('annot.png'))
depth_raw = np.asarray(cv2.imread('depth.png', IMREAD_ANYDEPTH))

height = color_raw.shape[0]
width = color_raw.shape[1]

points = uvd_xyz_array(depth_raw)
colors = color_raw.reshape(height*width, 3)[:, ::-1]/256

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])