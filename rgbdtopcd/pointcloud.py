from cv2 import IMREAD_ANYDEPTH, COLOR_BGR2RGB
import open3d as o3d
import numpy as np
import cv2


ImageScaleFactor = 5000


def uvd_xyz_array(data):
    """Function that converts depth image data to points in 3d space

    Args:
        data (numpy.ndarray): Array containing depth image data

    Returns:
        numpy.ndarray: Array containing points in 3d space
    """

    height, width = data.shape
    data = data.reshape(height, width, 1)

    # 2d array containing arrays with their coords in 2d array
    CoordsArray = np.indices((height, width))
    CoordsArray = np.append(
        (CoordsArray[0].reshape(height, width, 1)),
        (CoordsArray[1].reshape(height, width, 1)),
        axis=2,
    )
    # merging coords of points with their depth
    data = np.append(CoordsArray, data, axis=2)
    # normalizing depth by image scale factor
    data = data * [1, 1, 1 / ImageScaleFactor]
    # extending points data from 3 element arrays to 4 element arrays
    data = np.append(data, np.array([[[1]] * width] * height), axis=2)
    # reshaping 3d array into 2d
    data = data.reshape(height * width, 4)
    # Swapping 3 and 4 points parameters
    data = data @ [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    # raising 4-th parameters of points to the power of -1
    ones = np.array([[1, 1, 1]] * height * width)
    raw = np.array([data[:, 3]]).T
    data = data / np.append(ones, raw, axis=1) ** 2
    # Camera calibrating matrix
    InvCalibrationMatrix = np.linalg.inv(
        [
            [481.20, 0.0, 319.50, 0.0],
            [0.0, -480.00, 239.50, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    # multiplying points data by calibration matrix and their 4 parameters
    data = (InvCalibrationMatrix @ (data.T)).T * raw
    # First 3 parameters of points are their coords in 3d space
    return data[:, :3]


def main():
    color_raw = cv2.imread("annot.png")
    color_raw = cv2.cvtColor(color_raw, COLOR_BGR2RGB)
    depth_raw = cv2.imread("depth.png", IMREAD_ANYDEPTH)

    height, width = depth_raw.shape

    points = uvd_xyz_array(depth_raw)
    colors = color_raw.reshape(height * width, 3) / 256

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
