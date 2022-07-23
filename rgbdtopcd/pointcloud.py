from cv2 import IMREAD_ANYDEPTH, COLOR_BGR2RGB
import open3d as o3d
import numpy as np
import cv2


ImageScaleFactor = 5000

fx = 481.20
fy = -480.00
cx = 319.50
cy = 239.50


def uvd_xyz_array(data):
    """Function that converts depth image data to points in 3d space

    Args:
        data (numpy.ndarray): Array containing depth image data

    Returns:
        numpy.ndarray: Array containing points in 3d space
    """

    height, width = data.shape
    data = data.reshape(height * width)

    u = np.repeat(np.arange(height), width)
    v = np.tile(np.arange(width), height)
    z = data / ImageScaleFactor

    x = (cx - u) * z / fx
    y = (cy - v) * z / fy

    return np.column_stack((x, y, z))


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
