from cv2 import IMREAD_ANYDEPTH, COLOR_BGR2RGB
import open3d as o3d
import numpy as np
import cv2


ImageScaleFactor = 5000


def uvd_xyz_array(data):
    # данная функция получает на вход двумерный массив с информацией о глубине
    # соответсвующих положению пикселей и возвращает координаты точек,
    # соответсвующих данным пискелям в трёхмерном пространстве

    # получаем размеры входных данных
    height, width = data.shape
    # вкладываем каждый элемент в входном двумерном массиве в массив
    data = data.reshape(height, width, 1)

    # cоздаём двумерный индексный массив
    IndiciesArray = np.indices((height, width))
    IndiciesArray = np.append((IndiciesArray[0].reshape(height, width, 1)),
                              (IndiciesArray[1].reshape(height, width, 1)),
                              axis=2)
    # объеденяем данные глубины с координатами
    data = np.append(IndiciesArray, data, axis=2)
    # нормируем глубину по ImageScaleFactor
    data = data*[1, 1, 1/ImageScaleFactor]
    # добавляем четвёртый параметр для данных точек
    data = np.append(data, np.array([[[1]]*width]*height), axis=2)
    # раскладываем трёхмерный массив в двумерный
    data = data.reshape(height*width, 4)
    # меняем третий и четвёрый пераметр местами
    data = data@[[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]]
    # создаём массив из единиц и четвёртого столбца
    ones = np.array([[1, 1, 1]]*height*width)
    raw = np.array([data[:, 3]]).T
    # возводим четвёртый параметр в степень -1 (делим на себя же дважды)
    data = data/np.append(ones, raw, axis=1)**2
    # объявляем калибровочную матрицу с информацией о камере
    InvCalibrationMatrix = np.linalg.inv([[481.20, 0., 319.50, 0.],
                                          [0., -480.00, 239.50, 0.],
                                          [0., 0., 1., 0.],
                                          [0., 0., 0., 1.]])
    # находим матричное произвдение калибровочной матрицы и точек
    # результат домножаем на четвёртый параметр до возведения в степень -1
    data = (InvCalibrationMatrix@(data.T)).T*raw
    # первые три параметра это координаты точки в трёмерном пространсттве
    return data[:, :3]


def main():
    color_raw = cv2.imread('annot.png')
    color_raw = cv2.cvtColor(color_raw, COLOR_BGR2RGB)
    depth_raw = cv2.imread('depth.png', IMREAD_ANYDEPTH)

    height, width = depth_raw.shape

    points = uvd_xyz_array(depth_raw)
    colors = color_raw.reshape(height * width, 3)/256

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()
