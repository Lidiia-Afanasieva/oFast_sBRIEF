# from zmq.constants import QUEUE
# import numpy as np
# import cv2
# from google.colab.patches import cv2_imshow
# from matplotlib import pyplot as plt
# from copy import deepcopy

import numpy as np
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist
import cv
import matplotlib.pyplot as plt
from time import time
from skimage.feature import plot_matches
from skimage.transform import pyramid_gaussian

# import cv2
# import matplotlib.pyplot as plt
# from time import time
# from skimage.feature import plot_matches
# from skimage.transform import pyramid_gaussian


class KeyPoint:

    pass


def get_nms_kp(V, kp_map, near_kp_count):
    fewer_kps = []
    for [col, row] in kp_map:
        # окно в пределах которого будут сравниваться значения пикселей с итер
        window = V[row - near_kp_count:row + near_kp_count + 1, col - near_kp_count:col + near_kp_count + 1]
        # v_max = window.max()
        # нахожнение наибольшего в окне
        print(f'flatten version: {window.argmax()}, shape of 2D win: {window.shape}')
        loc_row_col = np.unravel_index(window.argmax(), window.shape)
        # приведение к реальным координатам
        col_new = col + loc_row_col[1] - near_kp_count
        row_new = row + loc_row_col[0] - near_kp_count
        new_kp = [col_new, row_new]
        if new_kp not in fewer_kps:
            fewer_kps.append(new_kp)
    return fewer_kps


# FAST
# N=9/12
def FAST(img, N=9, threshold=0.15, nms=2):
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 16  # 3x3 Gaussian kernel/Gaussian Window

    img = convolve2d(img, kernel, mode='same')

    # для окружности в 3 пикселя из алгоритма Брезенхема
    circle_idx = np.array([[3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3],
                           [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1]])
    cross_idx = np.array([[3, 0, -3, 0], [0, 3, 0, -3]])

    nms_V_func = np.zeros(img.shape)
    keypoints = []
    for row in range(3, img.shape[0] - 3):
        for col in range(3, img.shape[1] - 3):
            Ip = img[row, col]

            # пороговая величина
            # 15% от интенсивности выбранного пикселя
            t = threshold * Ip if threshold < 1 else threshold

            # проверка 1, 5, 9 и 13 пикселей
            # проверка интенсивности Ip±t
            current_I_fast = img[row + cross_idx[0, :], col + cross_idx[1, :]]
            if np.count_nonzero(Ip + t < current_I_fast) >= 3 or np.count_nonzero(Ip - t > current_I_fast) >= 3:

                # проверка всех пикселей на окружности
                current_I_full = img[row + circle_idx[0, :], col + circle_idx[1, :]]
                if np.count_nonzero(current_I_full >= Ip + t) >= N or np.count_nonzero(current_I_full <= Ip - t) >= N:

                    # добавление ключевой точки
                    keypoints.append([col, row])  # keypoint = [col, row]
                    nms_V_func[row, col] = np.sum(np.abs(Ip - current_I_full))  # NMS

    if nms != 0:
        fewer_kps = get_nms_kp(nms_V_func, keypoints, nms)
    else:
        fewer_kps = keypoints

    return np.array(fewer_kps)


def find_centroid(img, corners, mask, mask_r, mask_c, middle_r, middle_c):
    teta = []
    for i in range(corners.shape[0]):
        c0, r0 = corners[i, :]
        m01, m10 = 0, 0  # равны нулю при grayscale
        for r in range(mask_r):
            m01_temp = 0
            for c in range(mask_c):
                if mask[r, c]:
                    I = img[r0 + r, c0 + c]  # интенсивность элементов маски
                    m10 = m10 + I * (c - middle_c)
                    m01_temp = m01_temp + I
            m01 = m01 + m01_temp * (r - middle_r)
        teta.append(np.arctan2(m01, m10))
    return np.array(teta)


def get_intensity_centroind(img, corners):
    # создание маски окна ключевой точки для дальнейшего вычисления центроида
    orientation_mask = np.zeros((31, 31), dtype=np.int32)
    # kp_pose = [15, 15]
    u_max = [15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3]
    for i in range(-15, 16):
        for j in range(-u_max[abs(i)], u_max[abs(i)] + 1):
            orientation_mask[15 + j, 15 + i] = 1
            # orientation_mask[map(sum, zip(kp_pose, [j, i]))]
    mask_r, mask_c = orientation_mask.shape
    middle_r = (mask_r - 1) // 2
    middle_c = (mask_c - 1) // 2
    img = np.pad(img, (middle_r, middle_c), mode='constant', constant_values=0)
    try:
        return find_centroid(img, corners, orientation_mask, mask_r, mask_c, middle_r, middle_c)
    except Exception:
        print('!!!!SOMETHING IN CENTROID OR-TION!!!!')


'''
    patch_size - размер квадрата, определяющего соседствующие с пикселем пикс
    в брифе дескриптор бинарных признаков представлен вектором из 0 и 1 размерностью 128-512
'''


def BRIEF(img, keypoints, orientations=None, n=256, patch_size=9, sigma=1, mode='uniform', sample_seed=42):
    '''
    BRIEF [Binary Robust Independent Elementary Features] keypoint/corner descriptor
    '''
    random = np.random.RandomState(seed=sample_seed)

    # kernel = np.array([[1,2,1],
    #                    [2,4,2],
    #                    [1,2,1]])/16      # 3x3 Gaussian Window

    kernel = np.array([[1, 4, 7, 4, 1],
                       [4, 16, 26, 16, 4],
                       [7, 26, 41, 26, 7],
                       [4, 16, 26, 16, 4],
                       [1, 4, 7, 4, 1]]) / 273  # 5x5 Gaussian Window

    img = convolve2d(img, kernel, mode='same')

    ''' Uniform (G1) 
        пары пикселей x,y случайно берутся из нормального распределения
        или из разбросса patch_size/2 вокруг kp
        patch_size - 2, чтобы исключить попадание на крайний?
    '''
    if mode == 'normal':
        # попарное распределение из
        samples = (patch_size / 5.0) * random.randn(n * 8)
        samples = np.array(samples, dtype=np.int32)
        samples = samples[(samples < (patch_size // 2)) & (samples > - (patch_size - 2) // 2)]
        pos1 = samples[:n * 2].reshape(n, 2)
        pos2 = samples[n * 2:n * 4].reshape(n, 2)

    elif mode == 'uniform':
        # генерация пар координат пикселей в рабочей зоне в массив dim[2,2]
        samples = random.randint(-(patch_size - 2) // 2 + 1, (patch_size // 2), (n * 2, 2))
        samples = np.array(samples, dtype=np.int32)

        # каждый набор - массив пикселей x,y из тестовых пикселей
        pos1, pos2 = np.split(samples, 2)  # делит на два np размерностью n,2

    rows, cols = img.shape

    if orientations is None:
        # получение маски интенсивности пикселей патча? нет
        # маска позволяет проверить находится ли ключевая точка в точках, где будет не середина и больше
        # до стенки изображения на крайнем патче
        mask = (((patch_size // 2 - 1) < keypoints[:, 0])
                & (keypoints[:, 0] < (cols - patch_size // 2 + 1))
                & ((patch_size // 2 - 1) < keypoints[:, 1])
                & (keypoints[:, 1] < (rows - patch_size // 2 + 1)))

        # тип данных индекс
        keypoints = np.array(keypoints[mask, :], dtype=np.intp, copy=False)
        descriptors = np.zeros((keypoints.shape[0], n), dtype=bool)  # карта дескриптора
        # массив из списков дескрипторов для каждой ключевой точки без именения ориентации

        # нахождение центроида?
        # первый пиксель с которого начинается сравнение интенсивности?

        # всё-таки запись пар тестовых пикселей///
        # итерация по количеству пар тестовых пикселей
        for p in range(pos1.shape[0]):  # количество пар
            pr0 = pos1[p, 0]  # row
            pc0 = pos1[p, 1]  # col
            pr1 = pos2[p, 0]
            pc1 = pos2[p, 1]

            # сравнение
            # сопоставление первого тестового пикселя и
            for k in range(keypoints.shape[0]):
                # нахождение координат ключевой точки, к которой привязан патч
                # позволяет получить интенсивность тестового пикселя из реального изображения
                kr = keypoints[k, 1]
                kc = keypoints[k, 0]
                if img[kr + pr0, kc + pc0] < img[kr + pr1, kc + pc1]:
                    descriptors[k, p] = True  # 0 or 1
    else:
        # Using orientations

        # masking the keypoints with a safe distance from borders
        # instead of the patch_size//2 distance used in case of no rotations.
        distance = int((patch_size // 2) * 1.5)  # безопасное расстояние от бортов патча с учётом поворота
        mask = (((distance - 1) < keypoints[:, 0])
                & (keypoints[:, 0] < (cols - distance + 1))
                & ((distance - 1) < keypoints[:, 1])
                & (keypoints[:, 1] < (rows - distance + 1)))

        keypoints = np.array(keypoints[mask], dtype=np.intp, copy=False)  # индексы
        orientations = np.array(orientations[mask], copy=False)
        descriptors = np.zeros((keypoints.shape[0], n), dtype=bool)

        for i in range(descriptors.shape[0]):
            angle = orientations[i]  # радианное представления угла отклонения от 12часов??????
            sin_theta = np.sin(angle)
            cos_theta = np.cos(angle)

            kr = keypoints[i, 1]
            kc = keypoints[i, 0]
            for p in range(pos1.shape[0]):
                pr0 = pos1[p, 0]
                pc0 = pos1[p, 1]
                pr1 = pos2[p, 0]
                pc1 = pos2[p, 1]

                # Rotation is based on the idea that:
                # x` = x*cos(th) - y*sin(th)
                # y` = x*sin(th) + y*cos(th)
                # c -> x & r -> y

                # сумма проекций
                spr0 = round(sin_theta * pr0 + cos_theta * pc0)
                spc0 = round(cos_theta * pr0 - sin_theta * pc0)
                spr1 = round(sin_theta * pr1 + cos_theta * pc1)
                spc1 = round(cos_theta * pr1 - sin_theta * pc1)

                if img[kr + spr0, kc + spc0] < img[kr + spr1, kc + spc1]:
                    descriptors[i, p] = True
    return descriptors  # вернёт таблицу True/False


'''
    Brute force matching of the BRIEF descriptors based on hamming distance, with the option to perform cross-check, and to 
    remove ambiguous matches (confusing matches) using a distance_ratio.
'''


def match(descriptors1, descriptors2, max_distance=np.inf, cross_check=True, distance_ratio=None):
    distances = cdist(descriptors1, descriptors2, metric='hamming')  # distances.shape: [len(d1), len(d2)]

    indices1 = np.arange(descriptors1.shape[0])  # [0, 1, 2, 3, 4, 5, 6, 7, ..., len(d1)] "indices of d1"
    indices2 = np.argmin(distances, axis=1)  # [12, 465, 23, 111, 123, 45, 67, 2, 265, ..., len(d1)] "list of the indices of d2 points that are closest to d1 points"
    # Each d1 point has a d2 point that is the most close to it.
    if cross_check:
        '''
        Cross check idea:
        what d1 matches with in d2 [indices2], should be equal to 
        what that point in d2 matches with in d1 [matches1]
        '''
        matches1 = np.argmin(distances, axis=0)  # [15, 37, 283, ..., len(d2)] "list of d1 points closest to d2 points"
        # Each d2 point has a d1 point that is closest to it.
        # indices2 is the forward matches [d1 -> d2], while matches1 is the backward matches [d2 -> d1].
        mask = indices1 == matches1[indices2]  # len(mask) = len(d1)
        # we are basically asking does this point in d1 matches with a point in d2 that is also matching to the same point in d1 ?
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if max_distance < np.inf:
        mask = distances[indices1, indices2] < max_distance
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if distance_ratio is not None:
        '''
        the idea of distance_ratio is to use this ratio to remove ambigous matches.
        ambigous matches: matches where the closest match distance is similar to the second closest match distance
                          basically, the algorithm is confused about 2 points, and is not sure enough with the closest match.
        solution: if the ratio between the distance of the closest match and
                  that of the second closest match is more than the defined "distance_ratio",
                  we remove this match entirly. if not, we leave it as is.
        '''
        modified_dist = distances
        fc = np.min(modified_dist[indices1, :], axis=1)
        modified_dist[indices1, indices2] = np.inf
        fs = np.min(modified_dist[indices1, :], axis=1)
        mask = fc / fs <= 0.5
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    # sort matches using distances
    dist = distances[indices1, indices2]
    sorted_indices = dist.argsort()

    matches = np.column_stack((indices1[sorted_indices], indices2[sorted_indices]))
    return matches


if __name__ == "__main__":

    # Trying multi-scale
    N_LAYERS = 4
    DOWNSCALE = 2

    img1 = cv.imread('images/chess3.jpg')
    original_img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    gray1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    grays1 = list(pyramid_gaussian(gray1, downscale=DOWNSCALE, max_layer=N_LAYERS, multichannel=False))

    img2 = cv.imread('images/chess.jpg')
    original_img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    gray2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
    grays2 = list(pyramid_gaussian(gray2, downscale=2, max_layer=4, multichannel=False))

    scales = [(i * DOWNSCALE if i > 0 else 1) for i in range(N_LAYERS)]
    features_img1 = np.copy(img1)
    features_img2 = np.copy(img2)

    kps1 = []
    kps2 = []
    ds1 = []
    ds2 = []
    ms = []
    for i in range(len(scales)):
        scale_kp1 = FAST(grays1[i], N=9, threshold=0.15, nms_window=3)
        kps1.append(scale_kp1 * scales[i])
        scale_kp2 = FAST(grays2[i], N=9, threshold=0.15, nms_window=3)
        kps2.append(scale_kp2 * scales[i])
        for keypoint in scale_kp1:
            features_img1 = cv.circle(features_img1, tuple(keypoint * scales[i]), 3 * scales[i], (0, 255, 0), 1)
        for keypoint in scale_kp2:
            features_img2 = cv.circle(features_img2, tuple(keypoint * scales[i]), 3 * scales[i], (0, 255, 0), 1)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(grays1[i], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(features_img1)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(grays2[i], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(features_img2)

        d1 = BRIEF(grays1[i], scale_kp1, mode='uniform', patch_size=8, n=512)
        ds1.append(d1)
        d2 = BRIEF(grays2[i], scale_kp2, mode='uniform', patch_size=8, n=512)
        ds2.append(d2)

        matches = match(d1, d2, cross_check=True)
        ms.append(matches)
        print('no. of matches: ', matches.shape[0])

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1)

        plot_matches(ax, grays1[i], grays2[i], np.flip(scale_kp1, 1), np.flip(scale_kp2, 1), matches)
        plt.show()

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(features_img1)
    plt.subplot(1, 2, 2)
    plt.imshow(features_img2)
    plt.show()