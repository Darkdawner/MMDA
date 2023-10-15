import cv2
import numpy as np
import math


def generate_heatmap(bodys, output_shape, stride, keypoint_num, kernel=(7, 7)):
    heatmaps = np.zeros((keypoint_num, *output_shape), dtype='float32')

    for i in range(keypoint_num):
        hm_tmp = np.zeros((len(bodys), *output_shape))
        for j in range(len(bodys)):
            if bodys[j][i][3] < 1:
                continue
            target_y = bodys[j][i][1] / stride
            target_x = bodys[j][i][0] / stride
            hm_tmp[j, int(target_y), int(target_x)] = 1
            hm_tmp[j] = cv2.GaussianBlur(hm_tmp[j], kernel, 0)
            maxi = np.amax(hm_tmp[j])
            if maxi > 1e-8:
                hm_tmp[j] /= maxi
        heatmaps[i] = np.max(hm_tmp, axis=0)
        maxi = np.amax(heatmaps[i])
        if maxi <= 1e-8:
            continue
        heatmaps[i] /= maxi / 255

    return heatmaps

def generate_yz_heatmap(bodys, scale, output_shape, stride, keypoint_num, kernel=(7, 7)):
    heatmaps = np.zeros((keypoint_num*2, *output_shape), dtype='float32')

    for i in range(keypoint_num):
        hm_tmp1 = np.zeros((len(bodys), *output_shape))
        hm_tmp2 = np.zeros((len(bodys), *output_shape))
        for j in range(len(bodys)):
            if bodys[j][2][3] < 1:
                break
            root_d = bodys[j][2][2]/((bodys[j][i][-4]*bodys[j][i][-3])**0.5)*100
            root_d_aug = bodys[j][2][2]/((bodys[j][i][-4]*bodys[j][i][-3])**0.5)/scale*44
            if bodys[j][i][3] < 1:
                continue
            d = bodys[j][i][2]/((bodys[j][i][-4]*bodys[j][i][-3])**0.5)*100
            target_z = d-root_d+root_d_aug  #最大可以*120
            # target_z_for_yz = bodys[j][i][2]*0.2*output_shape[1]/output_shape[0]
            target_y = bodys[j][i][1] / stride
            target_x = bodys[j][i][0] / stride
            hm_tmp1[j, int(target_z), int(target_x)] = 1
            hm_tmp2[j, int(target_y), 207-int(target_z)] = 1
            hm_tmp1[j] = cv2.GaussianBlur(hm_tmp1[j], kernel, 0)
            hm_tmp2[j] = cv2.GaussianBlur(hm_tmp2[j], kernel, 0)
            maxi = np.amax(hm_tmp1[j])
            maxj = np.amax(hm_tmp2[j])
            if maxi > 1e-8:
                hm_tmp1[j] /= maxi
            if maxj > 1e-8:
                hm_tmp2[j] /= maxj
        heatmaps[2*i] = np.max(hm_tmp1, axis=0)
        heatmaps[2*i+1] = np.max(hm_tmp2, axis=0)
        maxi = np.amax(heatmaps[2*i])
        maxj = np.amax(heatmaps[2*i+1])
        if maxi > 1e-8:
            heatmaps[2*i] /= maxi / 255
        if maxj > 1e-8:
            heatmaps[2*i+1] /= maxj / 255

    # heatmaps[keypoint_num*2:] = generate_xz_PIF(bodys, output_shape, stride, keypoint_num)
    # heatmaps[keypoint_num*2:] *= 25

    return heatmaps

def generate_xz_heatmap_aug(bodys, scale, output_shape, stride, keypoint_num, kernel=(7, 7)):
    heatmaps = np.zeros((keypoint_num, *output_shape), dtype='float32')

    for i in range(keypoint_num):
        hm_tmp1 = np.zeros((len(bodys), *output_shape))
        for j in range(len(bodys)):
            if bodys[j][2][3] < 1:
                break
            if bodys[j][i][3] < 1:
                continue
            target_z = bodys[j][i][2]/((bodys[j][i][-4]*bodys[j][i][-3])**0.5)/scale*180/208*512    #CMUP
            # target_z = bodys[j][i][2]/((bodys[j][i][-4]*bodys[j][i][-3])**0.5)/scale*44/128*208  #MuCo
            target_x = bodys[j][i][0] / stride
            hm_tmp1[j, int(target_z), int(target_x)] = 1
            hm_tmp1[j] = cv2.GaussianBlur(hm_tmp1[j], kernel, 0)
            maxi = np.amax(hm_tmp1[j])
            if maxi > 1e-8:
                hm_tmp1[j] /= maxi
        heatmaps[i] = np.max(hm_tmp1, axis=0)
        maxi = np.amax(heatmaps[i])
        if maxi > 1e-8:
            heatmaps[i] /= maxi / 255

    # heatmaps[keypoint_num*2:] = generate_xz_PIF(bodys, output_shape, stride, keypoint_num)
    # heatmaps[keypoint_num*2:] *= 25

    return heatmaps

def generate_yz_heatmap_aug(bodys, scale, output_shape, stride, keypoint_num, kernel=(7, 7)):
    heatmaps = np.zeros((keypoint_num, *output_shape), dtype='float32')

    for i in range(keypoint_num):
        hm_tmp1 = np.zeros((len(bodys), *output_shape))
        for j in range(len(bodys)):
            if bodys[j][2][3] < 1:
                break
            if bodys[j][i][3] < 1:
                continue
            target_z = bodys[j][i][2]/((bodys[j][i][-4]*bodys[j][i][-3])**0.5)/scale*180/208*512    #CMUP
            # target_z = bodys[j][i][2]/((bodys[j][i][-4]*bodys[j][i][-3])**0.5)/scale*44/128*208  #最大可以*120
            # target_z_for_yz = bodys[j][i][2]*0.2*output_shape[1]/output_shape[0]
            target_y = bodys[j][i][1] / stride
            hm_tmp1[j, int(target_y), int(target_z)] = 1
            hm_tmp1[j] = cv2.GaussianBlur(hm_tmp1[j], kernel, 0)
            maxi = np.amax(hm_tmp1[j])
            if maxi > 1e-8:
                hm_tmp1[j] /= maxi
        heatmaps[i] = np.max(hm_tmp1, axis=0)
        maxi = np.amax(heatmaps[i])
        if maxi > 1e-8:
            heatmaps[i] /= maxi / 255

    # heatmaps[keypoint_num*2:] = generate_xz_PIF(bodys, output_shape, stride, keypoint_num)
    # heatmaps[keypoint_num*2:] *= 25

    return heatmaps

def generate_rel_yz_heatmap(bodys, scale, output_shape, stride, keypoint_num, kernel=(7, 7)):
    heatmaps = np.zeros((keypoint_num*2, *output_shape), dtype='float32')

    for i in range(keypoint_num):
        hm_tmp1 = np.zeros((len(bodys), *output_shape))
        hm_tmp2 = np.zeros((len(bodys), *output_shape))
        for j in range(len(bodys)):
            if bodys[j][2][3] < 1:
                break
            if bodys[j][i][3] < 1:
                continue
            if i==2:
                target_z = bodys[j][2][2]/((bodys[j][i][-4]*bodys[j][i][-3])**0.5)/scale*44
            else:
                root_d = bodys[j][2][2]/((bodys[j][i][-4]*bodys[j][i][-3])**0.5)
                d = bodys[j][i][2]/((bodys[j][i][-4]*bodys[j][i][-3])**0.5)
                target_z = (d-root_d)*420+63.5  #max d-root_d = 0.134
            # target_z = bodys[j][i][2]/((bodys[j][i][-4]*bodys[j][i][-3])**0.5)/scale*44  #最大可以*120
            # target_root_z = bodys[j][2][2]/((bodys[j][2][-4]*bodys[j][2][-3])**0.5)/scale*44
            # target_z = target_z - target_root_z + 64
            # target_z_for_yz = bodys[j][i][2]*0.2*output_shape[1]/output_shape[0]
            target_y = bodys[j][i][1] / stride
            target_x = bodys[j][i][0] / stride
            hm_tmp1[j, int(target_z), int(target_x)] = 1
            hm_tmp2[j, int(target_y), int(target_z)] = 1
            hm_tmp1[j] = cv2.GaussianBlur(hm_tmp1[j], kernel, 0)
            hm_tmp2[j] = cv2.GaussianBlur(hm_tmp2[j], kernel, 0)
            maxi = np.amax(hm_tmp1[j])
            maxj = np.amax(hm_tmp2[j])
            if maxi > 1e-8:
                hm_tmp1[j] /= maxi
            if maxj > 1e-8:
                hm_tmp2[j] /= maxj
        heatmaps[2*i] = np.max(hm_tmp1, axis=0)
        heatmaps[2*i+1] = np.max(hm_tmp2, axis=0)
        maxi = np.amax(heatmaps[2*i])
        maxj = np.amax(heatmaps[2*i+1])
        if maxi > 1e-8:
            heatmaps[2*i] /= maxi / 255
        if maxj > 1e-8:
            heatmaps[2*i+1] /= maxj / 255

    # heatmaps[keypoint_num*2:] = generate_xz_PIF(bodys, output_shape, stride, keypoint_num)
    # heatmaps[keypoint_num*2:] *= 25

    return heatmaps


def generate_xyz_heatmap(bodys, output_shape, stride, keypoint_num, kernel=(7, 7)):
    heatmaps = np.zeros((keypoint_num*3, *output_shape), dtype='float32')

    for i in range(keypoint_num):
        for j in range(len(bodys)):
            if bodys[j][i][3] < 1:
                continue
            target_z = bodys[j][i][2]*0.2
            target_z_for_yz = bodys[j][i][2]*0.2*output_shape[1]/output_shape[0]
            target_y = bodys[j][i][1] / stride
            target_x = bodys[j][i][0] / stride
            add_Gaussian(heatmaps[3*i], target_y, target_x, kernel, stride)
            add_Gaussian(heatmaps[3*i+1], target_z, target_x, kernel, stride)
            add_Gaussian(heatmaps[3*i+2], target_y, target_z_for_yz, kernel, stride)
        heatmaps[3*i] *= 255
        heatmaps[3*i+1] *= 255
        heatmaps[3*i+2] *= 255

    return heatmaps


def generate_high_yz_heatmap(bodys, scale, output_shape, stride, keypoint_num, kernel=(7, 7)):
    heatmaps = np.zeros((keypoint_num*2, *output_shape), dtype='float32')

    for i in range(keypoint_num):
        hm_tmp1 = np.zeros((len(bodys), *output_shape))
        hm_tmp2 = np.zeros((len(bodys), *output_shape))
        for j in range(len(bodys)):
            if bodys[j][i][3] < 1:
                continue
            target_z = bodys[j][i][2]/((bodys[j][i][-4]*bodys[j][i][-3])**0.5)/scale*44*4  #最大可以*120
            target_y = bodys[j][i][1] / stride
            target_x = bodys[j][i][0] / stride
            hm_tmp1[j, int(target_z), int(target_x)] = 1
            hm_tmp2[j, int(target_y), int(target_z)] = 1
            hm_tmp1[j] = cv2.GaussianBlur(hm_tmp1[j], kernel, 0)
            hm_tmp2[j] = cv2.GaussianBlur(hm_tmp2[j], kernel, 0)
            maxi = np.amax(hm_tmp1[j])
            maxj = np.amax(hm_tmp2[j])
            if maxi > 1e-8:
                hm_tmp1[j] /= maxi
            if maxj > 1e-8:
                hm_tmp2[j] /= maxj
        heatmaps[2*i] = np.max(hm_tmp1, axis=0)
        heatmaps[2*i+1] = np.max(hm_tmp2, axis=0)
        maxi = np.amax(heatmaps[2*i])
        maxj = np.amax(heatmaps[2*i+1])
        if maxi > 1e-8:
            heatmaps[2*i] /= maxi / 255
        if maxj > 1e-8:
            heatmaps[2*i+1] /= maxj / 255
            
    return heatmaps

def generate_rdepth(meta, stride, root_idx, max_people):
    bodys = meta['bodys']
    scale = meta['scale']
    rdepth = np.zeros((max_people, 3), dtype='float32')
    for j in range(len(bodys)):
        if bodys[j][root_idx, 3] < 1 or j >= max_people:
            continue
        rdepth[j, 0] = bodys[j][root_idx, 1] / stride
        rdepth[j, 1] = bodys[j][root_idx, 0] / stride
        rdepth[j, 2] = bodys[j][root_idx, 2] / bodys[j][root_idx, 7] / scale  # normalize by f and scale
    rdepth = rdepth[np.argsort(-rdepth[:, 2])]
    return rdepth

def generate_paf(bodys, output_shape, params_transform, stride, paf_num, paf_vector, paf_thre, with_mds):
    pafs = np.zeros((paf_num * 2, *output_shape), dtype='float32')
    count = np.zeros((paf_num, *output_shape), dtype='float32')
    for i in range(paf_num):
        for j in range(len(bodys)):
            if paf_thre > 1 and with_mds:
                if bodys[j][paf_vector[i][0]][3] < 2 or bodys[j][paf_vector[i][1]][3] < 2:
                    continue
            elif bodys[j][paf_vector[i][0]][3] < 1 or bodys[j][paf_vector[i][1]][3] < 1:
                continue
            centerA = np.array(bodys[j][paf_vector[i][0]][:2], dtype=int)
            centerB = np.array(bodys[j][paf_vector[i][1]][:2], dtype=int)
            pafs[i*2:i*2+2], count[i] = putVecMaps(centerA, centerB, pafs[i*2:i*2+2], count[i], \
                                                     params_transform, stride, paf_thre)
    pafs[0::2] *= 127
    pafs[1::2] *= 127

    return pafs

def putVecMaps3D(centerA, centerB, accumulate_vec_map, count, params_transform, stride, thre):
    centerA = centerA.astype(float)
    centerB = centerB.astype(float)
    z_A = centerA[2]
    z_B = centerB[2]
    centerA = centerA[:2]
    centerB = centerB[:2]

    # stride = params_transform['stride']
    crop_size_y = params_transform['crop_size_y']
    crop_size_x = params_transform['crop_size_x']
    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    centerB = centerB / stride
    centerA = centerA / stride

    limb_vec = centerB - centerA
    limb_z = z_B - z_A
    norm = np.linalg.norm(limb_vec)
    if norm < 1.0:  # limb is too short, ignore it
        return accumulate_vec_map, count

    limb_vec_unit = limb_vec / norm

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    range_x = list(range(int(min_x), int(max_x), 1))
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)
    ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
    mask = limb_width < thre  # mask is 2D

    vec_map = np.copy(accumulate_vec_map) * 0.0
    vec_map[:, yy, xx] = np.repeat(mask[np.newaxis, :, :], 3, axis=0)
    vec_map[:2, yy, xx] *= limb_vec_unit[:, np.newaxis, np.newaxis]
    vec_map[2, yy, xx] *= limb_z
    mask = np.logical_or.reduce(
        (np.abs(vec_map[0, :, :]) != 0, np.abs(vec_map[1, :, :]) != 0))

    accumulate_vec_map = np.multiply(
        accumulate_vec_map, count[np.newaxis, :, :])
    accumulate_vec_map += vec_map

    count[mask == True] += 1

    mask = count == 0

    count[mask == True] = 1

    accumulate_vec_map = np.divide(accumulate_vec_map, count[np.newaxis, :, :])
    count[mask == True] = 0

    return accumulate_vec_map, count

def putVecMaps(centerA, centerB, accumulate_vec_map, count, params_transform, stride, thre):
    """Implement Part Affinity Fields
    :param centerA: int with shape (2,) or (3,), centerA will pointed by centerB.
    :param centerB: int with shape (2,) or (3,), centerB will point to centerA.
    :param accumulate_vec_map: one channel of paf.
    :param count: store how many pafs overlaped in one coordinate of accumulate_vec_map.
    :param params_transform: store the value of stride and crop_szie_y, crop_size_x                 
    """

    centerA = centerA.astype(float)
    centerB = centerB.astype(float)

    # stride = params_transform['stride']
    crop_size_y = params_transform['crop_size_y']
    crop_size_x = params_transform['crop_size_x']
    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    centerB = centerB / stride
    centerA = centerA / stride

    limb_vec = centerB - centerA
    norm = np.linalg.norm(limb_vec)
    if norm < 1.0:
        # print 'limb is too short, ignore it...'
        return accumulate_vec_map, count
    limb_vec_unit = limb_vec / norm
    # print 'limb unit vector: {}'.format(limb_vec_unit)

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    range_x = list(range(int(min_x), int(max_x), 1))
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)
    ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
    mask = limb_width < thre  # mask is 2D

    vec_map = np.copy(accumulate_vec_map) * 0.0
    vec_map[:, yy, xx] = np.repeat(mask[np.newaxis, :, :], 2, axis=0)
    vec_map[:, yy, xx] *= limb_vec_unit[:, np.newaxis, np.newaxis]

    mask = np.logical_or.reduce(
        (np.abs(vec_map[0, :, :]) > 0, np.abs(vec_map[1, :, :]) > 0))

    accumulate_vec_map = np.multiply(
        accumulate_vec_map, count[np.newaxis, :, :])
    accumulate_vec_map += vec_map
    count[mask == True] += 1

    mask = count == 0

    count[mask == True] = 1

    accumulate_vec_map = np.divide(accumulate_vec_map, count[np.newaxis, :, :])
    count[mask == True] = 0

    return accumulate_vec_map, count

def add_Gaussian(heatmap, x, y, kernel, stride):
    n_sigma = 4
    sigma = kernel[0]
    tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
    tl[0] = max(tl[0], 0)
    tl[1] = max(tl[1], 0)

    br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
    map_h, map_w = heatmap.shape
    br[0] = min(br[0], map_h)
    br[1] = min(br[1], map_w)

    for map_y in range(tl[1], br[1]):
        for map_x in range(tl[0], br[0]):
            d2 = (map_x * stride - x * stride) * (map_x * stride - x * stride) + \
                (map_y * stride - y * stride) * (map_y * stride - y * stride)
            exponent = d2 / 2 / sigma / sigma
            if exponent > 4.6052:  # threshold, ln(100), ~0.01
                continue
            heatmap[map_x, map_y] = max(math.exp(-exponent), heatmap[map_x, map_y])

