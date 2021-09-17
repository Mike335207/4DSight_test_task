import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math


MIN_MATCH_COUNT = 10
F = 100.0
P_X = 960.0
P_Y = 540.0

#find homography and draw matches
def findHomography(matches, kp1, kp2, img1, img2, output_name):
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    cv.imwrite(output_name, img3)
    return M


# Calculates rotation matrix to euler angles
def rotationMatrixToEulerAngles(R) :

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

#Calculates rotation matrix and translation vector from homography matrix
def homographyToRotAndT(M):
    K= np.matrix([[F, 0.0, P_X], [0.0, F, P_Y], [0.0, 0.0, 1.0]])
    H = M.T
    h1 = H[0]
    h2 = H[1]
    h3 = H[2]
    K_inv = np.linalg.inv(K)
    L = 1 / np.linalg.norm(np.dot(K_inv, h1))
    r1 = L * np.dot(K_inv, h1)
    r2 = L * np.dot(K_inv, h2)
    r3 = np.cross(r1, r2)
    T = L * (K_inv @ h3.reshape(3, 1))
    T = T/T[2]
    R = np.array([[r1], [r2], [r3]])
    R = np.reshape(R, (3, 3))
    return R, T

if __name__ == '__main__':
    img1 = cv.imread('data/imgs/img1.png', 0)
    img2 = cv.imread('data/imgs/img2.png', 0)
    img3 = cv.imread('data/imgs/img3.png', 0)
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    kp3, des3 = sift.detectAndCompute(img3, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches_1_2 = flann.knnMatch(des1, des2, k=2)
    matches_1_3 = flann.knnMatch(des1, des3, k=2)

    M_1_2 = findHomography(matches_1_2, kp1, kp2, img1, img2, 'matched_1_2.png')
    M_1_3 = findHomography(matches_1_3, kp1, kp3, img1, img3, 'matched_1_3.png')

    R_1_2, T_1_2 = homographyToRotAndT(M_1_2)
    R_1_3, T_1_3 = homographyToRotAndT(M_1_3)

    rot_1_2 = rotationMatrixToEulerAngles(R_1_2)
    rot_1_3 = rotationMatrixToEulerAngles(R_1_3)

    plt.clf()

    plt.xlim([-10, 1])
    plt.ylim([-10, 1])
    plt.text(0, 0, 'Cam1: R[0.0 0.0 0.0]')
    plt.text(T_1_2[0], T_1_2[1], 'Cam2: R=' + str(rot_1_2))
    plt.text(T_1_3[0], T_1_3[1] + 0.3, 'Cam3: R=' + str(rot_1_3))
    plt.scatter(0.0, 0.0, None, 'orange')
    plt.scatter(float(T_1_2[0]), float(T_1_2[1]), None, 'red')
    plt.scatter(float(T_1_3[0]), float(T_1_3[1]), None, 'blue')

    plt.savefig('result.png')
