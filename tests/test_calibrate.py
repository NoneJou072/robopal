import glob
import os
import numpy as np
import cv2

"""
# 本程序用于 Eye-to-Hand 相机标定，包括相机的内参标定、外参标定和手眼标定。
为了得到一个好的标定结果，应该使得标定板尽量出现在摄像头视野的各个位置里，
（如标定板出现在视野中的左边，右边，上边和下边，标定板既有倾斜的，也有水平的。）

具体标定流程为：
1. 将标定板固定在机械臂末端
2. 开启机械臂和摄像头，机械臂切换到拖动示教模式
3. 在距离摄像头的不同位置上，使用各种角度各拍照15-20张照片，一共45-60张。
    每次拍摄时同时保存照片以及对应拍照时机械臂位姿

手眼标定需要用到的参数：
A ：机器人末端坐标系到机器人基座标系的转换矩阵，机器人运动学正解可知。
? ：标定板坐标系到机器人末端坐标系的转换矩阵，标定板是固定安装在机器人末端的，所以固定不变，但未知（不需要）。
B ：相机坐标系到标定板坐标系的转换矩阵，即【相机外参】，可知。
X ：相机坐标系到机器人基坐标系的转换矩阵，待求量（AX=XB）。

# REFERENCES:
# https://www.guyuehome.com/36266
# https://www.sensemoment.com/fenxiang/84.html
# https://github.com/JayaoLiu/eye_to_hand_calibrate/blob/main/eye_to_hand_calibrate.py
"""


def intrinsics_calibrate():
    """ 内参标定
    """
    # 获取标定板角点的位置， 我们先标定内参，因此无需关注世界坐标系的位置。按内角点的顺序从0-6,0-9即可
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)
    square_size = 0.02  # 标定板黑白格子的大小，单位是m
    objp *= square_size

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    obj_points = []
    img_points = []
    images = glob.glob("../robopal/commons/cv_cache/*.png")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (6, 9), None)
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)

            cv2.drawChessboardCorners(img, (6, 9), corners, ret)  # 显示角点顺序
            cv2.imshow('findCorners',img)
            cv2.waitKey(100)
    cv2.destroyAllWindows()

    img = cv2.imread("../robopal/commons/cv_cache/cv_cache_image_9.png")
    # 返回内参矩阵mtx,畸变系数dist,旋转向量rvecs,平移矩阵tvecs。由于obj_points没有输入真实的世界坐标，因此rvecs和tvecs不作参考
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    h, w = img.shape[:2]
    # 获取畸变矫正outer newcameramtx，alpha参数设为1，保留所有像素点，和矫正前图片尺寸相同，但是会有黑边
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  # 进行畸变矫正
    x, y, w, h = roi  # 矫正后的内矩阵区域，也就是去掉黑边的区域。参考：https://www.cnblogs.com/riddick/p/6711263.html
    # dst1 = dst[y:y+h,x:x+w] # 裁剪矫正后的黑边
    # dst1 = cv2.resize(dst1,(1280,960))
    cv2.imwrite('calibrate_alpha1_2.jpg', dst)

    print("畸变内参矩阵mtx:\n", mtx)
    print("无畸变内参矩阵newcameramtx:\n", newcameramtx)
    print("畸变系数dist:\n", dist)
    return mtx, dist, newcameramtx, obj_points, img_points


def extrinsics_calibrate(mtx, dist, newcameramtx, obj_points_list, img_points_list):
    """ 外参标定
    外参标定的目的是求出标定板在相机坐标系下的位姿，即旋转矩阵R和平移矩阵T。
    我们需要将每一组位姿保存下来，用于后续的手眼标定。
    """
    R_target2cam_list = []
    t_target2cam_list = []
    for obj_points, img_points in zip(obj_points_list, img_points_list):
        ret, rvecs, tvecs = cv2.solvePnP(obj_points, img_points, mtx, dist)
        # 这里注意：如果内参矩阵传的是outer newcameramtx，则img_coordinate为矫正后的坐标。如果传mtx，则为矫正前原图的坐标。
        R = cv2.Rodrigues(rvecs)[0]
        T = tvecs
        print("旋转矩阵R:\n", R)
        print("平移矩阵T:\n", T)
        R_target2cam_list.append(R)
        t_target2cam_list.append(T)
    return R_target2cam_list, t_target2cam_list


def hand_eye_calibration(R_target2cam, t_target2cam):
    """ 手眼标定
    """
    R_gripper2base = np.array(
        [
            [[-0.81523847, 0.06176942, -0.57582183],
             [0.01805324, 0.99652295, 0.08133931],
             [0.57884395, 0.05591548, -0.81351899]],
            [[-0.80721508, 0.24517133, -0.53693094],
             [0.08473714, 0.94836682, 0.30564685],
             [0.58414333, 0.20122476, -0.78631111]],
            [[-0.77483772, 0.00514916, -0.63213922],
             [-0.04113122, 0.99743731, 0.05854087],
             [0.63082068, 0.07136034, -0.77264026]],
            [[-0.71722597, -0.12670822, -0.685224],
             [-0.10447456, 0.99176777, -0.07403895],
             [0.68896442, 0.01848581, -0.72455938]],
            [[-0.5947953, -0.37653275, -0.71024055],
             [-0.21256046, 0.92573947, -0.31276906],
             [0.77526551, -0.03506451, -0.63066146]],
            [[-0.56223147, -0.301357, -0.7701167],
             [0.08667812, 0.90463646, -0.41727663],
             [0.82242487, -0.30135832, -0.48249403]],
            [[-0.64943399, 0.46846927, -0.59897583],
             [0.46477387, 0.86797682, 0.17493281],
             [0.60184778, -0.164781, -0.78142592]],
            [[-0.59526203, 0.25313355, -0.76261821],
             [0.4052811, 0.91410071, -0.01292769],
             [0.69383741, -0.31677011, -0.64671968]],
            [[-0.55661209, 0.07806561, -0.82709657],
             [0.19079222, 0.98097712, -0.03580801],
             [0.80856744, -0.17773476, -0.56091804]],
            [[-0.58755935, 0.06565756, -0.80651293],
             [0.27344166, 0.95418057, -0.12152817],
             [0.76157972, -0.29193924, -0.57859122]]
        ], dtype=np.float32
    )  # 机械臂末端的旋转矩阵

    t_gripper2base = np.array(
        [
            [0.77231561, -0.01693119, 0.6262425],
            [0.68792659, -0.07435932, 0.51365529],
            [0.67232182, -0.0024266, 0.47715268],
            [0.65857607, 0.03667907, 0.44405299],
            [0.63827217, 0.10568696, 0.41220694],
            [0.55225239, 0.0687032, 0.45068874],
            [0.4840232, -0.13791102, 0.49385753],
            [0.56884939, -0.12711942, 0.53411788],
            [0.68522775, 0.01432181, 0.53735318],
            [0.71383748, 0.0201255, 0.61142279]
        ], dtype=np.float32
    )  # 机械臂末端的坐标

    R_cam2base, t_cam2base = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, cv2.CALIB_HAND_EYE_HORAUD)
    print("相机坐标系到机器人基坐标系的旋转矩阵:\n", R_cam2base)
    print("相机坐标系到机器人基坐标系的平移矩阵:\n", t_cam2base)


if __name__ == "__main__":
    print('########## 一、开始内参标定 ##########')
    mtx, dist, newcameramtx, obj_points, img_points = intrinsics_calibrate()
    print('########## 二、开始外参标定 ##########')
    R_target2cam_list, t_target2cam_list = extrinsics_calibrate(mtx, dist, newcameramtx, obj_points, img_points)
    print('########## 三、开始手眼标定 ##########')
    hand_eye_calibration(R_target2cam_list, t_target2cam_list)
    print('########## 所有标定完成 ##########')
