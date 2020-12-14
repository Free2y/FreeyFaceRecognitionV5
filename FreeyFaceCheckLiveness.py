#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import OrderedDict

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

from FreezyFaceDRecognition import FreezyFaceDRecognition

# face_recognition比较慢  dlib比较快
class FreezyFaceCheckLiveness():
    # 关键点排序
    FACIAL_LANDMARKS_68_IDXS = OrderedDict([
        ("mouth", (48, 68)),
        ("right_eyebrow", (17, 22)),
        ("left_eyebrow", (22, 27)),
        ("right_eye", (36, 42)),
        ("left_eye", (42, 48)),
        ("nose", (27, 36)),
        ("jaw", (0, 17))
    ])

    # 设置判断参数
    EYE_AR_THRESH = 0.2  # 低于该值则判断为眨眼
    EYE_AR_CONSEC_FRAMES = 3
    BLINK_THRESH = 1  # 眨眼次数的阈值
    SCALE_WIDTH = 320

    def __init__(self, ):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
        self.ffdr = FreezyFaceDRecognition()

    def eye_aspect_ratio(self, eye):
        """
        计算眼睛上下关键点欧式距离
        :param eye:眼睛关键点位置
        :return: 眼睛睁开程度
        """
        # 计算距离，竖直的
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # 计算距离，水平的
        C = dist.euclidean(eye[0], eye[3])
        # ear值
        ear = (A + B) / (2.0 * C)
        return ear

    def shape_to_np(self, shape, dtype="int"):
        # 创建68*2
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)
        # 遍历每一个关键点
        # 得到坐标
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def check_uri_dlib(self, video_uri):
        # 读取视频
        TOTAL_BLINK = 0
        COUNTER = 0
        (lStart, lEnd) = self.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = self.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        print("[INFO] starting video stream thread...")
        # 湖南直播数据rtmp://58.200.131.2:1935/livetv/hunantv
        print(video_uri)
        vs = cv2.VideoCapture(video_uri)
        rate = vs.get(cv2.CAP_PROP_FPS)
        # 遍历每一帧
        while True:
            # 预处理
            frame = vs.read()[1]
            if frame is None:
                break
            (h, w) = frame.shape[:2]
            width = self.SCALE_WIDTH
            r = width / float(w)
            dim = (width, int(h * r))
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            rects = self.detector(gray, 0)

            # 遍历每一个检测到的人脸
            for rect in rects:
                # 获取坐标
                shape = self.predictor(gray, rect)
                shape = self.shape_to_np(shape)

                # 分别计算ear值
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)

                # 算一个平均的
                ear = (leftEAR + rightEAR) / 2.0

                # 绘制眼睛区域
                # leftEyeHull = cv2.convexHull(leftEye)
                # rightEyeHull = cv2.convexHull(rightEye)
                # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # 检查是否满足阈值
                if ear < self.EYE_AR_THRESH:
                    COUNTER += 1
                else:
                    # 如果连续几帧都是闭眼的，总数算一次
                    if COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        TOTAL_BLINK += 1
                        # 重置
                        COUNTER = 0
                        if TOTAL_BLINK > self.BLINK_THRESH:
                            vs.release()
                            # cv2.destroyAllWindows()
                            return TOTAL_BLINK
        #         cv2.putText(frame, "Blinks: {}".format(TOTAL_BLINK), (10, 30),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
        #         cv2.putText(frame, "EAR: {:.2f}".format(ear), (150, 30),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
        #     cv2.imshow("Frame", frame)
        #     cv2.waitKey(int(rate))
        # cv2.destroyAllWindows()
        vs.release()
        return TOTAL_BLINK

    def check_uri_face_recognition(self, video_uri):
        # 读取视频
        TOTAL_BLINK = 0
        COUNTER = 0

        print("[INFO] starting video stream thread...")
        # 湖南直播数据rtmp://58.200.131.2:1935/livetv/hunantv
        vs = cv2.VideoCapture(video_uri)
        rate = vs.get(cv2.CAP_PROP_FPS)
        process_this_frame = True
        # 遍历每一帧
        while True:
            # 预处理
            frame = vs.read()[1]
            if frame is None:
                break
            (h, w) = frame.shape[:2]
            width = self.SCALE_WIDTH
            r = width / float(w)
            dim = (width, int(h * r))
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

            if process_this_frame:
                # 检测人脸
                face_landmarks_list = self.ffdr.detectFace68Features(frame)

                # 遍历每一个检测到的人脸
                for face_landmarks in face_landmarks_list:
                    # 获取坐标

                    leftEye = np.array(face_landmarks['left_eye'])
                    rightEye = np.array(face_landmarks['right_eye'])
                    leftEAR = self.eye_aspect_ratio(leftEye)
                    rightEAR = self.eye_aspect_ratio(rightEye)

                    # 算一个平均的
                    ear = (leftEAR + rightEAR) / 2.0

                    # 绘制眼睛区域
                    # leftEyeHull = cv2.convexHull(leftEye)
                    # rightEyeHull = cv2.convexHull(rightEye)
                    # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                    # 检查是否满足阈值
                    if ear < self.EYE_AR_THRESH:
                        COUNTER += 1
                    else:
                        # 如果连续几帧都是闭眼的，总数算一次
                        if COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                            TOTAL_BLINK += 1
                            # 重置
                            COUNTER = 0
                            if TOTAL_BLINK > self.BLINK_THRESH:
                                vs.release()
                                # cv2.destroyAllWindows()
                                return TOTAL_BLINK
        #     process_this_frame = not process_this_frame
        #     cv2.imshow("Frame", frame)
        #     cv2.waitKey(int(rate))
        # cv2.destroyAllWindows()
        vs.release()
        return TOTAL_BLINK

    def check_imgfiles_dlib(self, imgfiles):
        # 读取视频
        TOTAL_BLINK = 0
        COUNTER = 0
        (lStart, lEnd) = self.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = self.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        print("[INFO] starting load image frames...")

        # 遍历每一帧
        for file in imgfiles:
            file_bytes = file.read()
            frame = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype='uint8'), cv2.IMREAD_COLOR)
            (h, w) = frame.shape[:2]
            width = self.SCALE_WIDTH
            r = width / float(w)
            dim = (width, int(h * r))
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 检测人脸
            rects = self.detector(gray, 0)

            # 遍历每一个检测到的人脸
            for rect in rects:
                # 获取坐标
                shape = self.predictor(gray, rect)
                shape = self.shape_to_np(shape)
                # 分别计算ear值
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)
                # 算一个平均的
                ear = (leftEAR + rightEAR) / 2.0
                # 检查是否满足阈值
                if ear < self.EYE_AR_THRESH:
                    COUNTER += 1
                else:
                    # 如果连续几帧都是闭眼的，总数算一次
                    if COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        TOTAL_BLINK += 1
                        # 重置
                        COUNTER = 0
                        if TOTAL_BLINK > self.BLINK_THRESH:
                            return TOTAL_BLINK
        return TOTAL_BLINK

    def check_imgfiles_face_recognition(self, imgfiles):
        # 读取视频
        TOTAL_BLINK = 0
        COUNTER = 0
        print("[INFO] starting load image frames...")
        # 遍历每一帧
        for file in imgfiles:
            file_bytes = file.read()
            frame = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype='uint8'), cv2.IMREAD_COLOR)
            (h, w) = frame.shape[:2]
            width = self.SCALE_WIDTH
            r = width / float(w)
            dim = (width, int(h * r))
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            # 检测人脸
            face_landmarks_list = self.ffdr.detectFace68Features(frame)
            # 遍历每一个检测到的人脸
            for face_landmarks in face_landmarks_list:
                # 获取坐标
                leftEye = np.array(face_landmarks['left_eye'])
                rightEye = np.array(face_landmarks['right_eye'])
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)
                # 算一个平均的
                ear = (leftEAR + rightEAR) / 2.0
                # 检查是否满足阈值
                if ear < self.EYE_AR_THRESH:
                    COUNTER += 1
                else:
                    # 如果连续几帧都是闭眼的，总数算一次
                    if COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        TOTAL_BLINK += 1
                        # 重置
                        COUNTER = 0
                        if TOTAL_BLINK > self.BLINK_THRESH:
                            return TOTAL_BLINK

        return TOTAL_BLINK


if __name__ == '__main__':
    ffcl = FreezyFaceCheckLiveness()
    result = ffcl.check_uri_dlib('./test/blink.mp4')
    print(result)
