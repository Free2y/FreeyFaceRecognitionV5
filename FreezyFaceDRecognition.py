#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import face_recognition
import numpy as np

class FreezyFaceDRecognition():

    def __init__(self):
        pass

    # top, right, bottom, left
    def detectFaceLocations(self, frame, model=''):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if model == 'cnn':
            face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
        else:
            face_locations = face_recognition.face_locations(image)
        print("I found {} face(s) in this photograph.".format(len(face_locations)))
        # for face_location in face_locations:
        #     # Print the location of each face in this image
        #     top, right, bottom, left = face_location
        #     print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,
        #                                                                                                 right))
        #
        #     # You can access the actual face itself like this:
        #     face_image = image[top:bottom, left:right]
        #     pil_image = Image.fromarray(face_image)
        #     pil_image.show()
        return face_locations

    # chin,left_eyebrow,right_eyebrow,nose_bridge,nose_tip
    # left_eye,right_eye,top_lip,bottom_lip
    def detectFace68Features(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_landmarks_list = face_recognition.face_landmarks(image)
        # print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))
        # for face_landmarks in face_landmarks_list:
        #     # Print the location of each facial feature in this image
        #     for facial_feature in face_landmarks.keys():
        #         print("The {} in this face has the following points: {}".format(facial_feature,
        #                                                                         face_landmarks[facial_feature]))

        return face_landmarks_list

    def drSingleFrame(self, frame, ext = 0):
        if ext == 0:
            return self.detectFaceLocations(frame)
        else:
            face_locations = self.detectFaceLocations(frame)
            face_landmarks_list = self.detectFace68Features(frame)
            return list(zip(face_locations, face_landmarks_list))


def showSingleFrame(frame, ffdr, ext):
    point_size = 1
    point_color = (171, 207, 49)  # BGR
    thickness = 4  # 可以为 0 、4、8
    results = ffdr.drSingleFrame(frame, ext)
    print(results)
    frameHeight = frame.shape[0]
    if len(results) == 0:
        print('抱歉，未检测到人脸')
    else:
        if ext == 0:
            for face_location in results:
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (171, 207, 49), int(round(frameHeight / 240)), 8)
        else:
            for face_location, face_landmarks in results:
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (171, 207, 49), int(round(frameHeight / 240)), 8)
                for facial_feature in face_landmarks.keys():
                    feature = np.array(face_landmarks[facial_feature])
                    featureHull = cv2.convexHull(feature)
                    cv2.drawContours(frame, [featureHull], -1, (0, 255, 0), 1)

    cv2.imshow("Face Detection Comparison", frame)


if __name__ == '__main__':
    video_capture = cv2.VideoCapture(
        'https://ss3.bdstatic.com/70cFv8Sh_Q1YnxGkpoWK1HF6hhy/it/u=232249614,3151086243&fm=26&gp=0.jpg')

    ffdr = FreezyFaceDRecognition()
    ext = 1
    # 图片
    frame = cv2.imread('./test/face.jpg')
    showSingleFrame(frame, ffdr, ext)
    cv2.waitKey(0)
    # 视频
    # while True:
    #     flag, frame = video_capture.read()
    #     if flag:
    #         showSingleFrame(frame, ffdr, ext)
    #         cv2.waitKey(1)
    #     else:
    #         break

    cv2.destroyAllWindows()
