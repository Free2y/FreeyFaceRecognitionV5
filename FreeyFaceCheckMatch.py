#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import face_recognition

from FreezyFaceDRecognition import FreezyFaceDRecognition


class FreezyFaceCheckMatch():

    def __init__(self):
        self.ffdr = FreezyFaceDRecognition()

    def checkMatchImage(self, sure_image, auth_image):
        if len(self.ffdr.drSingleFrame(sure_image)) != 1:
            return '已认证图片不符合规范'
        if len(self.ffdr.drSingleFrame(auth_image)) == 0:
            return 'no'
        sure_face = cv2.cvtColor(sure_image, cv2.COLOR_BGR2RGB)
        auth_face = cv2.cvtColor(auth_image, cv2.COLOR_BGR2RGB)
        sure_face_encoding = face_recognition.face_encodings(sure_face)[0]
        auth_face_encoding = face_recognition.face_encodings(auth_face)[0]
        known_faces = [
            sure_face_encoding
        ]

        face_distances = face_recognition.face_distance(known_faces, auth_face_encoding)
        result = 'no'
        for i, face_distance in enumerate(face_distances):
            if face_distance < 0.5:
                result = 'yes'
                break
        return result


if __name__ == '__main__':
    ffcm = FreezyFaceCheckMatch()
    # 图片
    sure = cv2.imread('./test/sure.jpg')
    auth = cv2.imread('./test/auth.jpg')
    result = ffcm.checkMatchImage(sure, auth)
    print(result)
    cv2.imshow('sure', sure)
    cv2.waitKey(0)
    cv2.imshow('auth', auth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
