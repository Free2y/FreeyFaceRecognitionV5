#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import face_recognition
import numpy as np

from FreeyFaceCheckAuth import FreezyFaceCheckAuth
from FreeyFaceCheckLiveness import FreezyFaceCheckLiveness
from FreeyFaceCheckMatch import FreezyFaceCheckMatch
from FreezyFaceDRecognition import FreezyFaceDRecognition
from tools.mat_base64_cov import base64_to_frame


class FFRService:

    def __init__(self):
        self.ffdr = FreezyFaceDRecognition()
        self.ffcm = FreezyFaceCheckMatch()
        self.ffcl = FreezyFaceCheckLiveness()
        self.ffca = FreezyFaceCheckAuth()

    def getFaceRecognitionResults(self, img_files, img_uri, imgbase64, ext):
        if len(img_files) > 0:
            return self.getFaceDRResultsByFiles(img_files, ext)
        elif img_uri is not None:
            return self.getFaceDRResultsByUri(img_uri, ext)
        elif imgbase64 is not None:
            return self.getFaceDRResultsByBase64(imgbase64, ext)
        else:
            return []

    def getFaceDRResultsByFiles(self, img_files, ext):
        responses = []
        for file in img_files:
            file_bytes = file.read()
            # print(file_bytes)
            image = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype='uint8'), cv2.IMREAD_COLOR)
            # cv2.imshow("a",image)
            # cv2.waitKey(0)
            results = self.ffdr.drSingleFrame(image, ext)
            responses.append((results, file.filename))

        return responses

    def getFaceDRResultsByUri(self, img_uri, ext):
        video_capture = cv2.VideoCapture(img_uri)
        responses = []
        while True:
            flag, frame = video_capture.read()
            if flag:
                results = self.ffdr.drSingleFrame(frame, ext)
                responses.append((results, img_uri))
            else:
                break
        return responses

    def getFaceDRResultsByBase64(self, imageBase64, ext):
        responses = []
        frame = base64_to_frame(imageBase64)
        results = self.ffdr.drSingleFrame(frame, ext)
        responses.append((results, imageBase64))

        return responses

    def checkMatchImage(self, sure_file, auth_file, sure_image, auth_image, sure_image_base64, auth_image_base64):
        if sure_file != None:
            file_bytes = sure_file.read()
            sure_frame = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype='uint8'), cv2.IMREAD_COLOR)
        elif sure_image != None:
            sure = cv2.VideoCapture(sure_image)
            while True:
                flag, frame = sure.read()
                if flag:
                    sure_frame = frame
                else:
                    break
        elif sure_image_base64 != None:
            sure_frame = base64_to_frame(sure_image_base64)
        else:
            return ""

        if auth_file != None:
            file_bytes = auth_file.read()
            auth_frame = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype='uint8'), cv2.IMREAD_COLOR)
        elif auth_image != None:
            sure = cv2.VideoCapture(auth_image)
            while True:
                flag, frame = sure.read()
                if flag:
                    auth_frame = frame
                else:
                    break
        elif auth_image_base64 != None:
            auth_frame = base64_to_frame(auth_image_base64)
        else:
            return ""

        results = self.ffcm.checkMatchImage(sure_frame, auth_frame)
        return results

    def checkLiveness(self, img_files, video_uri, type=0):
        if type == 0:
            if len(img_files) > 0:
                print(len(img_files))
                return self.ffcl.check_imgfiles_dlib(img_files)
            elif video_uri is not None:
                return self.ffcl.check_uri_dlib(video_uri)
            else:
                return ""
        else:
            if len(img_files) > 0:
                print(len(img_files))
                return self.ffcl.check_imgfiles_face_recognition(img_files)
            elif video_uri is not None:
                return self.ffcl.check_uri_face_recognition(video_uri)
            else:
                return ""

    def checkAuth(self, sure_img_file, sure_image, sure_image_base64, img_files, video_uri, type=0):
        auth_pass = ''

        if sure_img_file != None:
            file_bytes = sure_img_file.read()
            sure_frame = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype='uint8'), cv2.IMREAD_COLOR)
        elif sure_image != None:
            sure = cv2.VideoCapture(sure_image)
            while True:
                flag, frame = sure.read()
                if flag:
                    sure_frame = frame
                else:
                    break
        elif sure_image_base64 != None:
            sure_frame = base64_to_frame(sure_image_base64)
        else:
            return auth_pass

        if len(self.ffdr.drSingleFrame(sure_frame)) != 1:
            return '已认证图片不符合规范'
        else:
            known_faces = self.ffca.getSureImgEncoding(sure_frame)
        if type == 0:
            if len(img_files) > 0:
                print(len(img_files))
                return self.ffca.check_imgfiles_dlib(img_files, known_faces)
            elif video_uri is not None:
                return self.ffca.check_uri_dlib(video_uri, known_faces)
            else:
                return ""
        else:
            if len(img_files) > 0:
                print(len(img_files))
                return self.ffca.check_imgfiles_face_recognition(img_files, known_faces)
            elif video_uri is not None:
                return self.ffca.check_uri_face_recognition(video_uri, known_faces)
            else:
                return ""
