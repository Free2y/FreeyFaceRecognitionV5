
import cv2
import requests
import numpy as np
from tools.mat_base64_cov import img_to_base64,frame_to_base64

def showFrame(frame,frameinfo,ext):
    point_size = 1
    point_color = (171, 207, 49)  # BGR
    thickness = 4  # 可以为 0 、4、8
    frameHeight = frame.shape[0]
    faces = frameinfo['faces']
    id = frameinfo['id']
    if ext == 0:
        for face in faces:
            face_location = face['face_location']
            cv2.rectangle(frame, (face_location[3], face_location[0]), (face_location[1], face_location[2]), (171, 207, 49), int(round(frameHeight / 240)), 8)
    else:
        for face in faces:
            face_location = face['face_location']
            face_landmarks = face['face_landmarks']
            cv2.rectangle(frame, (face_location[3], face_location[0]), (face_location[1], face_location[2]), (171, 207, 49), int(round(frameHeight / 240)), 8)
            for facial_feature in face_landmarks.keys():
                feature = np.array(face_landmarks[facial_feature])
                featureHull = cv2.convexHull(feature)
                cv2.drawContours(frame, [featureHull], -1, (0, 255, 0), 1)

    cv2.imshow("id", frame)

if __name__ == '__main__':
    host = 'http://127.0.0.1'
    fr_url = host+':5678/api/freeyService/faceRecognition'
    cm_url = host+':5678/api/freeyService/checkMatch'
    cl_url = host+':5678/api/freeyService/checkLiveness'
    ca_url = host+':5678/api/freeyService/checkAuth'
    ext = 1
    frame = cv2.imread('./face.jpg')
    img_b64 = img_to_base64('./face.jpg')
    res = requests.post(url=fr_url,data={'imgbase64':img_b64,'need_ext':ext})
    r = res.json()
    print(r)
    result = r['result']
    print(result['speed_time'])
    frames = result['frames']
    for frameinfo in frames:
        showFrame(frame,frameinfo,ext)
        cv2.waitKey(0)
    # video_capture = cv2.VideoCapture('https://ss0.bdstatic.com/70cFuHSh_Q1YnxGkpoWK1HF6hhy/it/u=1869556629,1229702275&fm=11&gp=0.jpg')
    # while True:
    #     flag, frame = video_capture.read()
    #     if flag:
    #         res = requests.post(url=fr_url,data={'img_uri':'https://ss0.bdstatic.com/70cFuHSh_Q1YnxGkpoWK1HF6hhy/it/u=1869556629,1229702275&fm=11&gp=0.jpg','need_ext':ext})
    #         r = res.json()
    #         print(r)
    #         result = r['result']
    #         print(result['speed_time'])
    #         frames = result['frames']
    #         for frameinfo in frames:
    #             showFrame(frame,frameinfo,ext)
    #             cv2.waitKey(0)
    #     else:
    #         break

    cv2.destroyAllWindows()

