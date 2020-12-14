# -*- coding: utf-8 -*-
from io import BytesIO

import cv2
import base64
import numpy as np
from PIL import Image


def img_to_base64(img_path):
    with open(img_path, 'rb') as read:
        b64 = base64.b64encode(read.read())
    return b64


def base64_to_image(imageBase64):
    raw_image = base64.b64decode(imageBase64.encode('utf8'))
    image = Image.open(BytesIO(raw_image))
    return image


def base64_to_frame(imageBase64):
    raw_image = base64.b64decode(imageBase64.encode('utf8'))
    nparr = np.fromstring(raw_image, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame


def frame_to_base64(frame):
    img = Image.fromarray(frame)  # 将每一帧转为Image
    output_buffer = BytesIO()  # 创建一个BytesIO
    img.save(output_buffer, format='JPEG')  # 写入output_buffer
    byte_data = output_buffer.getvalue()  # 在内存中读取
    base64_data = base64.b64encode(byte_data)  # 转为BASE64
    return base64_data  # 转码成功 返回base64编码
