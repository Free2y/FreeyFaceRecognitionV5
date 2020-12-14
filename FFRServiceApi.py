# -*- coding:utf-8 -*-
import json
import os
import time

from flask import Flask, request, g
from gevent import pywsgi

from FFRService import FFRService
from tools.np_encoder import NpEncoder
import logging

# init server config
request_time = {}
now_time = time.strftime("%Y-%m-%d", time.localtime(time.time()))
max_post_time = 100
white_ips = []
app = Flask(__name__)

# init logger
logger = logging.getLogger('FFRService.' + __name__)
logger.setLevel(logging.INFO)
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path = './logs/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_name = log_path + rq + '.log'
logfile = log_name
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.INFO)  # 输出到file的log等级的开关
# 第三步，定义handler的输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
# 第四步，将logger添加到handler里面
logger.addHandler(fh)


def isLimited():
    return getattr(g, 'limit', False)


# add route handler
@app.before_request
def print_request_info():
    # print("请求地址：" + str(request.path))
    # print("请求方法：" + str(request.method))
    # print("---请求headers--start--")
    # print(str(request.headers).rstrip())
    # print("---请求headers--end----")
    # print("GET参数：" + str(request.args))
    # print("POST参数：" + str(request.form))
    global now_time
    global request_time
    # 限制接口使用次数
    time_day = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    if time_day != now_time:
        now_time = time_day
        request_time = {}
    remote_ip_now = request.remote_addr

    if remote_ip_now not in request_time:
        request_time[remote_ip_now] = 1
    elif request_time[remote_ip_now] > max_post_time - 1 and remote_ip_now not in white_ips:
        setattr(g, 'limit', True)
    else:
        request_time[remote_ip_now] += 1


@app.route('/api/freeyService/faceRecognition', methods=['POST'])
def freezyFaceRecognition():
    if isLimited():
        return result(999, '失败', '已经超出免费使用次数')
    start_time = time.time()
    time_now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))

    # 优先级 img_files > img_uri > imgbase64
    img_files = request.files.getlist('img_files', None)
    img_uri = request.form.get('img_uri', None)
    imgbase64 = request.form.get('imgbase64', None)
    ext = request.form.get('need_ext', 0, type=int)
    if img_uri is None and imgbase64 is None and len(img_files) == 0:
        return result(-1, "失败", '请传入正确的参数')
    else:
        try:
            results = ffrService.getFaceRecognitionResults(img_files, img_uri, imgbase64, ext)
            # print(results)
            frames = []
            for frame, id in results:
                faces = []
                if ext == 0:
                    for face_location in frame:
                        faces.append({'face_location': face_location})
                else:
                    for face_location, face_landmarks in frame:
                        faces.append({'face_location': face_location, 'face_landmarks': face_landmarks})
                frames.append({'faces': faces, 'id': id})
            log_info = {
                'ip': request.remote_addr,
                'return': frames,
                'time': time_now
            }
            logger.info(fromatJsonDumps(log_info))
            return result(0, '成功', {'frames': frames, 'speed_time': round(time.time() - start_time, 2)})
        except Exception as ex:
            error_log = result(-1, '产生了一点错误，请检查日志', str(ex))
            logger.error(error_logger(time_now, str(ex)), exc_info=True)
            return error_log


@app.route('/api/freeyService/checkMatch', methods=['POST'])
def freezyCheckMatch():
    if isLimited():
        return result(999, '失败', '已经超出免费使用次数')
    start_time = time.time()
    time_now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))

    # 优先级 sure_img_file > sure_img_uri > sure_imgbase64
    sure_img_file = request.files.get('sure_img_file', None)
    auth_img_file = request.files.get('auth_img_file', None)
    sure_img_uri = request.form.get('sure_img_uri', None)
    sure_imgbase64 = request.form.get('sure_imgbase64', None)
    auth_img_uri = request.form.get('auth_img_uri', None)
    auth_imgbase64 = request.form.get('auth_imgbase64', None)

    try:
        results = ffrService.checkMatchImage(sure_img_file, auth_img_file, sure_img_uri, auth_img_uri, sure_imgbase64,
                                             auth_imgbase64)
        # print(results)
        if results == "":
            return result(-1, '失败', '请传入正确的参数')
        elif results != 'yes' and results != 'no':
            return result(-1, '失败', results)
        log_info = {
            'ip': request.remote_addr,
            'return': results,
            'time': time_now
        }
        logger.info(fromatJsonDumps(log_info))
        return result(0, '成功', {'match': results,
                                'speed_time': round(time.time() - start_time, 2)})
    except Exception as ex:
        error_log = {'code': -1, 'message': '产生了一点错误，请检查日志', 'result': str(ex)}
        logger.error(error_logger(time_now, str(ex)), exc_info=True)
        return error_log


@app.route('/api/freeyService/checkLiveness', methods=['POST'])
def freezyCheckLiveness():
    if isLimited():
        return result(999, '失败', '已经超出免费使用次数')
    start_time = time.time()
    time_now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))

    # 优先级 img_files > video_uri
    img_files = request.files.getlist('img_files', None)
    video_uri = request.form.get('video_uri', None)
    check_type = request.form.get('check_type', 0, type=int)

    try:
        results = ffrService.checkLiveness(img_files, video_uri, check_type)
        # print(results)
        if results == "":
            return result(-1, '失败', '请传入正确的参数')
        elif results > ffrService.ffcl.BLINK_THRESH:
            liveness = 'yes'
        else:
            liveness = 'no'
        log_info = {
            'ip': request.remote_addr,
            'return': results,
            'time': time_now
        }
        logger.info(fromatJsonDumps(log_info))
        return result(0, '成功', {'liveness': liveness,
                                'speed_time': round(time.time() - start_time, 2)})
    except Exception as ex:
        error_log = {'code': -1, 'message': '产生了一点错误，请检查日志', 'result': str(ex)}
        logger.error(error_logger(time_now, str(ex)), exc_info=True)
        return error_log


@app.route('/api/freeyService/checkAuth', methods=['POST'])
def freezyCheckAuth():
    if isLimited():
        return result(999, '失败', '已经超出免费使用次数')
    start_time = time.time()
    time_now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))

    # 优先级 sure_img_file > sure_img_uri > sure_imgbase64
    sure_img_file = request.files.get('sure_img_file', None)
    sure_img_uri = request.form.get('sure_img_uri', None)
    sure_imgbase64 = request.form.get('sure_imgbase64', None)
    # 优先级 img_files > video_uri
    img_files = request.files.getlist('img_files', None)
    video_uri = request.form.get('video_uri', None)
    check_type = request.form.get('check_type', 0, type=int)

    try:
        results = ffrService.checkAuth(sure_img_file, sure_img_uri, sure_imgbase64, img_files, video_uri, check_type)
        # print(results)
        if results == "":
            return result(-1, '失败', '请传入正确的参数')
        elif results != 'yes' and results != 'no':
            return result(-1, '失败', results)
        log_info = {
            'ip': request.remote_addr,
            'return': results,
            'time': time_now
        }
        logger.info(fromatJsonDumps(log_info))
        return result(0, '成功', {'auth_pass': results,
                                'speed_time': round(time.time() - start_time, 2)})
    except Exception as ex:
        error_log = {'code': -1, 'message': '产生了一点错误，请检查日志', 'result': str(ex)}
        logger.error(error_logger(time_now, str(ex)), exc_info=True)
        return error_log


# 返回对象
def result(code, msg, data):
    resultDic = {}
    resultDic['code'] = code
    resultDic['message'] = msg
    resultDic['result'] = data
    return fromatJsonDumps(resultDic)


def error_logger(time, ex):
    error_logger = {}
    error_logger['ip'] = str(request.remote_addr)
    error_logger['time'] = time
    error_logger['api'] = str(request.path)
    error_logger['method'] = str(request.method)
    error_logger['headers'] = str(request.headers).rstrip()
    error_logger['get_args'] = str(request.args)
    error_logger['post_args'] = str(request.form)
    error_logger['file_args'] = str(request.files)
    error_logger['ex_info'] = ex
    return fromatJsonDumps(error_logger)


def fromatJsonDumps(data):
    return json.dumps(data, cls=NpEncoder, ensure_ascii=False)


if __name__ == '__main__':
    print('正在启动服务......')
    logger.info('正在启动服务......')
    ffrService = FFRService()
    server = pywsgi.WSGIServer(('0.0.0.0', 5678), app)
    print('服务已经启动')
    logger.info('服务已经启动')
    server.serve_forever()
