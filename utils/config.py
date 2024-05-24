import yaml


class Namespace(object):
    pass


config = Namespace()


def set_config(file):
    with open(file, 'r') as f:
        _config = yaml.load(f, Loader=yaml.FullLoader)

    # Env
    config.device = _config['ENV']['DEVICE']
    config.gpu_num = _config['ENV']['GPU_NUM']

    # Media
    config.media_source = str(_config['MEDIA']['SOURCE'])
    config.media_opt_auto = _config['MEDIA']['OPT_AUTO']
    config.media_fourcc = _config['MEDIA']['FOURCC']
    config.media_width = _config['MEDIA']['WIDTH']
    config.media_height = _config['MEDIA']['HEIGHT']
    config.media_fps = _config['MEDIA']['FPS']
    config.media_realtime = _config['MEDIA']['REALTIME']
    config.media_bgr = _config['MEDIA']['BGR']

    # Det
    config.det_model_type = _config['DET']['MODEL_TYPE']
    config.det_model_path = _config['DET']['DET_MODEL_PATH']
    config.det_half = _config['DET']['HALF']
    config.det_conf_thres = _config['DET']['CONF_THRES']
    config.det_obj_classes = eval(str(_config['DET']['OBJ_CLASSES']))

    # YOLOV5
    config.yolov5_img_size = _config['YOLOV5']['IMG_SIZE']
    config.yolov5_nms_iou = _config['YOLOV5']['NMS_IOU']
    config.yolov5_agnostic_nms = _config['YOLOV5']['AGNOSTIC_NMS']
    config.yolov5_max_det = _config['YOLOV5']['MAX_DET']

    # YOLOV8
    config.yolov8_img_size = _config['YOLOV8']['IMG_SIZE']
    config.yolov8_nms_iou = _config['YOLOV8']['NMS_IOU']
    config.yolov8_agnostic_nms = _config['YOLOV8']['AGNOSTIC_NMS']
    config.yolov8_max_det = _config['YOLOV8']['MAX_DET']


def get_config():
    return config
