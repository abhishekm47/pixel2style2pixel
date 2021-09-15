
import cv2
import numpy as np
from skimage import transform as trans
from PIL import Image

try:
    from hprams import Pixel2Style2PixelConfig as config
except:
    class Pixel2Style2PixelConfig:
        padding=0.25
        size=256
        moving=0.0
        model_path = "service/scripts/pixel2style2pixel/pretrained_models/psp_celebs_super_resolution.pt"
        resize_factors = 1
        output_size = 1024
        resize_outputs = False
        
    config = Pixel2Style2PixelConfig

def face_recover(img, M, ori_img):
    # img:rgb, ori_img:bgr
    # dst:rgb
    dst = ori_img.copy()
    cv2.warpAffine(img, M, (dst.shape[1], dst.shape[0]), dst,
                   flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_TRANSPARENT)
    return dst


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def tensor2im(var):
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def dlib_alignment(img, landmarks, padding=0.25, size=256, moving=0.0):
    x_src = np.array([0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
                      0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
                      0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
                      0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
                      0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
                      0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
                      0.553364, 0.490127, 0.42689])
    y_src = np.array([0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
                      0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
                      0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
                      0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
                      0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
                      0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
                      0.784792, 0.824182, 0.831803, 0.824182])
    x_src = (padding + x_src) / (2 * padding + 1)
    y_src = (padding + y_src) / (2 * padding + 1)
    y_src += moving
    x_src *= size
    y_src *= size

    src = np.concatenate([np.expand_dims(x_src, -1), np.expand_dims(y_src, -1)], -1)
    dst = landmarks.astype(np.float32)
    src = np.concatenate([src[10:38, :], src[43:48, :]], axis=0)
    dst = np.concatenate([dst[27:55, :], dst[60:65, :]], axis=0)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]

    warped = cv2.warpAffine(img, M, (size, size), borderValue=0.0)

    return warped, M


def dlib_detect_face(img, landmarks, image_size=(config.size, config.size), padding=config.padding, moving=config.moving):
    
    img_aligned, M = dlib_alignment(img, landmarks, size=image_size[0], padding=padding, moving=moving)
    return img_aligned, M



def read_cv2_img(path):
    '''
    Read color images
    :param path: Path to image
    :return: Only returns color images
    '''
    img = cv2.imread(path, -1)

    if img is not None:
        if len(img.shape) != 3:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img




