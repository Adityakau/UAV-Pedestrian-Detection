import os
import cv2
import torch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device('cuda')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import numpy as np
_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            0.000,
            0.447,
            0.741,
            0.314,
            0.717,
            0.741,
            0.50,
            0.5,
            0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3))


import cv2
import matplotlib as mpl
import matplotlib.figure as mplfigure
import numpy as np
import math
import pycocotools.mask as mask_util
from matplotlib.backends.backend_agg import FigureCanvasAgg

_SMALL_OBJECT_AREA_THRESH = 1000
def overlay_bbox_cv(img, dets, class_names, score_thresh,ht,centroid):
    all_box = []
    for label in dets:
        for bbox in dets[label]:
            score = bbox[-1]
            if score > score_thresh:
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                #print(x0,y0)
                all_box.append([label, x0, y0, x1, y1, score])
    all_box.sort(key=lambda v: v[5])
    print(len(all_box))
    l=[]
    i=0
    for box in all_box:
        label, x0, y0, x1, y1, score = box
        # color = self.cmap(i)[:3]
        #color = (_COLORS[label] * 255).astype(np.uint8).tolist()
        text = "{}:{:.1f}%".format(class_names[label], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[label]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        temp=[]
        temp.extend([x0,y0,x1,y1])
        l.append(temp)
        color = (0, 255, 0)
        print("label :",i,"coordinates :",(x0+x1)/2,y1)
        i=i+1

        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        cv2.rectangle(
            img,
            (x0, y0 - txt_size[1] - 1),
            (x0 + txt_size[0] + txt_size[1], y0 - 1),
            color,
            -1,
        )
        cv2.putText(img, text, (x0, y0 - 1), font, 0.5, txt_color, thickness=1)
    coordinates=[]
    if(ht==10):
      H = np.array([[ 2.63838501e+00 , 1.31940946e+00 ,-2.31132884e+03],
        [ 1.41040253e-01, 4.57253773e+00, -2.35878868e+03],
        [ 9.60857297e-05 , 5.38603322e-04,  1.00000000e+00]])
      lambda_scale = 0.00374
    else:
      H = np.array([[ 3.54609972e+00,  1.25184637e+00 ,-3.31224183e+03],
         [ 4.44497355e-01 , 4.56524555e+00 ,-4.28714974e+03],
          [ 3.34794806e-04 , 5.17806197e-04 , 1.00000000e+00]])
      lambda_scale = 0.0051
    if(centroid==1):
      print('using centroid')
    else:
      print('using foot')
    for i in range(len(l)):
        x0,y0,x1,y1=l[i][0],l[i][1],l[i][2],l[i][3]
        if(centroid==1):
          centx=(x0+x1)/2
          centy=(y0+y1)/2
        else:
          centx=(x0+x1)/2
          centy=(y1)
        #print(centx,centy)
        pixel_coordinates=np.array([[centx,centy,1]])


        world_coordinates = np.dot(H, pixel_coordinates.T).T

        for row in world_coordinates:
         #formatted_row = [f"{element:.2f}" for element in row]
         #print(" ".join(formatted_row))
          normalized_coordinates = [[x / z, y / z, 1] for x, y, z in world_coordinates]

# Print the normalized coordinates
        #print('NORMALIZED COORDINATES')
        for coord in normalized_coordinates:
            coord[0]*=lambda_scale
            coord[1]*=lambda_scale
            coordinates.append(coord)
# Normalize by lambda to get real-world coordinates
        #real_world_coordinates = [[x * lambda_scale, y * lambda_scale, 1] for x, y, _ in normalized_coordinates]

    def calculate_distance(point1, point2):
      x1, y1, _ = point1  # Ignore the z-coordinate
      x2, y2, _ = point2  # Ignore the z-coordinate
      distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
      return distance

    def check_distances(coordinates, l,max_distance=2):
     n = len(coordinates)
     for i in range(n):
        for j in range(i+1, n):
            point1 = coordinates[i]
            point2 = coordinates[j]
            distance = calculate_distance(point1, point2)
           # print(point1[0]/lambda_scale,point1[1]/lambda_scale)
           # print(point2[0]/lambda_scale,point2[1]/lambda_scale)
            print(i,j,distance)
            if distance < max_distance:
                color = ( 0, 0,255)
                cv2.rectangle(img, (l[i][0], l[i][1]), (l[i][2], l[i][3]), color, 2)
                cv2.rectangle(img, (l[j][0], l[j][1]), (l[j][2], l[j][3]), color, 2)
                #print(f"Distance between {point1[:2]} and {point2[:2]} is {distance} (less than {max_distance} units).")

# Call the function to check distances
    check_distances(coordinates,l)

    return img



from nanodet.util import overlay_bbox_cv

from IPython.display import display
from PIL import Image

def cv2_imshow(a, convert_bgr_to_rgb=True):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:
        a: np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
            (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
            image.
        convert_bgr_to_rgb: switch to convert BGR to RGB channel.
    """
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if convert_bgr_to_rgb and a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display(Image.fromarray(a))

from nanodet.util import cfg, load_config, Logger
config_path = './config/nanodet-plus-m_416-yolo.yml'
model_path = '../nanodet_model_best.pth'
image_path = '../1.jpeg'
load_config(cfg, config_path)
logger = Logger(-1, use_tensorboard=False)
from demo.demo import Predictor
predictor = Predictor(cfg, model_path, logger, device=device)

meta, res = predictor.inference(image_path)

result = overlay_bbox_cv(meta['raw_img'][0], res[0], cfg.class_names, score_thresh=0.35)
imshow_scale = 1.0
cv2_imshow(cv2.resize(result, None, fx=imshow_scale, fy=imshow_scale))
result = overlay_bbox_cv(meta['raw_img'][0], res[0], cfg.class_names, score_thresh=0.35,ht=10,centroid=1)

imshow_scale = 1.0
cv2_imshow(cv2.resize(result, None, fx=imshow_scale, fy=imshow_scale))

result = overlay_bbox_cv(meta['raw_img'][0], res[0], cfg.class_names, score_thresh=0.35,ht=10,centroid=1)
imshow_scale = 1.0
cv2_imshow(cv2.resize(result, None, fx=imshow_scale, fy=imshow_scale))

result = overlay_bbox_cv(meta['raw_img'][0], res[0], cfg.class_names, score_thresh=0.35,ht=10,centroid=1)
imshow_scale = 1.0
cv2_imshow(cv2.resize(result, None, fx=imshow_scale, fy=imshow_scale))

result = overlay_bbox_cv(meta['raw_img'][0], res[0], cfg.class_names, score_thresh=0.35,ht=14,centroid=1)
imshow_scale = 1.0
cv2_imshow(cv2.resize(result, None, fx=imshow_scale, fy=imshow_scale))