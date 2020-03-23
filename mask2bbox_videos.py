#coding=utf-8
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
import glob
import json

import sys
import os
import argparse 
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'pf'))
sys.path.append(os.path.join(os.getcwd(), 'pf', 'networks'))
sys.path.append(os.path.join(os.getcwd(), 'pf', 'networks'))
print(sys.path)

import config as cfg
from pf.positive_filter import PositiveFilter

#获取mask轮廓，输出格式 [轮廓1, 轮廓2, ...], 每个轮廓是个由坐标点对构成的list。
def get_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray=cv2.threshold(gray,253,255,cv2.THRESH_BINARY) #大于100的像素变为255，其余变为0
    gray_temp = gray.copy() #copy the gray image because function
    
    #findContours will change the imput image into another  
    contours, _= cv2.findContours(gray_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    
    return  contours

def sort_contours(contours):
    num_list = [len(contours[i]) for i in range(len(contours))]
    num_np = np.array(num_list)
    id_sort = np.argsort(-num_np)

    top_contours = [contours[i] for i in id_sort[0:3]] if len(id_sort) >= 5 else [contours[i] for i in id_sort]
    return top_contours


def generate_bbox(input_dir):
    img = cv2.imread(input_dir) # (h, w, c) = (720, 1280, 1)

    
    #kernel = np.ones((5,5),np.uint8)
    #kernel = np.ones((2,2),np.uint8)

    #img = cv2.dilate(img, kernel, iterations = 1)
    contours = get_contours(img) # （轮廓数 , 轮廓里的点数 , x坐标, y坐标） 
    #exit()
    if len(contours) == 0:
        return None
    else:
        sorted_contours = sort_contours(contours)
        bboxes = []
        for contour in sorted_contours:
            xmin=int(np.min(contour[:,0,0]))
            xmax=int(np.max(contour[:,0,0]))
            ymin=int(np.min(contour[:,0,1]))
            ymax=int(np.max(contour[:,0,1]))
            bboxes.append([xmin, xmax, ymin, ymax])
        
        return bboxes
    #img_new = cv2.rectangle(img, (xmin,ymin), (xmax, ymax), (0, 255, 0), 2)
        
    #cv2.imwrite(output_dir, img_new)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', '-o', default=str(1), type=str, required=True)
    args = parser.parse_args()

    obj_name = args.object
    # root_dir = './output/videos/{}'.format(obj_name)
    root_dir = './output/videos/augmented/{}'.format(obj_name)
    # root_dir = './output/videos/'

    pf = PositiveFilter(model_path=cfg.pf_path)
    gallery = os.path.join('/data/videos_data/gallery/{}.jpg'.format(obj_name))
    # gallery = os.path.join('/data/videos_data/gallery/1.jpg')
    pf.initialize_gallery(gallery)

    filenames = []
    for img in os.listdir(root_dir):
        if img.endswith('jpg'):
            filenames.append(os.path.join(root_dir, img))
    det_dict = {}
    det_dict['items']=[]
    pbar = tqdm(total=len(filenames))
    bbox_num = 0
    # @param: the avearge activitaion within the obtained bounding box msut be over a threshold
    actvn_thresh = 255/1.2

    save_dir = './vis/videos/augmented/{}'.format(obj_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for filename in filenames:
        savename = os.path.basename(filename)
        savepath = os.path.join(save_dir, savename)
        img = cv2.imread(os.path.join(filename))
        height, width, _ = img.shape
        imgpath = os.path.join(cfg.video_root, 'JPEGImages', obj_name, savepath.split('/')[-1])
        item = {}
        item['imgpath'] = imgpath
        
        # Return a series of bounding boxes based on their respective contours
        bboxes = generate_bbox(filename)
        if not bboxes:
            item['bbox'] = None
            bbox = None
            det_dict['items'].append(item)
        else:
            mask = cv2.imread(filename)
            img = cv2.imread(imgpath)
            qualified_boxes = []

            # Determine if the bbox is qualified according to scale/activation info
            for bbox in bboxes:
                if bbox:
                    # @param: flag size must be greater than 1/2500 of the image size
                    if bbox[1]-bbox[0]> width/50 and bbox[3]-bbox[2]>height/50 and np.sum(mask[bbox[2]:bbox[3],bbox[0]:bbox[1],:])/((bbox[1]-bbox[0])*(bbox[3]-bbox[2])*3) > actvn_thresh: 
                        qualified_boxes.append(bbox)

            if len(qualified_boxes) <= 0:
                item['bbox'] = None
                det_dict['items'].append(item)
            else:
                # Use pf to rank the remaining boxes 
                score = 0
                box = [0, 0, 0, 0]
                if np.any(img):
                    for bbox in qualified_boxes:
                        query = img[bbox[2]:bbox[3], bbox[0]:bbox[1], :] # h*w*c
                        if np.any(query):
                            similarity = pf.test(query)
                            if similarity >= score:
                                score = similarity
                                box = bbox 
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
                item['bbox'] = box
                bbox_num += 1
                img = cv2.rectangle(img, (bbox[0], bbox[2]), (bbox[1], bbox[3]), (0, 255, 0), 2)
                cv2.imwrite(savepath, img)
                det_dict['items'].append(item)

        print('bbox_num:', bbox_num)
        pbar.update()
        #print(type(item['bbox']))
    pbar.close()

    print('bbox_num:', bbox_num)
    with open('./det_files/augmented_det_video_test_obj{}.json'.format(obj_name), 'w') as f:
       json.dump(det_dict, f)

if __name__ == "__main__":
    main()
