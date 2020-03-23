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

import config as cfg
# from pf.positive_filter import PositiveFilter
from oneshotmatching import OneShotMatching


#获取mask轮廓，输出格式 [轮廓1, 轮廓2, ...], 每个轮廓是个由坐标点对构成的list。
def get_contours(img):
    cv2.imwrite("try3.jpg", img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray=cv2.threshold(gray,245,255,cv2.THRESH_BINARY) #大于100的像素变为255，其余变为0
    gray_temp = gray.copy() #copy the gray image because function
    cv2.imwrite('try2.jpg', gray)
    #findContours will change the imput image into another  
    contours, _= cv2.findContours(gray_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    
    return  contours

def sort_contours(contours):
    num_list = [len(contours[i]) for i in range(len(contours))]
    num_np = np.array(num_list)
    id_sort = np.argsort(-num_np)
    top_contours = [contours[i] for i in id_sort[0:5]] if len(id_sort) >= 5 else [contours[i] for i in id_sort]
    return top_contours


def main(input_dir):
    img = cv2.imread(input_dir) # (h, w, c) = (720, 1280, 1)

    contours = get_contours(img) # （轮廓数 , 轮廓里的点数 , x坐标, y坐标） 

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
        

if __name__ == "__main__":
    # filename = os.path.join(cfg.save_dir_helmets, 'part2_000119.jpg')
    # imgname = '/data/yellow_hats/train_imgs_yellow/part2_000119.jpg'
    # img = cv2.imread(imgname)
    # bboxes = main(filename)
    # print(img)
    # print(bboxes)
    # for box in bboxes:
    #     img = cv2.rectangle(img, (box[0], box[2]), (box[1], box[3]), (0, 255, 0), 2)
    # cv2.imwrite('try.jpg', img)
    root_dir ='/data/one-shot-segmentation/output/helmets/pedastrains'
    filenames = []
    for img in os.listdir(root_dir):
        if img.endswith('jpg'):
            filenames.append(os.path.join(root_dir, img))
    filenames.sort()

    # pf = PositiveFilter(model_path=cfg.pf_path)
    pf = OneShotMatching()

    gallery = '/data/yellow_hats/Helmet_temp.jpg'
    # gallery = os.path.join('/data/videos_data/gallery/1.jpg')
    # pf.initialize_gallery(gallery)

    det_dict = {}
    det_dict['items']=[]
    pbar = tqdm(total=len(filenames))
    bbox_num = 0
    img_with_boxes = 0
    # @param: the avearge activitaion within the obtained bounding box msut be over a threshold
    actvn_thresh = 255/1.2

    save_dir = './vis/helmets/pedastrians/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for filename in filenames:
        savename = os.path.basename(filename)
        savepath = os.path.join(save_dir, savename)
        imgpath = os.path.join(cfg.pedastrians, savename)
        # print(imgpath)
        img = cv2.imread(imgpath)
        try:
            height, width, _ = img.shape
        except AttributeError:
            continue
        item = {}
        item['imgpath'] = imgpath


        
        bboxes = main(filename)
        print(bboxes)
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
                    if bbox[1]-bbox[0]> width/50 and bbox[3]-bbox[2]>height/50: 
                        qualified_boxes.append(bbox)

            if len(qualified_boxes) <= 0:
                item['bbox'] = None
                det_dict['items'].append(item)
            else:
                # Use pf to rank the remaining boxes 
                final_boxes = []
                if np.any(img):
                    for bbox in qualified_boxes:
                        query = img[bbox[2]:bbox[3], bbox[0]:bbox[1], :] # h*w*c
                        if np.any(query):
                            similarity = pf.get_similarity(gallery, query) 
                            print(similarity)
                            if similarity.any() >= cfg.pf_thresh:
                                final_boxes.append(bbox)
                                # item['bbox'] = box
                                bbox_num += 1
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
                
                if len(final_boxes) > 0:
                    img_with_boxes += 1
                    for b in final_boxes:
                        img = cv2.rectangle(img, (b[0], b[2]), (b[1], b[3]), (255, 0, 0), 2)
                    # img = cv2.rectangle(img, (box[0], box[2]), (box[1], box[3]), (0, 255, 0), 2)
                    cv2.imwrite(savepath, img)
                    item['bbox'] = final_boxes
                    item['height'] = height
                    item['width'] = width
                    det_dict['items'].append(item)
                else:
                    item['bbox'] = None
                    item['height'] = height
                    item['width'] = width
                    det_dict['items'].append(item)

        print('bbox_num:', bbox_num)
        print('img with boxes:', img_with_boxes)
        pbar.update()
    pbar.close()

    with open('./det_files/det_helmet_yellow_pedastrian.hw.pf.0.5.json', 'w') as f:
        json.dump(det_dict, f)

