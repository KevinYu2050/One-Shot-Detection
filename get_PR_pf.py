#coding=utf-8
import cv2
import numpy as np
import json

import argparse
import sys
import os
sys.path.append(os.getcwd())
print(sys.path)

import config as cfg


def get_iou(box1, box2):
    x1_min, x1_max, y1_min, y1_max = box1
    x2_min, x2_max, y2_min, y2_max = box2
    s1=(x1_max-x1_min)*(y1_max-y1_min) 
    s2=(x2_max-x2_min)*(y2_max-y2_min)
    
    xmin=max(x1_min, x2_min)
    xmax=min(x1_max, x2_max)
    ymin=max(y1_min, y2_min)
    ymax=min(y1_max, y2_max)   
    if xmin>xmax or ymin>ymax:
        return 0
    else:
        return (xmax-xmin)*(ymax-ymin)*1.0/(s1+s2-(xmax-xmin)*(ymax-ymin))






def main(iou_thresh):
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', '-o', default=str(1), type=str, required=True)
    args = parser.parse_args()

    obj_name = args.object
    det_file = './det_files/incre_video_test_conv.2_obj{}.json'.format(obj_name)
    gt_file = './gt_files/gt_files/gt_{}.json'.format(obj_name)

    with open(gt_file) as f:
        gt_dict = json.load(f)
        del gt_dict['items'][4133]



    with open(det_file) as f:
        det_dict = json.load(f)
        # del det_dict['items'][25]


    # if len(gt_dict['items'])!=len(det_dict['items']):
    #     for i in range(len(gt_dict['items'])):
    #         gt_item = gt_dict['items'][i]
    #         det_item = det_dict['items'][i]
    #         print(gt_item['imgpath'].split('/')[-1], det_item['imgpath'].split('/')[-1])
    #         # if gt_item['imgpath'].split('/')[-1] == det_item['imgpath'].split('/')[-1]:
            #     pass
            # else:
            #     del gt_item
            #     break
    
    # assert len(gt_dict['items'])==len(det_dict['items'])
    gt_num=len(gt_dict['items'])
    det_num=len(det_dict['items'])

    ious = []
    for i in range(gt_num):
        gt_item = gt_dict['items'][i]
        det_item = det_dict['items'][i]
        print(gt_item['imgpath'].split('/')[-1], det_item['imgpath'].split('/')[-1])
        assert gt_item['imgpath'].split('/')[-1] == det_item['imgpath'].split('/')[-1]

        bbox = det_item['bbox']

        # Calculates IoU
        if bbox!= None:
            iou = get_iou(gt_item['bbox'], bbox)
            ious.append(iou)
        else:
            iou=0
            det_num-=1
            ious.append(iou)
 
    print(len(ious))

    tp = np.array(ious>iou_thresh*np.ones(len(ious)), dtype=int)

    print(tp.sum())
    print('finish')
    print('category %s precision: | %.3f' % (obj_name, tp.sum()/det_num))
    print('category %s recall: | %.3f' % (obj_name, tp.sum()/gt_num))

    # save_dir = "./pre_rec/"
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    
    with open("results/incre_results{}.txt".format(obj_name), "w") as f:
        f.write('category %s precision: | %.3f' % (obj_name, tp.sum()/det_num))
        f.write('category %s recall: | %.3f' % (obj_name, tp.sum()/gt_num))


if __name__ == "__main__":
    iou_thresh = cfg.iou_thresh
    main(iou_thresh)

