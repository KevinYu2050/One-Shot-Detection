#coding=utf-8
import cv2
import xml.sax
import xml.etree.ElementTree as ET
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


def parser(file):
    tree = ET.parse(file)
    root = tree.getroot()
    bboxes = []
    for child in root:
        if child.tag == "object":
            name = child.find("name")
            if name.text == "hat":
                box = child.find("bndbox")
                xmin = int(box.find('xmin').text)
                xmax = int(box.find('xmax').text)
                ymin = int(box.find('ymin').text)
                ymax = int(box.find('ymax').text)
                bboxes.append([xmin, xmax, ymin, 1/2*ymax+1/2*ymin])
            else:
                pass

    return bboxes


def main(iou_thresh):
    # create an XMLReader
    det_file = './det_files/det_helmet_yellow.pf.0.5.json'
    gt_dir = cfg.anno_helmets_test
    
    files = []
    for file in np.sort(os.listdir(gt_dir)):
        files.append(os.path.join(gt_dir, file))

    gt_boxes = []
    gt_num = 0
    for file in files:
        bboxes = parser(file)
        fname = file.split('/')[-1].strip('.xml')
        gt_boxes.append({'fname':fname, 'bboxes':bboxes}) 
        # gt_num += len(bboxes)

    with open(det_file) as f:
        det_dict = json.load(f)
    
    # assert len(gt_boxes) == len(det_dict['items'])
    det_num=len(det_dict['items'])
    box_num = 0
    for box in det_dict['items']:
        try :
            if type(box['bbox']) is not None:
                print(box)
                box = box['bbox']
                if len(box) >0:
                    box_num += 1
            print("llll", box_num)
        except TypeError:
            pass

    # print(det_num, gt_num)
    #  Delete gt_boxes with no detection
    imgnames = [int(b['imgpath'].split('_')[-1].strip('.jpg')) for b in det_dict['items']]
    imgnames = [str(num) for num in imgnames]
    fnames_delete = []
    for box in gt_boxes:
        fname = str(int(box['fname'].split('_')[-1]))
        if fname not in imgnames:
            # gt_boxes.remove(box)
            fnames_delete.append(box['fname'])
    gt_boxes = [box for box in gt_boxes if box['fname'] not in fnames_delete]
    for box in gt_boxes:
        gt_num += len(box['bboxes'])
    
    print(len(imgnames), len(gt_boxes))        
    det_boxes_num = 0

    print("gt_num:", gt_num)
    ious = []
    for i in range(det_num):
        gt_item = gt_boxes[i]
        det_item = det_dict['items'][i]
        assert int(gt_item['fname'].split('_')[-1]) == int(det_item['imgpath'].split('_')[-1].strip('.jpg'))

        bboxes = det_item['bbox']

        # Calculates IoU
        if bboxes:
            det_boxes_num += len(bboxes)
            for det_box in bboxes:
                for gt_box in gt_item['bboxes']:
                    iou = get_iou(gt_box, det_box)
                    ious.append(iou)
        
    print(len(ious))
    print(det_boxes_num)

    tp = np.array(ious>iou_thresh*np.ones(len(ious)), dtype=int)

    print(tp.sum())
    print('finish')
    print('precision: | %.3f' % (tp.sum()/det_boxes_num))
    print('recall: | %.3f' % (tp.sum()/gt_num))

    # # save_dir = "./pre_rec/"
    # # if not os.path.exists(save_dir):
    # #     os.mkdir(save_dir)
    
    # with open("results/incre_results{}.txt".format("w") as f:
    #     f.write('category %s precision: | %.3f' % (tp.sum()/det_num))
    #     f.write('category %s recall: | %.3f' % (tp.sum()/gt_num))


if __name__ == "__main__":
    iou_thresh = cfg.iou_thresh
    main(iou_thresh)

