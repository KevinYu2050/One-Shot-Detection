# -*- coding: utf-8 -*-
 
import cv2
import argparse
from math import *
import numpy as np
from glob import glob
import random
import os
from rotate_img import rotate_bound_white_bg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', '-o', default=str(1), type=str, required=True)
    parser.add_argument('--length', '-l', default=1, type=int, required=True)
    args = parser.parse_args()
    obj_name = args.object
    length = args.length

    obj365_filenames = glob('./train_data/tibet/flag-tibet/obj365_paste/*.jpg')
    random.shuffle(obj365_filenames)
    obj365_filenames = obj365_filenames[:length]
    augmented_filenames = glob('./train_data/video_120/Augmented{}/*.png'.format(obj_name))
    print(augmented_filenames)
    angles = range(-90, 91, 10)
    augmented_filenames.sort()
    save_img_dir = './train_data/video_120/JPEGImages-Augmented/{}/'.format(obj_name)
    save_anno_dir = './train_data/video_120/Annotations-Augmented/{}/'.format(obj_name)
    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)
    if not os.path.exists(save_anno_dir):
        os.mkdir(save_anno_dir) 

    for obj365_filename in obj365_filenames:
        augmented_filename = random.choice(augmented_filenames)
        angle = random.choice(angles)
        obj365_basename = os.path.basename(obj365_filename)
        obj365_basename = obj365_basename.split('.')[0] + obj365_basename.split('.')[1]
        r_basename = os.path.basename(augmented_filename).split('.')[0]


        print(augmented_filename)
        obj365_img = cv2.imread(obj365_filename)
        img = cv2.imread(augmented_filename)
        r_img = rotate_bound_white_bg(img, angle)
        obj365_h, obj365_w, _ = obj365_img.shape
        randx = -1
        randy = -1
        while randx<0.1 or randx > 0.9:
            randx = random.normalvariate(0.5,1)
        while randy<0.1 or randy > 0.9:
            randy = random.normalvariate(0.5,1)
        
        randx = int(randx*obj365_w*2/3)
        randy = int(randy*obj365_h*2/3)

        randw = -1
        randh = -1
        while randw<0.1 or randw > 0.6:
            randw = random.normalvariate(0.35,1)

        randw = int((obj365_w -randx)*randw)
        r_h, r_w, _ = r_img.shape
        randh = int(r_h/r_w*randw)

        try:
            r_img = cv2.resize(r_img, (randw, randh), interpolation=cv2.INTER_CUBIC)


            img_mask = np.zeros((obj365_h, obj365_w, 3), dtype=int)
            for i in range(randy, randy+randh):
                for j in range(randx, randx+randw):
                    if r_img[i-randy, j- randx, 0]==0:
                        if i<obj365_h and j<obj365_w:
                            img_mask[i,j,:] = [0, 0, 0]
                    else:
                        if i<obj365_h and j<obj365_w:
                            img_mask[i,j,:] = [255, 255, 255]
                            obj365_img[i,j,:] = r_img[i-randy, j- randx, :]

            cv2.imwrite(os.path.join(save_anno_dir, 'mask-{}-{}.png'.format(obj365_basename, r_basename)), img_mask)
            cv2.imwrite(os.path.join(save_img_dir, 'img-{}-{}.png'.format(obj365_basename, r_basename)), obj365_img)
        
        except: 
            pass

if __name__ == "__main__":
    main()







        