import numpy as np 
import argparse
import imgaug as ia
import imgaug.augmenters as iaa 
import cv2

from glob import glob
import os 
import sys
sys.path.append(os.getcwd())
import config as cfg

def augment(images):
    # The transformer
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            # crop images by -5% to 10% of their height/width
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((1, 5),
                [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    iaa.pooling.AveragePooling(kernel_size=2), # Avr Pooling
                    iaa.AddToSaturation(value=25),
                    iaa.FastSnowyLandscape(lightness_multiplier=2.0, lightness_threshold=(0, 50)) 
                ]
            )
        ]
    )

    images_aug = seq(images=images)

    return images_aug

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', '-o', default=str(1), type=str, required=True)
    args = parser.parse_args()
    obj_name = args.object

    for j in range(5):
        img_dir = glob('./train_data/video_120/JPEGImages/{}/*.jpg'.format(obj_name))
        anno_dir = glob('./train_data/video_120/Annotations/{}/*.png'.format(obj_name))
        
        img_dir.sort()
        anno_dir.sort()

        img_dir = img_dir[0:-1:300]
        anno_dir = anno_dir[0:-1:300]
        for i in range(len(img_dir)):
            assert img_dir[i].split('/')[-1].split('.')[0] == anno_dir[i].split('/')[-1].split('.')[0]
        # imgs = [(img.split('/')[-1], cv2.imread(img)) for img in img_dir]
        imgs = [cv2.imread(img) for img in img_dir]
        annos = [cv2.imread(anno) for anno in anno_dir]

        
        save_dir_img = './train_data/video_120/JPEGImages-Augmented/{}/'.format(obj_name)
        save_dir_anno = './train_data/video_120/Annotations-Augmented/{}/'.format(obj_name)


        imgs_save = []

        imgs_aug = augment(imgs)
        imgs_save += imgs_aug
        
        print(len(imgs_save))
        for i, img in enumerate(imgs_save):
            cv2.imwrite(os.path.join(save_dir_img, 'ft {} {}'.format(j, img_dir[i].split('/')[-1])), img)
            cv2.imwrite(os.path.join(save_dir_anno, 'ft mask {} {}'.format(j, anno_dir[i].split('/')[-1])), annos[i])

if __name__ == "__main__":
    main()
