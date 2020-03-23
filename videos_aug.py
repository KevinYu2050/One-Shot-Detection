import os
import sys 
from glob import glob

# categories = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 19, 21, 22]
categories = [2]
for cat in categories:
    length = len(glob('./train_data/video_120/JPEGImages/{}/*.jpg'.format(str(cat))))
    # Augment the fine-tune images
    # os.system("python3.6 ./script/image_aug_video_fine_tune_ds.py -o {}".format(str(cat)))
    # # Augment the templates
    # os.system("python3.6 ./script/image_aug_video.py -o {}".format(str(cat)))
    # # Paste the augmented templates
    # os.system("python3.6 ./script/paste.py -o {} -l {}".format(str(cat), length//60))
    os.system("python3.6 ./script/paste.py -o {} -l {}".format(str(cat), 5))
