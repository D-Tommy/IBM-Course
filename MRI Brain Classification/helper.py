# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 11:55:41 2021

@author: Tommy
"""

import os 
import cv2
import numpy as np


def fetch_resize_save(pick_path, save_folder, resize_size, sample_type = ''):
    filenames = []
    images = []
    for tumor_type_folder in os.listdir(pick_path):
        tumor_type = tumor_type_folder.split('_')[0]
        count = 0
        for filename in os.listdir(pick_path+f'/{tumor_type_folder}'):
            filenames.append(filename) # path pointing to images.
            image = cv2.imread(os.path.join(pick_path, tumor_type_folder, filename)) # Fetching images
            image = cv2.resize(image, resize_size, interpolation = cv2.INTER_CUBIC) #resizing to 240*240
            
            #building complete save_path
            list_of_digit = [int(i) for i in list(filename) if i.isdigit()]
            tumor_id = ''
            for digit in list_of_digit:
                tumor_id += str(digit)
            save_path = save_folder + f'/{sample_type}/{tumor_type}_{tumor_id}_{count}.jpg'
            res = cv2.imwrite(save_path, image)
            assert(res == True)
            images.append(image)
            count += 1 
    print(f'total number of images loaded : {len(images)}')