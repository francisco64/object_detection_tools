#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 13:56:52 2020

@author: francisco
"""
import cv2
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import glob


path_save='./'
IMAGE_PATHS='/home/francisco/Desktop/ObjectDetectionPeru/imagen'
for imgFile in glob.glob(IMAGE_PATHS + '/*.jpg'):
    

    
    image=cv2.imread(imgFile)
    PATH_TO_LABELS = '/home/francisco/Desktop/pretrained/label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)
    tree = ET.parse(imgFile.replace('.jpg','.xml'))
    root = tree.getroot()
    xml_list=[]
    for member in root.findall('object'):
        bbx = member.find('bndbox')
        xmin = int(bbx.find('xmin').text)
        ymin = int(bbx.find('ymin').text)
        xmax = int(bbx.find('xmax').text)
        ymax = int(bbx.find('ymax').text)
        label = member.find('name').text
    
        value = (root.find('filename').text,
                 int(root.find('size')[0].text),
                 int(root.find('size')[1].text),
                 label,
                 xmin,
                 ymin,
                 xmax,
                 ymax
                 )
        xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                       'Class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    
    p_score=np.array([1]*len(xml_list))
    
    p_bbox=np.concatenate((xml_df['ymin'].to_numpy().reshape(len(xml_df['ymin']),1),
                           xml_df['xmin'].to_numpy().reshape(len(xml_df['xmin']),1),
                           xml_df['ymax'].to_numpy().reshape(len(xml_df['ymax']),1),
                           xml_df['xmax'].to_numpy().reshape(len(xml_df['xmax']),1)),axis=1)
    
    my_dict=category_index
    labelInt=[]
    for label in xml_df['Class']:
        for i in range(len(my_dict)):
            if my_dict[i+1]['name']==label:
                labelInt.append(i+1)
     
    p_labelInt=np.array(labelInt)
    
    image_with_detections = image.copy()
        
        #SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
    #try:
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_with_detections,
          p_bbox,
          p_labelInt,
          p_score,
          category_index,
          use_normalized_coordinates=False,
          max_boxes_to_draw=200,
          min_score_thresh=0.1,
          agnostic_mode=False)
             
             
         # print('Done prediccion para: '+IMAGE_PATHS.split('/')[-1])
         # # DISPLAYS OUTPUT IMAGE
    print('imagen creada')
    cv2.imwrite(path_save+'/'+imgFile.split('/')[-1], image_with_detections)     
    # except:
    #     continue