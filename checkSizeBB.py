#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 10:02:14 2020

@author: francisco
"""

import cv2
import xml.etree.ElementTree as ET
import glob

def checkbbox(pathXML):
    img=cv2.imread(pathXML.replace('.xml','.jpg'))
    h,w,_=img.shape
    
    tree = ET.parse(pathXML)
    root = tree.getroot()
    for member in root.findall('object'):
        bbx = member.find('bndbox')
        xmin = int(bbx.find('xmin').text)
        ymin = int(bbx.find('ymin').text)
        xmax = int(bbx.find('xmax').text)
        ymax = int(bbx.find('ymax').text)
    
        if xmin<0 or ymin<0:
            print('minimos del BBox para '+pathXML+' son negativos')
            print('xmin: '+str(xmin)+' ymin: '+str(ymin))
            return False
        if xmax>w or ymax>h:
            print('maximos del BBox para '+pathXML+' son mayores que sus dimensiones')
            print('xmax: '+str(xmax)+' ymax: '+str(ymax)+' (h,w): '+'('+str(h)+','+str(w)+')')
            return False

    return True        

# for ruta in glob.glob('/home/francisco/Desktop/ObjectDetectionPeru/train' + '/*.xml'):
#     checkbbox(ruta)