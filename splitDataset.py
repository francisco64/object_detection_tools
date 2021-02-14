#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 09:35:25 2020

@author: francisco
"""

import os
import glob
import numpy as np
import random
from os import path as pth
import multiprocessing as mp


pathDataset=['/home/francisco/Desktop/ObjectDetectionPeru/datasets/dataset_corregido/07',
             '/home/francisco/Desktop/ObjectDetectionPeru/datasets/dataset_corregido/24',
             ]
particiones=[1]
noPudoCopiar=[]

comando='mkdir '

#os.system('mkdir ./train ./test')

def copy(xml_file):
    if pth.exists(xml_file) and pth.exists(xml_file.replace('.jpg','.xml')):
            comando='cp '+xml_file+' '+pathSplitedData[i]
            a=os.system(comando)
            
            b=os.system(comando.replace('.jpg.jpg','.jpg').replace('.jpg','.xml'))
            if a!=0 or b!=0:
                noPudoCopiar.append(comando)
                a=0
                b=0
            print('comando '+comando+' ha sido ejecutado correctamente')    

pathSplitedData=[]
for i in range(len(particiones)):
    if i==0:
        pathSplitedData.append('./train')
        comando+='./train '
    elif i==1:   
        pathSplitedData.append('./test')
        comando+='./test '
    else:
        pathSplitedData.append('./test'+str(i))
        comando+='./test'+str(i)+' '
os.system(comando)
rutasTodas=[]
for path in pathDataset:
    rutasTodas+=glob.glob(path + '/*.jpg')
    
random.shuffle(rutasTodas)

particiones.insert(0,0)
acumulador=0
index_particion=[]
for i in range(len(particiones)):
    acumulador+=particiones[i]

    index_particion.append(acumulador)  
    
rutasPartidas=[]

for i in range(len(index_particion)):
    if i!=len(index_particion)-1:
        rutasPartidas.append(rutasTodas[round(index_particion[i]*len(rutasTodas)):round(index_particion[i+1]*len(rutasTodas)-1)])

for i,rutas in enumerate(rutasPartidas):  
    pool = mp.Pool(mp.cpu_count()) 
    for xml_file in rutas:
        pool.apply_async(copy, args=(xml_file))
        
    pool.close()
    pool.join()    

