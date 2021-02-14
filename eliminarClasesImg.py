#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:36:46 2020

@author: francisco
"""
import glob 
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from transformacionesImagenes import aumentar
import random
import multiprocessing as mp

    
def busquedaClase(root,etiqueta):
    encontroClase=False
    for member in root.findall('object'):
        if member.find('name').text==etiqueta:
            encontroClase=True
            break
    return encontroClase    

def eliminarClases(root,etiqueta,img=None,eliminarClaseImagen=True):
    fondo=np.uint8([img[:,:,0].mean(),img[:,:,1].mean(),img[:,:,2].mean()])   
    if eliminarClaseImagen:
        img_new=img.copy()
    for member in root.findall('object'):
            if member.find('name').text==etiqueta:
                root.remove(member)
                bbx = member.find('bndbox')
                xmin = int(bbx.find('xmin').text)
                ymin = int(bbx.find('ymin').text)
                xmax = int(bbx.find('xmax').text)
                ymax = int(bbx.find('ymax').text)
                if eliminarClaseImagen:
                    img_new[ymin:ymax,xmin:xmax,:]=fondo
    if eliminarClaseImagen:
        return img_new
    else: return None
    
def eliminarClase(root,member,img=None,eliminarClaseImagen=True):
    fondo=np.uint8([img[:,:,0].mean(),img[:,:,1].mean(),img[:,:,2].mean()])   
    if eliminarClaseImagen:
        img_new=img.copy()
    bbx = member.find('bndbox')
    xmin = int(bbx.find('xmin').text)
    ymin = int(bbx.find('ymin').text)
    xmax = int(bbx.find('xmax').text)
    ymax = int(bbx.find('ymax').text)
    if eliminarClaseImagen:
        img_new[ymin:ymax,xmin:xmax,:]=fondo
    root.remove(member)
    if eliminarClaseImagen:
        return img_new
    else: return None
    
def eliminardistinto(root,etiqueta,img=None,eliminarClaseImagen=True):
    fondo=np.uint8([img[:,:,0].mean(),img[:,:,1].mean(),img[:,:,2].mean()])   
    if eliminarClaseImagen:
        img_new=img.copy()
    for member in root.findall('object'):
            if member.find('name').text!=etiqueta:
                root.remove(member)
                bbx = member.find('bndbox')
                xmin = int(bbx.find('xmin').text)
                ymin = int(bbx.find('ymin').text)
                xmax = int(bbx.find('xmax').text)
                ymax = int(bbx.find('ymax').text)
                if eliminarClaseImagen:
                    img_new[ymin:ymax,xmin:xmax,:]=fondo
    if eliminarClaseImagen:
        return img_new
    else: return None
 
def balancearDataset(distribucion,pathSave,scale_percent,incrementoC,incrementoS,incrementoH,incrementoI,desviacion,maximo):
    contador=np.uint64(distribucion[:,1].copy())
    etiquetas=distribucion[:,0].copy()
    c=0
     
    
    for etiqueta in etiquetas:
        print(str(round(100*(c/len(etiquetas))))+' %'+' Aumentando etiqueta: '+etiqueta+'...')#,end='\r')
        c+=1
        xmls=glob.glob(pathSave+'/*.xml')
        random.seed(0)
        random.shuffle(xmls)
        pool = mp.Pool(mp.cpu_count()) 
        for pathXML in xmls:
            
            pathImg=pathXML.replace('.xml','.jpg')
            img=cv2.imread(pathImg)
            imagen=img
            tree = ET.parse(pathXML)
            root = tree.getroot()
            lineaMaxima=maximo
            if np.uint64(contador[np.argmax(etiquetas==etiqueta)])>=lineaMaxima-10:
                print('Etiqueta: '+etiqueta+' se aumento a '+str(contador[np.argmax(etiquetas==etiqueta)]))
                break
            if busquedaClase(root,etiqueta)==True:
                
                img_new=eliminardistinto(root,etiqueta,img=imagen.copy(),eliminarClaseImagen=True)
                conta=pool.apply_async(aumentar, args=(pathImg,img_new,tree,root,pathSave,scale_percent,incrementoC,incrementoS,incrementoH,incrementoI,desviacion))
                #conta=aumentar(pathImg,img_new,tree,root,pathSave,scale_percent,incrementoC,incrementoS,incrementoH,incrementoI,desviacion)
                
                contador=contador+(etiquetas==etiqueta)*conta.get()
        pool.close()
        pool.join()          
    return contador       
                

 
          
            
#         else: continue    
        