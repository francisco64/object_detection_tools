#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 12:46:03 2020

@author: francisco
"""

import cv2
import numpy as np



def generarContraste(imagen,incremento):
    hsv=cv2.cvtColor(imagen,cv2.COLOR_BGR2HSV)
    v2=np.uint8((hsv[:,:,2]*(1+incremento)*(hsv[:,:,2]*(1+incremento)<=255))+255*(hsv[:,:,2]*(1+incremento)>255))
    v2=np.uint8(v2)
    hsv[:,:,2]=v2
    img2=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return img2

def generarSaturacion(imagen,incremento):
    hsv=cv2.cvtColor(imagen,cv2.COLOR_BGR2HSV)
    s2=np.uint8((hsv[:,:,1]*(1+incremento)*(hsv[:,:,1]*(1+incremento)<=255))+255*(hsv[:,:,1]*(1+incremento)>255))
    hsv[:,:,1]=s2
    img2=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return img2

def generarMatiz(imagen,incremento):
    hsv=cv2.cvtColor(imagen,cv2.COLOR_BGR2HSV)
    h2=np.uint8((hsv[:,:,0]*(1+incremento)*(hsv[:,:,0]*(1+incremento)<=255))+255*(hsv[:,:,0]*(1+incremento)>255))
    hsv[:,:,0]=h2
    img2=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return img2

def generarIluminacion(imagen,incremento):
    hsv=cv2.cvtColor(imagen,cv2.COLOR_BGR2HSV)
    media=hsv[:,:,2].mean()
    v2=np.uint8((hsv[:,:,2]+(media*incremento))*((hsv[:,:,2]+(media*incremento))<=255)+255*(((hsv[:,:,2]+(media*incremento))>255)))
    v2=np.uint8(v2)
    hsv[:,:,2]=v2
    img2=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return img2

def generarBlured(imagen,desviacion):
    if desviacion%2==0: desviacion = desviacion+1
    blur = cv2.GaussianBlur(imagen,(desviacion,desviacion),0)
    return blur
    
def aumentar(pathImg,imagen,tree,root,pathSave,scale_percent,incrementoC,incrementoS,incrementoH,incrementoI,desviacion):
 
    if len(incrementoC)==0:
        contraste=False
    else: contraste=True    
    
    if len(incrementoS)==0:
        saturacion=False
    else: saturacion=True  
    
    if len(incrementoH)==0:
        matiz=False
    else: matiz=True  

    if len(incrementoI)==0:
        iluminacion=False
    else: iluminacion=True  
    
    if len(desviacion)==0:
        desviacionBloor=False
    else: desviacionBloor=True  
    
    contador=0
    pathSave=pathSave if pathSave[-1]=='/' else pathSave+'/'   
    path=pathImg
    
    for scale in scale_percent: 
        width = int(imagen.shape[1] * scale)
        height = int(imagen.shape[0] * scale)
        dim = (width, height)
        
        for member in root.findall('object'):
            bbx = member.find('bndbox')
            bbx.find('xmin').text = str(int(scale*int(bbx.find('xmin').text)))
            bbx.find('ymin').text = str(int(scale*int(bbx.find('ymin').text)))
            bbx.find('xmax').text = str(int(scale*int(bbx.find('xmax').text)))
            bbx.find('ymax').text = str(int(scale*int(bbx.find('ymax').text)))
            
        cantidadObjetos=len(root.findall('object'))    
        img = cv2.resize(imagen, dim, interpolation = cv2.INTER_AREA) 
        
        if contraste:
            for i in incrementoC:
                cv2.imwrite(pathSave+path.split('/')[-1].split('.')[0]+'-'+'cont:'+'['+str(i)+']'+'escala:'+str(scale)+'.jpg',generarContraste(img,i))
                root.find('filename').text=path.split('/')[-1].split('.')[0]+'-'+'cont:'+'['+str(i)+']'+'escala:'+str(scale)+'.jpg'
                tree.write(pathSave+path.split('/')[-1].split('.')[0]+'-'+'cont:'+'['+str(i)+']'+'escala:'+str(scale)+'.xml')
                contador+=cantidadObjetos
        if saturacion:    
            for i in incrementoS:
                cv2.imwrite(pathSave+path.split('/')[-1].split('.')[0]+'-'+'sat:'+'['+str(i)+']'+'escala:'+str(scale)+'.jpg',generarSaturacion(img,i))
                root.find('filename').text=path.split('/')[-1].split('.')[0]+'-'+'sat:'+'['+str(i)+']'+'escala:'+str(scale)+'.jpg'
                tree.write(pathSave+path.split('/')[-1].split('.')[0]+'-'+'sat:'+'['+str(i)+']'+'escala:'+str(scale)+'.xml')
                contador+=cantidadObjetos
        if matiz:    
            for i in incrementoH:
                cv2.imwrite(pathSave+path.split('/')[-1].split('.')[0]+'-'+'matiz:'+'['+str(i)+']'+'escala:'+str(scale)+'.jpg',generarMatiz(img,i))
                root.find('filename').text=path.split('/')[-1].split('.')[0]+'-'+'matiz:'+'['+str(i)+']'+'escala:'+str(scale)+'.jpg'
                tree.write(pathSave+path.split('/')[-1].split('.')[0]+'-'+'matiz:'+'['+str(i)+']'+'escala:'+str(scale)+'.xml')
                contador+=cantidadObjetos
        if iluminacion:    
            for i in incrementoI:
                cv2.imwrite(pathSave+path.split('/')[-1].split('.')[0]+'-'+'ilum:'+'['+str(i)+']'+'escala:'+str(scale)+'.jpg',generarIluminacion(img,i))
                root.find('filename').text=path.split('/')[-1].split('.')[0]+'-'+'ilum:'+'['+str(i)+']'+'escala:'+str(scale)+'.jpg'
                tree.write(pathSave+path.split('/')[-1].split('.')[0]+'-'+'ilum:'+'['+str(i)+']'+'escala:'+str(scale)+'.xml')
                contador+=cantidadObjetos
        if desviacionBloor:    
            for i in desviacion:
                cv2.imwrite(pathSave+path.split('/')[-1].split('.')[0]+'-'+'desv:'+'['+str(i)+']'+'escala:'+str(scale)+'.jpg',generarBlured(img,i))
                root.find('filename').text=path.split('/')[-1].split('.')[0]+'-'+'desv:'+'['+str(i)+']'+'escala:'+str(scale)+'.jpg'
                tree.write(pathSave+path.split('/')[-1].split('.')[0]+'-'+'desv:'+'['+str(i)+']'+'escala:'+str(scale)+'.xml')
                contador+=cantidadObjetos
    return contador
# path='./922-14244267.jpg'

# pathSave='./transformaciones' 
# scale_percent = [0.5,0.6,0.7,0.8,0.9,1] # percent of original size
# incrementoC=[-0.3,-0.2,0.1,0.2,0.3]
# incrementoS=[-0.2,0.3,0.4,0.5,0.7,0.8]
# incrementoH=[-0.05,-0.02,0.01,0.02,0.05]#variaciones deben ser muy pequeñas
# incrementoI=[0.1,0.3,0.5,0.6,0.8]
# desviacion=[5,9,11,15,17]

# aumentar(path,pathSave,scale_percent,incrementoC,incrementoS,incrementoH,incrementoI,desviacion,contraste=True, saturacion=True, matiz=True, iluminacion=True, desviacionBloor=True)
