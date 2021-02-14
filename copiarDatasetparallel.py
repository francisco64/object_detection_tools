#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 11:56:28 2020

@author: francisco
"""
import xml.etree.ElementTree as ET
from clasesUnicas import obtenerClasesFaltantes
import numpy as np
import pandas as pd
import glob
import cv2
import os
from eliminarClasesImg import balancearDataset, eliminarClase
from generate_label_map import generate
from checkSizeBB import checkbbox
from XML2PD import xml2pd
import multiprocessing as mp

def cleanImage(pathXML,reducirMaximos,clasesReducir,decision_borrado,contadores,clasesFaltantes,etiquetaMarca,distribucionMarcas,df_catalogo,Tclase,distribucionNombres,pathSave):
    img=cv2.imread(pathXML.replace('.xml','.jpg'))
    if img is not None and checkbbox(pathXML):
        
        mytree = ET.parse(pathXML) 
        myroot = mytree.getroot() 
        # iterating throught the price values. 
        for member in myroot.findall('object'):
            # updates the price value 
            bbx = member.find('bndbox')
            xmin = int(bbx.find('xmin').text)
            ymin = int(bbx.find('ymin').text)
            xmax = int(bbx.find('xmax').text)
            ymax = int(bbx.find('ymax').text)
            if reducirMaximos:
                if member.find('name').text in clasesReducir:
                    decision=decision_borrado[np.argmax(clasesReducir==member.find('name').text)][contadores[np.argmax(clasesReducir==member.find('name').text)]]
                    contadores[np.argmax(clasesReducir==member.find('name').text)]+=1
                    if decision==True:
                        img=eliminarClase(myroot,member,img=img,eliminarClaseImagen=True)
                
            
            try:
                if not(member.find('name').text in clasesFaltantes):
                    if etiquetaMarca:
                        elimClase=distribucionMarcas['Cantidad'][df_catalogo['Marca'][df_catalogo['COD_DN']==member.find('name').text].values[0]]<=Tclase
                    else:    
                        elimClase=distribucionNombres['Cantidad'][distribucionNombres['Codigo']==member.find('name').text].values[0]<=Tclase
            except:
                img=eliminarClase(myroot,member,img=img,eliminarClaseImagen=True)
                continue
            
            if (member.find('name').text in clasesFaltantes) or elimClase:
                img=eliminarClase(myroot,member,img=img,eliminarClaseImagen=True)
                continue
            
            # if generateDataframe==True:
            #     relacionClases.append([pathXML.split('/')[-1].replace('.xml','.jpg'),
            #                             df_catalogo['Marca'][df_catalogo['COD_DN']==member.find('name').text].values[0],
            #                             df_catalogo['sku Para Entrenamiento'][df_catalogo['COD_DN']==member.find('name').text].values[0], 
            #                             member.find('name').text,
            #                             xmin,
            #                             ymin,
            #                             xmax,
            #                             ymax])
            if etiquetaMarca:        
                member.find('name').text=df_catalogo['Marca'][df_catalogo['COD_DN']==member.find('name').text].values[0]
        if len(myroot.findall('object'))>0:
            myroot.find('filename').text=pathXML.replace('.xml','.jpg').split('/')[-1]
            mytree.write(pathSave+'/'+pathXML.split('/')[-1])
            cv2.imwrite(pathSave+'/'+pathXML.replace('.xml','.jpg').split('/')[-1],img)
          

def getDataset(path,pathSave,clasesFaltantes,distribucionMarcas,df_catalogo,etiquetaMarca,Tclase,reducirMaximos,maxClase,marcas,distribucionNombres):
    #c=0
    #relacionClases=[]  
    
    if etiquetaMarca:
        distibucionMarcas_list=[]
        for i in marcas:
            if distribucionMarcas['Cantidad'][i]>Tclase:
                distibucionMarcas_list.append([i,np.uint64(distribucionMarcas['Cantidad'][i])])
        distribucion=np.array(distibucionMarcas_list)
    else:
        aux_etiqueta=distribucionNombres[['Codigo','Cantidad']]
        distribucion=aux_etiqueta[aux_etiqueta['Cantidad']>Tclase].to_numpy()
    
    #solo si es entrenamiento
    if reducirMaximos:
        clasesReducir=distribucion[distribucion[:,1]>=(maxClase)*(np.max(distribucion[:,1])),0]
        cantidadesReducir=distribucion[distribucion[:,1]>=(maxClase)*(np.max(distribucion[:,1])),1]
        prob_borrado=1-np.reciprocal(distribucion[distribucion[:,1]>=(maxClase)*(np.max(distribucion[:,1])),1])*(maxClase*np.max(distribucion[:,1]))
        clasesRemoveProb=np.concatenate((clasesReducir.reshape(len(clasesReducir),1),cantidadesReducir.reshape(len(cantidadesReducir),1),prob_borrado.reshape(len(prob_borrado),1)),axis=1)
        decision_borrado=[]
        contadores=np.uint64(np.zeros(len(clasesReducir)))
        for _,cant,prob in clasesRemoveProb:
            decision_borrado.append(np.random.uniform(0,1,cant)<prob)
            
    else:
        clasesReducir=None
        decision_borrado=None
        contadores=None    
            
    pool = mp.Pool(mp.cpu_count())      
    print('Cleaning dataset...')
    print(str(mp.cpu_count())+' CPU cores were found and will be used')
    for pathXML in glob.glob(path+'/*.xml'):
        pool.apply_async(cleanImage, args=(pathXML,reducirMaximos,clasesReducir,decision_borrado,contadores,clasesFaltantes,etiquetaMarca,distribucionMarcas,df_catalogo,Tclase,distribucionNombres,pathSave))
        #cleanImage(pathXML,reducirMaximos,clasesReducir,decision_borrado,contadores,clasesFaltantes,etiquetaMarca,distribucionMarcas,df_catalogo,Tclase,distribucionNombres,pathSave)
        
        
        #c+=1
        #print(str(round(100*(c/len(glob.glob(path+'/*.xml')))))+' % archivo: '+pathXML+'-> '+pathSave)#,end='\r')
    pool.close()
    pool.join()    
            
    # if generateDataframe==True:
    #     df = pd.DataFrame(data=np.array(relacionClases), columns=["imagen", "marca","etiqueta","clase","xmin","ymin","xmax","ymax"])    
    
     
    baseDatos=xml2pd(pathSave) 
    class_distribution=baseDatos['Class'].value_counts()
    
    dist_aux=[]
    for i in distribucion[:,0]:
        try:
            dist_aux.append([i,np.uint64(class_distribution[i])])
        except:
            continue
        
    distribucion=np.array(dist_aux)    
    return distribucion



Tclase=0
maxClase=1
paths=['/home/francisco/Desktop/ObjectDetectionPeru/train','/home/francisco/Desktop/ObjectDetectionPeru/datasets/dataset_corregido/12']
augment=False
reduce=False
pathLabelMap='/home/francisco/Desktop/datasets/label_map2.pbtxt'
pathsSave=[path+'CleanData2'+str(Tclase) for path in paths] 

anotaciones_list=[paths[0]]
rutaCatalogo='/home/francisco/Desktop/ObjectDetectionPeru/catalogo.xlsx'
path_distribution='/home/francisco/Desktop/ObjectDetectionPeru/distribution.npy'
clasesFaltantes,distribucionNombres,distribucionMarcas,marcas=obtenerClasesFaltantes(anotaciones_list,rutaCatalogo)
df_catalogo=pd.read_excel(rutaCatalogo)    

#generateDataframe=True
etiquetaMarca=False

scale_percent =[1,0.9,0.8]
incrementoC=[-0.3,0.1,0.3]
incrementoS=[-0.2,0.5,0.8]
incrementoH=[-0.05,-0.02,0.01,0.02,0.05]#variaciones deben ser muy pequeñas
incrementoI=[0.1,0.5,0.7]
desviacion=[9,11,15]
distribuciones=[]
reducirMaximos=False
if etiquetaMarca:
    maximo=np.max(distribucionMarcas)['Cantidad']*maxClase
else: maximo= np.max(distribucionNombres)['Cantidad']*maxClase  
 
for i in range(0,len(paths)):
    path=paths[i]
    
    pathSave=pathsSave[i]
    
    os.system('mkdir '+pathSave)
    
    #def generateXML(path,pathSave,conversionClasesPath,generateDataframe=True):
    
    
    if i==0:
        if etiquetaMarca:
            print('Las siguientes marcas serán borradas debido a que su cantidad de apariciones es menor o igual a '+str(Tclase)+' \n')
            print(distribucionMarcas[distribucionMarcas['Cantidad']<=Tclase])
        else:
            print('Las siguientes etiquetas serán borradas debido a que su cantidad de apariciones es menor o igual a '+str(Tclase)+' \n')
            print(distribucionNombres['Codigo'][distribucionNombres['Cantidad']<=Tclase])    
    if reduce:
        reducirMaximos=True if i==0 else False  
    
    distribucion=getDataset(path,pathSave,clasesFaltantes,distribucionMarcas,df_catalogo,etiquetaMarca,Tclase,reducirMaximos,maxClase,marcas,distribucionNombres)
    
    distribucion=np.array([[x,v] for v,x in sorted(zip(distribucion[:,1],distribucion[:,0]))][::-1])
    distribuciones.append(distribucion)
    if i==0:
        np.save(path_distribution,distribucion)
        print('Dataset distribution of the training set saved to '+path_distribution+' This file will be required for evaluation')
        if augment:
            print('Balancing training dataset...')
            distribucionBalanceada=balancearDataset(distribucion,pathSave,scale_percent,incrementoC,incrementoS,incrementoH,incrementoI,desviacion,maximo)  

    
generate(distribuciones[0][:,0],pathLabelMap)
      