#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 09:43:27 2020

@author: francisco
"""

from XML2PD import xml2pd
import pandas as pd
import numpy as np

# anotaciones_list=['/home/francisco/baseDatosObjectDetection/correccion/07', 
#                   '/home/francisco/baseDatosObjectDetection/correccion/12',
#                   '/home/francisco/baseDatosObjectDetection/correccion/24']

# anotaciones_list=['./datasetPrueba']
# rutaCatalogo='/home/francisco/Desktop/ObjectDetectionPeru/Listado Separacion SKUs Redes Final-1.xlsx'
def obtenerClasesFaltantes(anotaciones_list,rutaCatalogo):
    anotaciones_pd_list=[]
    
    for folder_anotacion in anotaciones_list:
        anotaciones_pd_list.append(xml2pd(folder_anotacion))
    
    baseDatos=pd.concat(anotaciones_pd_list)    
    clases=baseDatos['Class'].unique()
    class_distribution=baseDatos['Class'].value_counts()
    
    catalogo=pd.read_excel(rutaCatalogo)
    clases_catalogo=catalogo['COD_DN']
    
    clases_int=np.intersect1d(clases,clases_catalogo)
    
    faltantes=[]
    for clase in clases:
        faltantes.append(not(np.any(clases_int[:] == clase)))
    faltantes=np.array(faltantes)    
    
    clases_faltantes=clases[faltantes]
    distrib_faltantes=np.concatenate((clases_faltantes.reshape(len(clases_faltantes),1),np.array([class_distribution[clase] for clase in clases_faltantes]).reshape(len(clases_faltantes),1)),axis=1)
    pd_faltantes=pd.DataFrame(data=distrib_faltantes,columns=['Código_DN','Apariciones'])
    print('no se encontraron las siguientes clases en el catalogo ('+rutaCatalogo.split('/')[-1]+') y serán removidas del dataset aumentado: \n')
    print(pd_faltantes)
    print('\n')
    distribucion_nombres=[[catalogo['Marca'][catalogo['COD_DN']==clase].values[0],catalogo['sku Para Entrenamiento'][catalogo['COD_DN']==clase].values[0],clase,class_distribution[clase]] for clase in clases_int]

    distribucion_nombres=pd.DataFrame(data=distribucion_nombres,columns=['Marca','Etiqueta','Codigo','Cantidad'])
    
    distribucion_marcas=distribucion_nombres.groupby(['Marca']).sum()
    
    return clases_faltantes,distribucion_nombres,distribucion_marcas,pd.unique(distribucion_nombres['Marca'])

if __name__=='__main__':
    #anotaciones_list=['/home/francisco/Documents/trainCleanData50']
    anotaciones_list=['/home/francisco/Documents/datasetTest']
    
    rutaCatalogo='./catalogo.xlsx'
    clases_faltantes,distribucion_nombres,distribucion_marcas,marcas=obtenerClasesFaltantes(anotaciones_list,rutaCatalogo)
    distribucion_nombres=distribucion_nombres.sort_values(by=['Cantidad'])
    distribucion_marcas=distribucion_marcas.sort_values(by=['Cantidad'])
    distribucion_nombres.to_csv('./distribucion_nombres.csv')
    distribucion_marcas.to_csv('./distribucion_marcas.csv')
    
    distribution=np.concatenate((distribucion_nombres['Codigo'].to_numpy().reshape(len(distribucion_nombres),1),distribucion_nombres['Cantidad'].to_numpy().reshape(len(distribucion_nombres),1)),axis=1)
    np.save('./distribution.npy',distribution)    
    
    