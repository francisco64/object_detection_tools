#!/usr/bin/env python
# coding: utf-8
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #quitar comentario para CPU
import tensorflow as tf
import cv2
import numpy as np
import glob
import pandas as pd
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from XML2PD import xml2pd
from prediction import compute_iou, predict
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

#IMAGE_PATHS = '/home/francisco/Desktop/datasets/validationCleanData20/85743-14259014.jpg'
#IMAGE_PATHS = '/home/francisco/Documents/testCleanData50'

IMAGE_PATHS ='/home/francisco/Desktop/ObjectDetectionPeru/datasets/dataset_corregido'

#PATH_TO_MODEL_DIR = '/home/francisco/Desktop/pretrained/fasterRCNN-DA'
#PATH_TO_MODEL_DIR = '/home/francisco/Desktop/pretrained/frRCNN'
#PATH_TO_MODEL_DIR = '/home/francisco/Desktop/pretrained/efficientDetD4Todo'
PATH_TO_MODEL_DIR = '/home/francisco/Desktop/ObjectDetectionPeru/modelo-nasnet'

#PATH_TO_MODEL_DIR ='/home/francisco/Desktop/pretrained/modelPruebaTF1/inference_graph_red_1_da.pb'

PATH_TO_LABELS = './red1_label_map_da (2).pbtxt'

PATH_PREDICTIONS='/home/francisco/Desktop/ObjectDetectionPeru'


PATH_IMAGE_PREDICTIONS='/home/francisco/Desktop/ObjectDetectionPeru'



metricas=[]

Tscore=0.5
if not os.path.exists(PATH_PREDICTIONS+'/predictions.csv'):  
    pred,_=predict(PATH_TO_MODEL_DIR,IMAGE_PATHS,PATH_TO_LABELS,PATH_PREDICTIONS,PATH_IMAGE_PREDICTIONS,Tscore)    
    
else:
    pred=pd.read_csv(PATH_PREDICTIONS+'/predictions.csv')

test=xml2pd(image_path=IMAGE_PATHS,csv=True) 
    
#--------------------------------------------------------------------------------------------------------------



label_map = PATH_TO_LABELS
output_path = "./confusion_matrix.csv"    
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)    
labels=[i['name'] for i in list(category_index.values())]
ids=[i['id'] for i in list(category_index.values())]

categories=dict(zip(labels,ids))



# path_distribution='/home/francisco/Desktop/ObjectDetectionPeru/distribution.npy'
# distribution=np.load(path_distribution,allow_pickle=True)

# categories=[x for _,x in sorted(zip(distribution[:,1],distribution[:,0]))][::-1]

# categories=dict(zip(categories,range(1,len(categories)+1)))


confusion_matrix = np.zeros(shape=(len(categories) + 1, len(categories) + 1))

file_unique = test['filename'].unique()
FN_detection=0
FP_detection=0
TP_detection=0

FN_classification=0

for file in file_unique:
        #file='134357156.jpg'categories=[i['name'] for i in list(category_index.values())]

        test_df = test[test['filename']==file]
        test_df.reset_index(inplace = True, drop = True) 
        pred_df = pred[pred['filename']==file]
        pred_df.reset_index(inplace = True, drop = True) 
        
        
            
        pred_class = pred_df
       
        
        groundtruth_boxes = test_df[['xmin','ymin','xmax','ymax']].values.tolist()
        detection_boxes = [[int(xmin),int(ymin),int(xmax),int(ymax)] for xmin,ymin,xmax,ymax in pred_class[['xmin','ymin','xmax','ymax']].values.tolist()]
        
        matches = []   
        
        for i in range(len(groundtruth_boxes)):
            for j in range(len(detection_boxes)):
                iou = compute_iou(groundtruth_boxes[i], detection_boxes[j])
        
                if iou > 0.5:
                    matches.append([i, j, iou])
        
        
        matches = np.array(matches)

        
        
        if matches.shape[0] > 0:
            # Sort list of matches by descending IOU so we can remove duplicate detections
            # while keeping the highest IOU entry.
            matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
            
            # Remove duplicate detections from the list.
            matches = matches[np.unique(matches[:,1], return_index=True)[1]]
            
            # Sort the list again by descending IOU. Removing duplicates doesn't preserve
            # our previous sort.
            matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
            
            # Remove duplicate ground truths from the list.
            matches = matches[np.unique(matches[:,0], return_index=True)[1]]
        
        
        for i in range(len(groundtruth_boxes)):
            #print(i)
            try:
                if matches.shape[0] > 0 and matches[matches[:,0] == i].shape[0] == 1:
                    #print("inside : ",i)
                    
                    confusion_matrix[categories[test_df['Class'][i]] - 1][categories[pred_class['Class'][matches[matches[:,0] == i].tolist()[0][1]]] - 1] += 1
                    TP_detection+=1
                    if (categories[test_df['Class'][i]] - 1) != (categories[pred_class['Class'][matches[matches[:,0] == i].tolist()[0][1]]] - 1):
                        FN_classification+=1
                        
                        
                else:
                    #print('Objeto no detectado')
                    confusion_matrix[categories[test_df['Class'][i]] - 1][confusion_matrix.shape[1] - 1] += 1
                    FN_detection+=1
            except:
                continue

        for i in range(len(detection_boxes)):
            try:
               if matches.shape[0] > 0 and matches[matches[:,1] == i].shape[0] == 0:
                    
                    confusion_matrix[confusion_matrix.shape[0] - 1][categories[pred_class['Class'][i]] - 1] += 1
                    FP_detection+=1
            except:
                continue
        #break    
single_labels=list(categories.keys())
columns=single_labels.copy()
rows=single_labels.copy()
columns.append('Falso Negativo')
rows.append('Falso Positivo')
confusion_matrix_df=pd.DataFrame(data=confusion_matrix,columns=columns,index=rows)
confusion_matrix_df.to_excel('./confusion_matrix.xlsx')
TP=np.trace(confusion_matrix)

Negocio=TP/(TP+FN_detection+FN_classification+0.000000001)
PRECISION=TP/(TP+FP_detection+0.000000001)



precision_detection=TP_detection/(TP_detection+FP_detection+0.000001)
recall_detection=TP_detection/(TP_detection+FN_detection+0.000001)
f1score_detection=2*(precision_detection*recall_detection)/(precision_detection+recall_detection+0.000001)

precision_clases=[]
recall_clases=[]
f1score_clases=[]


clases_test=[]
for clase in categories:
    index_clase=categories[clase]-1
    if np.sum(confusion_matrix[index_clase][:])>0:
        precision=confusion_matrix[index_clase][index_clase]/(np.sum(confusion_matrix[index_clase][0:len(categories)])+0.000001)
        recall=confusion_matrix[index_clase][index_clase]/np.sum(confusion_matrix[0:len(categories)][index_clase]+0.000001)
        f1score=2*precision*recall/(precision+recall+0.000001)
        
        precisionTodo=confusion_matrix[index_clase][index_clase]/(np.sum(confusion_matrix[index_clase][:])+0.000001)
        recallTodo=confusion_matrix[index_clase][index_clase]/(np.sum(confusion_matrix[:][index_clase])+0.000001)
        
        precision_clases.append(precision)
        recall_clases.append(recall)
        f1score_clases.append(f1score)

        clases_test.append(clase)
    
mAP=Negocio
mAR=0#RECALL
mF1score=0#2*mAP*mAR/(mAP+mAR+0.000001)

# aparicionesTrain=[]
# for label in np.array(clases_test):
#     aparicionesTrain.append(distribution[distribution[:,0]==label][0][1])
    
# aparicionesTrain=np.array(aparicionesTrain)

# aparicionesTrain=np.concatenate((np.array(['']),aparicionesTrain,np.array([np.sum(aparicionesTrain)])))

aparicionesTest=np.sum(confusion_matrix,axis=1)[np.sum(confusion_matrix,axis=1)>0]
aparicionesTest[-1]=np.round(np.sum(aparicionesTest[0:len(aparicionesTest)-1]))
aparicionesTest=np.concatenate((np.array(['']),aparicionesTest))

clases=np.concatenate((np.array(['Detection']),np.array(clases_test),np.array(['Business Metric']))) 
precision=np.concatenate((np.array([round(100*precision_detection,2)]),np.round(100*np.array(precision_clases),2),np.round(100*np.array([mAP]))))
recall=np.concatenate((np.array([round(100*recall_detection,2)]),np.round(100*np.array(recall_clases),2),np.round(100*np.array([mAR]))))
f1score=np.concatenate((np.array([round(100*f1score_detection,2)]),np.round(100*np.array(f1score_clases),2),np.round(100*np.array([mF1score]))))      

data=np.concatenate((aparicionesTest.reshape(len(clases),1),clases.reshape(len(clases),1),precision.reshape(len(clases),1),recall.reshape(len(clases),1),f1score.reshape(len(clases),1)),axis=1)
#(aparicionesTrain.reshape(len(clases),1)
report=pd.DataFrame(data=data,columns=['Apariciones test','Label','Precision','Recall','F1score'])
report.to_excel('./report.xlsx')

#---------------------------------------------------------------------------reporte2

binario_aparicion=np.sum(confusion_matrix,axis=1)>0
single_labels=np.array(single_labels)
label_apariciones=single_labels[binario_aparicion[:-1]]
encontrados=confusion_matrix.diagonal()[binario_aparicion][:-1]
apariciones=np.sum(confusion_matrix,axis=1)[np.sum(confusion_matrix,axis=1)>0][:-1]

data=np.concatenate((label_apariciones.reshape(-1,1),apariciones.reshape(-1,1),encontrados.reshape(-1,1)),axis=1)

