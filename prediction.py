#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:23:17 2021

@author: francisco
"""
from object_detection.utils import visualization_utils as viz_utils
import tensorflow as tf
import time
from object_detection.utils import label_map_util
import glob
import cv2
import numpy as np
import pandas as pd
import json 

def compute_iou(box1, box2):
        g_xmin, g_ymin, g_xmax, g_ymax = box1[1],box1[0],box1[3],box1[2]
        d_xmin, d_ymin, d_xmax, d_ymax = box2[1],box2[0],box2[3],box2[2]
        
        xa = max(g_xmin, d_xmin)
        ya = max(g_ymin, d_ymin)
        xb = min(g_xmax, d_xmax)
        yb = min(g_ymax, d_ymax)
        intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)
        boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
        boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)
        
        return intersection / float(boxAArea + boxBArea - intersection)
def load_model(PATH_TO_MODEL_DIR):
    PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
    print('Loading model...', end='')
    
    start_time = time.time()

    
    if  tf.__version__[0]=='2':
        detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    else:
        detect_fn = tf.contrib.predictor.from_saved_model(PATH_TO_SAVED_MODEL)
    
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    return detect_fn

def predict_image(IMAGE_PATH,PATH_TO_LABELS,detect_fn,Tscore,timeP=None):
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)      
    image = cv2.imread(IMAGE_PATH) 
    
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    start=time.time()
    if  tf.__version__[0]=='2':
        detections = detect_fn(input_tensor)
    else:
        detections = detect_fn({"inputs": input_tensor.eval(session=tf.compat.v1.Session())})
    
    end=time.time()
    if timeP is not None:
        timeP.append(end-start)
    else:
        timeP=end-start
    
    num_detections = int(detections.pop('num_detections'))
    if  tf.__version__[0]=='2':
        detections = {key: value[0, :num_detections].numpy()
                       for key, value in detections.items()}
    else:
        detections = {key: value[0, :num_detections]
                       for key, value in detections.items()}
    
    detections['num_detections'] = num_detections
    
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    m,n,_=image.shape
    #detections['detection_boxes'][1]#ymin, xmin, ymax, xmax 
    #Tscore=0.01
    boxes=np.uint64(np.array([[d[0]*m,d[1]*n, d[2]*m, d[3]*n] for d in detections['detection_boxes']]))[detections['detection_scores']>Tscore]
    labels=np.array([category_index[i]['name'] for i in detections['detection_classes']])[detections['detection_scores']>Tscore]
    scores=detections['detection_scores'][detections['detection_scores']>Tscore]

    p_score=[]
    p_label=[]
    p_bbox=[]
    p_bboxNorm=[]
    p_labelInt=[]
   
    noMirar=[]
    
    for i in range(len(labels)):
        if i not in noMirar:
            noMirar.append(i)
            maximoBox=boxes[i]
            maximoScore=scores[i]
            maximoLabel=labels[i]
            maximoLabelInt=detections['detection_classes'][i]
            maximoBoxNorm=detections['detection_boxes'][i]
            for j in range(len(labels)):
                intersectado=compute_iou(boxes[i],boxes[j])>0.3
                if intersectado and (j not in noMirar):
                    #print('intersectado score:'+str(scores[j]))
                    if scores[j]>maximoScore:
                        maximoBox=boxes[j]
                        maximoScore=scores[j]
                        maximoLabel=labels[j]
                        maximoLabelInt=detections['detection_classes'][j]
                        maximoBoxNorm=detections['detection_boxes'][j]
                        
                    noMirar.append(j)        
            p_score.append(maximoScore)
            p_label.append(maximoLabel)
            p_bbox.append(maximoBox)
            p_bboxNorm.append(maximoBoxNorm)
            p_labelInt.append(maximoLabelInt)
            
    p_score=np.array(p_score)
    p_label=np.array(p_label)
    p_bbox=np.array(p_bbox)       
    p_bboxNorm=np.array(p_bboxNorm)
    p_labelInt=np.array(p_labelInt)
    

    label_unique,count_label=np.unique(p_label,return_counts=True)
    #json_pred=''
    resumen_predicciones_dict={}
    dict_pred_img={}
    imagen_pred={}
    
    for i in range(0,len(label_unique)):
        
        dict_pred={}
        dict_pred['id']=int(p_labelInt[p_label==label_unique[i]][0])
        dict_pred['nombre']=str(label_unique[i])
        dict_pred['cantidad']=int(count_label[i])
        bbx_detected=p_bbox[p_label==label_unique[i]]
        list_bbox=[]
        for bbox in bbx_detected:
            dict_box={}
            dict_box['left']=int(bbox[1])
            dict_box['top']=int(bbox[0])
            dict_box['width']=int(bbox[3]-bbox[1])
            dict_box['height']=int(bbox[2]-bbox[0])
            #dict_box['accuracy']=int(100*p_score[(p_bbox==bbox).any(axis=1)][0])
            list_bbox.append(dict_box)
        dict_pred['ubicaciones']=list_bbox
        resumen_predicciones_dict[str(i+1)]=dict_pred    
        
    imagen_pred['resumen_predicciones']=resumen_predicciones_dict    
    
        #
        # json_pred += json.dumps(dict_pred)
        # if i<len(label_unique)-1:
        #     json_pred+='|'
        
    return p_score,p_label,p_bbox,p_bboxNorm,p_labelInt,timeP,imagen_pred
        
    
def predict(PATH_TO_MODEL_DIR,IMAGE_PATHS,PATH_TO_LABELS,PATH_PREDICTIONS,PATH_IMAGE_PREDICTIONS=None,Tscore=0.011):
    
    detect_fn=load_model(PATH_TO_MODEL_DIR)#cargar modelo
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)  
    timeP=[]
      
    c=0    
    cant_imagenes=len(glob.glob(IMAGE_PATHS + '/*.jpg'))
    timeP=[]
    dict_pred_img={}
    for IMAGE_PATH in glob.glob(IMAGE_PATHS + '/*.jpg'):
        
        p_score,p_label,p_bbox,p_bboxNorm,p_labelInt,timeP,imagen_pred=predict_image(IMAGE_PATH,PATH_TO_LABELS,detect_fn,Tscore,timeP=timeP)
        dict_pred_img[IMAGE_PATH.split('/')[-1]]=imagen_pred
        
        
        
        if PATH_IMAGE_PREDICTIONS is not None:
            image_with_detections = cv2.imread(IMAGE_PATH) 
            
            #SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
            viz_utils.visualize_boxes_and_labels_on_image_array(
                  image_with_detections,
                  p_bbox,
                  p_labelInt,
                  p_score,
                  category_index,
                  use_normalized_coordinates=False,
                  max_boxes_to_draw=200,
                  min_score_thresh=0,
                  agnostic_mode=False)
            
            
            #print('Done prediccion para: '+IMAGE_PATHS.split('/')[-1])
            # DISPLAYS OUTPUT IMAGE
            cv2.imwrite(PATH_IMAGE_PREDICTIONS+'/'+IMAGE_PATH.split('/')[-1], image_with_detections)    
        
        #break
        files=np.array([IMAGE_PATH.split('/')[-1]]*len(p_bbox))   
        if c==0:
            predictions=np.concatenate((files.reshape(len(files),1),p_label.reshape(len(p_label),1),p_labelInt.reshape(len(p_labelInt),1),p_bbox),axis=1)
        else:
            try:
                predictions=np.concatenate((predictions,(np.concatenate((files.reshape(len(files),1),p_label.reshape(len(p_label),1),p_labelInt.reshape(len(p_labelInt),1),p_bbox),axis=1))),axis=0)            
            except:
                continue
        c+=1
        print('Progress: '+str(round(100*(c/cant_imagenes)))+' %'+' Prediction Time: '+str(round(np.mean(timeP),3))+' ± '+str(round(np.std(timeP),3))+' seconds')
        
        predictions_df=pd.DataFrame(data=predictions,columns=['filename','Class','label_int','ymin','xmin','ymax','xmax'])
        predictions_df.to_csv(PATH_PREDICTIONS+'/predictions.csv')   
    print('predictions have been saved to '+PATH_PREDICTIONS+'/predictions.csv')    
    print('Prediction Time: '+str(round(np.mean(timeP),3))+' ± '+str(round(np.std(timeP),3))+' seconds')
    
    with open("jsonindicadores.json", "w") as outfile:  
        json.dump(dict_pred_img, outfile,indent=3) 
    print('JSON file has been created and saved to: jsonindicadores.json')
    
    return predictions_df,dict_pred_img


# IMAGE_PATHS ='/home/francisco/Documents/datasetTest'

# PATH_TO_MODEL_DIR = '/home/francisco/Desktop/pretrained/efficientDetD4Testckpt42'

# PATH_TO_LABELS = '/home/francisco/Desktop/pretrained/label_map_test.pbtxt'

# PATH_PREDICTIONS='/home/francisco/Desktop/ObjectDetectionPeru'

# IMAGE_PATH='/home/francisco/Documents/datasetTest/1212726-14271784.jpg'

# PATH_IMAGE_PREDICTIONS='/home/francisco/Desktop/predicciones'


# detect_fn=load_model(PATH_TO_MODEL_DIR)

# p_score,p_label,p_bbox,p_bboxNorm,p_labelInt,timeP,json_pred=predict_image(IMAGE_PATH,PATH_TO_LABELS,detect_fn,Tscore=0.011,timeP=None)










# pred=predict(PATH_TO_MODEL_DIR,IMAGE_PATHS,PATH_TO_LABELS,PATH_PREDICTIONS,PATH_IMAGE_PREDICTIONS)  

# filenames=pred['filename'].unique()
