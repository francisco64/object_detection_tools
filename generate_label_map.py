#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:38:16 2020

@author: francisco
"""
import numpy as np
def generate(categories,output):
    end = '\n'
    s = ' '
    class_map = {}
    for ID, name in enumerate(categories):
        out = ''
        out += 'item' + s + '{' + end
        out += s*2 + 'id:' + ' ' + (str(ID+1)) + end
        out += s*2 + 'name:' + ' ' + '\'' + name + '\'' + end
        out += '}' + end*2
        
    
        with open(output, 'a') as f:
            f.write(out)
            
        class_map[name] = ID+1

if __name__ == '__main__':
    categories=np.load('./distribution.npy')[:,0]
    generate(categories,'./label_map.pbtxt')