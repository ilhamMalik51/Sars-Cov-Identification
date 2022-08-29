import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import LabelBinarizer

def load(lokasi):
    dirlist = listdir(lokasi)
    data = []
    label = []
    for(i,classdir) in enumerate(dirlist):
        listfile = listdir(join(lokasi,classdir))
        for(j,filename) in enumerate(listfile):                
           file_loc = join(lokasi,classdir,filename)
           print(file_loc)
           img = cv2.imread(file_loc,0)
           img = cv2.resize(img,(192,192))           
           data.append(img)
           label.append(classdir)
           
    return (np.array(data),np.array(label))
		

