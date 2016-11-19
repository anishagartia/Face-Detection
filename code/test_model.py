
# coding: utf-8

# In[157]:

#get_ipython().magic('matplotlib inline')


# In[158]:

import numpy as np
import glob
import os
from skimage import io
import argparse
import sys
import csv


# In[159]:

from keras.models import load_model
from skimage.transform import resize


# In[160]:

model = load_model('nn_model.h5')
model.summary();


# In[161]:

# parser = argparse.ArgumentParser()
# parser.add_argument('img_path', type=str)
# parser.add_argument('bbox',type=int,nargs=4)
# args = parser.parse_args()


# In[162]:

#Read csv file
bboxes_path = 'bboxes_for_nnet.csv'
img_paths = []
bboxes = []
home_path = '../data/test_scenes/test_jpg'
with open(bboxes_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        img_paths.append(os.path.join(home_path,row[0]))
        bboxes.append([int(row[1]),int(row[2]),int(row[3]),int(row[4])])
bboxes = np.asarray(bboxes)        


# In[166]:

X_Test = []
out_class = []
fout = open('nn_lables.txt','w')
for i in range(0,len(img_paths)):
    this_img = io.imread(img_paths[i],as_grey=True)
    orig_patch = this_img[bboxes[i][1]-1:bboxes[i][3]-1,bboxes[i][0]-1:bboxes[i][2]-1]
    X_Test = resize(orig_patch,(36,36))
    X_Test = np.array(X_Test)
    X_Test = X_Test.astype('float32')
    X_Test = X_Test.reshape(1,-1)
    this_class = model.predict_classes(X_Test,batch_size=32, verbose=0 )
    out_class.append(int(this_class[0][0]))   
    fout.write(str(int(this_class))+'\n')
    #print(this_class)


# In[167]:

fout.close()


# In[ ]:




# In[ ]:



