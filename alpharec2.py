import cv2 
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time


X=np.load('image.npz')['arr_0']
Y=pd.read_csv('https://raw.githubusercontent.com/whitehatjr/datasets/master/C%20122-123/labels.csv')['labels']
print(pd.Series(Y).value_counts())
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses=len(classes)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=2500,train_size=7500,random_state=9)
X_train_scaled=X_train/255.0
X_test_scaled=X_test/255.0
clf=LogisticRegression(solver='saga',multi_class='multinomial').fit(X_train_scaled,Y_train)
# y_pred = clf.predict(X_test_scaled) 
# accuracy = accuracy_score(Y_test, y_pred) 
# print(accuracy)
cap = cv2.VideoCapture(0)
while(True):
    try:
        
        ret, frame = cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width=gray.shape
        upper_left=(int(width/2 -56),int(height/2 -56))
        bottom_right=(int(width/2 +56),int(height/2 +56))
        cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)
        #roi = Region Of Interest : y2:y1, x2:x1
        roi=gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
        im_pil=Image.fromarray(roi)
        # 0 to 255
        image_bw=im_pil.convert('L')
        image_bw_resized=image_bw.resize((28,28),Image.ANTIALIAS)
        image_bw_resized_inverted=PIL.ImageOps.invert(image_bw_resized)
        pixel_filter=20
        min_pixel=np.percentile(image_bw_resized_inverted,pixel_filter)
        image_bw_resized_inverted_scaled=np.clip(image_bw_resized_inverted - min_pixel,0,255)
        max_pixel=np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        test_sample=np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        test_predict=clf.predict(test_sample)
        print("Predicted alphabet: ", test_predict)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        pass
        
cap.release()
cv2.destroyAllWindows()

