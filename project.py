import streamlit as st 
import cv2 
import tensorflow as tf 
import numpy as np 
import datetime 

st.title("Smoking/Not Smoking Alert Generator through video footage") 
st.text("upload your video footage here.......") 

date=datetime.datetime.now() 
alertmessage="Hii TMT Enterprices""\n""\n""This message from SECTION 307 control room,SECTION 307 is a smoking restricted area.""\n""some of your employees were smoked at near of the SECTION 307 area at %s .""\n""The place is so dangerous with the explosive items so please let them know about the seriousnes about the issue.""\n""Next time we will never forgive any type of this activities towards that area! ""\n""\n""Thank you"%date
supportmessage="By knowing about the seriousness about the issue,""\n""No one from the video were smoking in that area""\n""Thank You....." 

classes=[supportmessage,alertmessage] 

fileupload=st.file_uploader("upload video",type=[".mp4"]) 

if fileupload is not None : 
    
    def alert() : 
    
        video=cv2.VideoCapture("f{fileupload}") 

        for i in range (1) : 
    
            sucess,frame=video.read() 
            # rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) 
            resize=cv2.resize(frame,(150,150)) 
            reshape=resize.reshape(1,150,150,3) 
            model=tf.keras.models.load_model("smoking model.h5") 
    
            prediction=model.predict(reshape) 
    
            position=np.argmax(prediction) 
        
            return classes[position] 
    
    alertingstate=alert() 
    
    st.spinner("checking video......")
    st.success(f"Alert message : {alertingstate}") 