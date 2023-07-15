import numpy as np
import pandas as pd
from deepface import DeepFace
import streamlit as st
import cv2
import base64
import time

st.set_page_config(layout="wide")

def get_video_base64(video_path):
    with open(video_path, "rb") as file:
        video_bytes = file.read()
        base64_encoded = base64.b64encode(video_bytes).decode("utf-8")
        return base64_encoded

video_path = "deep.mp4"
video_base64 = get_video_base64(video_path)

video_html = f"""
	<style>
	#myVideo {{
		position: fixed;
		right: 0;
		bottom: 0;
		min-width: 100%; 
		min-height: 100%;
	}}
	.content {{
		position: fixed;
		bottom: 0;
		background: rgba(0, 0, 0, 0.5);
		color: #f1f1f1;
		width: 100%;
		padding: 20px;
	}}

	</style>

	<video autoplay loop muted id="myVideo">
		<source type="video/mp4" src="data:video/mp4;base64,{video_base64}">
	</video>
"""

st.markdown(video_html, unsafe_allow_html=True)



cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

import tempfile
import os

weights_paths = {
    'age': '/home/appuser/.deepface/weights/age_model_weights.h5',
    'gender': '/home/appuser/.deepface/weights/gender_model_weights.h5',
    'race': '/home/appuser/.deepface/weights/race_model_single_batch.h5',
    'emotion': '/home/appuser/.deepface/weights/facial_expression_model_weights.h5'
}

def upload():
    image=None
    initial_image = st.camera_input('Take a picture')
    original_image = initial_image
    temp_path = None
    if initial_image is not None:
        bytes_data = initial_image.getvalue()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    return image, original_image


    

def main(options):
    
   
    
    
   
    if st.checkbox('Take a picture for prediction'):
        
        image, original_image= upload()
        if original_image is not None and original_image is not None and st.button('Prediction'):  # Check if original_image is not None
            st.warning('Wait for few seconds!!')
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            result = DeepFace.analyze(image,detector_backend=options,actions=['age','gender','emotion'])
            
            for i in range(100):
                progress_bar.progress((i + 1) / 100)
                status_text.text(f"Processing {i+1}%")
                time.sleep(0.01)
            
            progress_bar.empty()
            gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray_frame, 1.1, 3)
            faces = sorted(faces, key=lambda f: -f[2] * f[3])

            if len(faces) > 0:
                x,y,w,h=faces[0]
                
                cv2.rectangle(image, (x, y), (x+w, y+h), (4, 29, 255), 2, cv2.LINE_4)
                user_selected_items = list(result[0].keys())
                if 'age' in user_selected_items:
                    age_label='Age: '+str(result[0]['age'])
                    cv2.putText(image, age_label, (x+w+10, y+45), cv2.FONT_ITALIC,1 ,(255,255,0), 2)
                if 'dominant_gender' in user_selected_items:
                    gender_label='Gender: '+str(result[0]['dominant_gender'])
                    cv2.putText(image, gender_label, (x+w+10, y+75), cv2.FONT_ITALIC,1, (0,255,255), 2)
                
                if 'dominant_emotion' in user_selected_items:
                    emotion_label='Emotion: '+str(result[0]['dominant_emotion']).title()
                    cv2.putText(image, emotion_label, (x+w+10, y+105), cv2.FONT_ITALIC,1 ,(255,0,255), 2)

            st.image(image, channels='BGR')
           

   
if __name__ == '__main__':
    def get_options():
        actions = ['opencv','mtcnn','retinaface']
        option2 = st.selectbox('Choose the following actions:', actions)
        return option2
   
    main(get_options())
