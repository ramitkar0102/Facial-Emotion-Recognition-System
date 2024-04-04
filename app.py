import cv2 as cv
import numpy as np
import streamlit as st

from streamlit_option_menu import option_menu
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

from utils.utils import get_models, get_image_file, get_video_file, mediapipe_detection, opencv_detection
from utils.utils import mp_face_detection, mp_drawing, emotion_dict, rescale, enhance_image, enhance_image_with_adaptive_histogram_equalization, enhance_image_with_histogram_equalization, enhance_image_with_unsharp_masking, enhance_image_with_bilateral_filter

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

# Add Sidebar and Main Window style
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 330px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 330px
        margin-left: -350px
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] > div:first-child h1{
        padding: 0rem 0rem 0rem 0rem;
        text-align: center;
        font-size: 2rem;
    }
    .css-1544g2n.e1fqkh3o4 {
        padding-top: 4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Basic App Scaffolding
st.title('Facial Emotion Recognition')
st.divider()

with st.sidebar:
    st.title('FaceVibes')
    st.divider()
    # Define available pages in selection box
    app_mode = option_menu("Page", ["About", "Image", "Video"],
                           icons=["person-fill", "images", "film"], menu_icon="list", default_index=0,
                           styles={
                               "icon": {"font-size": "1rem"},
                               "nav-link": {"font-family": "roboto", "font-size": "1rem", "text-align": "left"},
                               "nav-link-selected": {"background-color": "tomato"},
                           }
                           )

# About Page
if app_mode == 'About':
    st.markdown('''
                ## FaceVibes \n
                In this application we are using **OpenCV & MediaPipe** for the Face Detection.
                **Tensorflow** is to create the Facial Emotion Recognition Model.
                **StreamLit** is to create the Web Graphical User Interface (GUI) \n

                - [Github](https://github.com/ramitkar0102) \n
    ''')

# Image Page
elif app_mode == 'Image':

    # Sidebar
    st.sidebar.divider()
    model = get_models()
    mode = st.sidebar.radio('Mode', ('With full image', 'With cropped image'))
    detection_type = st.sidebar.radio('Detection Type', ['Mediapipe', 'OpenCV'])
    st.sidebar.divider()

    detection_confidence = 0.5
    if detection_type == 'Mediapipe':
        detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
        st.sidebar.divider()

    image = get_image_file()

    
    
    # Display Original Image on Sidebar
    st.sidebar.write('Original Image')
    st.sidebar.image(cv.cvtColor(image, cv.COLOR_BGR2RGB), use_column_width=True)


    # Enhance the inputed image
    # image = enhance_image(image=image)
    # image = enhance_image_with_adaptive_histogram_equalization(image)
    st.sidebar.divider()
    st.sidebar.write('Image Enhancement Options')
    enhancement_option = st.sidebar.selectbox('Select Enhancement Technique', ['None', 'Histogram Equalization', 'Adaptive Histogram Equalization', 'Unsharp Masking', 'Bilateral Filter'])
    if enhancement_option == 'Histogram Equalization':
        enhanced_image = enhance_image_with_histogram_equalization(image)
    elif enhancement_option == 'Adaptive Histogram Equalization':
        enhanced_image = enhance_image_with_adaptive_histogram_equalization(image)
    elif enhancement_option == 'Unsharp Masking':
        enhanced_image = enhance_image_with_unsharp_masking(image)
    elif enhancement_option == 'Bilateral Filter':
        enhanced_image = enhance_image_with_bilateral_filter(image)
    else:
        enhanced_image = image

    st.sidebar.image(enhanced_image, use_column_width=True, channels='BGR')

    if detection_type == 'Mediapipe':
        mediapipe_detection(detection_confidence, enhanced_image, model, mode)
    else:
        opencv_detection(enhanced_image, model, mode)
    


# Video Page
elif app_mode == 'Video':

    # Sidebar
    st.set_option('deprecation.showfileUploaderEncoding', False)
    use_webcam = st.sidebar.checkbox('Use Webcam')
    st.sidebar.divider()

    model = get_models()
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.divider()

    # Get Video
    stream = st.image("./assets/multi face.jpg", use_column_width=True)

    video = get_video_file(use_webcam)

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=detection_confidence) as face_detection:
        bg_subtractor = cv.createBackgroundSubtractorMOG2()
        while use_webcam:
            try:
                video = cv.VideoCapture(0)  # 0 for default camera
                if not video.isOpened():
                    raise Exception("Could not open video device.")
            except Exception as e:
                print(f"Error: {e}")
            try:
                ret, frame = video.read()
                image = frame.copy()

                if not ret:
                    print("Ignoring empty camera frame.")
                    video.release()
                    break

                img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                
                # Add background subtraction to isolate the foreground (faces)
                fg_mask = bg_subtractor.apply(img)
                img = cv.bitwise_and(img, img, fg_mask)
                results = face_detection.process(img)

                image_rows, image_cols, _ = frame.shape

                if results.detections:
                    for detection in results.detections:
                        try:
                            box = detection.location_data.relative_bounding_box

                            x = _normalized_to_pixel_coordinates(box.xmin, box.ymin, image_cols, image_rows)
                            y = _normalized_to_pixel_coordinates(box.xmin + box.width, box.ymin + box.height, image_cols,
                                                                image_rows)

                            # Draw face detection box
                            mp_drawing.draw_detection(image, detection)

                            # Crop image to face
                            cimg = frame[x[1]:y[1], x[0]:y[0]]
                            if rescale():
                                cropped_img = np.expand_dims(cv.resize(cimg, (48, 48)), 0)
                            else:
                                cropped_img = np.expand_dims(cv.resize(cimg, (48, 48)), 0) / 255.
                            
                            

                            # get model prediction
                            pred = model.predict(cropped_img)
                            idx = int(np.argmax(pred))

                            image = cv.flip(image, 1)
                            cv.putText(image, emotion_dict[idx], (image_rows - x[0], x[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2, cv.LINE_AA)
                            image = cv.flip(image, 1)

                        except Exception:
                            print("Ignoring empty camera frame.")
                            pass

                stream.image(cv.flip(image, 1), channels="BGR", use_column_width=True)

            except Exception as e:
                pass
        if video is not None:
            video.release()