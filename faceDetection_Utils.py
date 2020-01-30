import numpy as np
import intelNeuralStick_Utils
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path: sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import cv2
import image_Utils as imgDraw

FACE_CONFIDENCE = 0.75
GENDER_CONFIDENCE = 0.75
PADDING = 20

def faceDetection_genderAge_Analysis(data_face, confidence):
    if not data_face:
        return

    gender_Age = data_face
    age = int(gender_Age['age_conv3'][0][0][0][0] * 100)
    Gender_prediction = gender_Age['prob'][0]
    Female = Gender_prediction[0][0][0]
    Male = Gender_prediction[1][0][0]
    gender = " "
    confidence_info = 0
    if Male > Female and Male > confidence:
        gender, confidence_info = "MALE", Male
    elif Female > Male and Female > confidence:
        gender, confidence_info = "FEMALE", Female
    else:
        return
    #confidence face, xmin, ymin, xmax, ymax, confidence gender, gender, Age
    genderAge_prediction = [confidence_info, gender, age]
    return  genderAge_prediction



def faceDetection_GenderAgeDetection_Complete(image, faceDetection, genderAgeDetection):
    w, h = image.shape[0], image.shape[1]
    faceDetected = intelNeuralStick_Utils.intelNeuralStick_preprocess_object_image(image, faceDetection)

    if faceDetected is not None:
        out_blob = faceDetection[2]
        data_face = faceDetected[out_blob]
        data_face = data_face[0][0]
        data_face = data_face[np.where(data_face[:,2] > FACE_CONFIDENCE)]

        if len(data_face) > 0:
            for i in data_face:
                confidence_info = i[2]
                face_xmin, face_ymin, face_xmax, face_ymax = int(i[3] * h), int(i[4] * w), int(i[5] * h), int(i[6] * w)
                face_dataBox = [face_xmin, face_ymin, face_xmax, face_ymax]

                face_img = image[max(0, face_ymin - PADDING) : min(face_ymax + PADDING, w - 1), max(0, face_xmin - PADDING): min(face_xmax + PADDING , h-1)]
                gender_Age = intelNeuralStick_Utils.intelNeuralStick_preprocess_object_image(face_img, genderAgeDetection)

                if gender_Age is not None:
                    ageGender_info = faceDetection_genderAge_Analysis(gender_Age, GENDER_CONFIDENCE)

                    if ageGender_info is not None:
                        text_complete = " {}, {} years".format(ageGender_info[1], ageGender_info[2])
                        imgDraw.image_Write(image, text_complete, face_dataBox, False)
                        imgDraw.image_DrawBox(image, face_dataBox, False)
    
   
 