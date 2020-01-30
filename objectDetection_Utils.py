import numpy as np
import faceDetection_Utils as faceAgeGender
import image_Utils as imgDraw

LABELS_FILE = "./SSDMobilenetV2COCO/mscoco_complete_label_map.pbtxt"
OBJ_CONFIDENCE = 0.5

def objectDetection_readLabels():
    with open(LABELS_FILE, 'r') as f:
        labels = [x.strip() for x in f]
    return labels

def objectDetection_Analysis(image, data, w_Original, h_Original, objectDetection, labels, faceDetection, genderAgeDetection):
    object_info = []
    out_blob = objectDetection[2]
    object_data = data[out_blob]
    object_data = object_data[0][0]
    object_data = object_data[np.where(object_data[:,2] > OBJ_CONFIDENCE)] #CONFIDENCE

    if object_data.size == 0:
        return
    
    for obj in object_data:
        class_id = int(obj[1])
        confidence = round(obj[2] * 100, 1)
        x_min,  y_min, x_max,  y_max  = int(obj[3] * w_Original) , int(obj[4] * h_Original), int(obj[5] * w_Original), int(obj[6] * h_Original)
        objDataBox = [x_min, y_min, x_max, y_max]
        label = labels[class_id] if labels else str(class_id)
        #Draw Object Detection
        imgDraw.image_DrawBox(image, objDataBox, True)  
        imgDraw.image_Write(image, label, objDataBox, True)  

        if class_id == 1:
            ageGenderPrediction = faceAgeGender.faceDetection_GenderAgeDetection_Complete(image, faceDetection, genderAgeDetection)


    return object_info
