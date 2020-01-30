#!/usr/bin/python3


#===========================================================================================================
#                                LIBRARIES TO EXPORT         
#===========================================================================================================


#NeuralComputeStick
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path: sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import logging as log
import faceDetection_Utils
import intelNeuralStick_Utils
import objectDetection_Utils
import image_Utils
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore

#Common
import cv2


def process_event_queue (capture, objectDetection, face_Detection, genderAgeDetection):
    labels_map = objectDetection_Utils.objectDetection_readLabels()
  

    while (True):
        start = time.time()
        ret, frame = capture.read()

        initial_w, initial_h = frame.shape[1], frame.shape[0]
        res = intelNeuralStick_Utils.intelNeuralStick_preprocess_object_image(frame, objectDetection)
        
        objectDetection_Utils.objectDetection_Analysis(frame, res, initial_w, initial_h, objectDetection, labels_map, face_Detection, genderAgeDetection)

        
        end = time.time() - start
        FPS = round(1.0 / end)
      
        cv2.putText(frame, "FPS = {}".format(FPS), (5,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (204,255,255), 1)

        frameG = cv2.resize(frame, (1200, 800))            
        cv2.imshow("Detection Results", frameG)
       

        key = cv2.waitKey(1)
        if key == 27:
            break



def main ():
    
    objectDetection, faceDetection, genderAgeDetection = intelNeuralStick_Utils.intelNeuralStick_init()
    capture = cv2.VideoCapture(0)
   
    log.info("Starting Object Detection...")
    process_event_queue (capture, objectDetection, faceDetection, genderAgeDetection)
    
    cv2.destroyAllWindows
            


if (__name__ == "__main__"):
    sys.exit(main() or 0)
