import logging as log
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path: sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import cv2
from openvino.inference_engine import IENetwork, IECore

DEVICE_NAME = "MYRIAD"
MODEL_OBJECT_DETECTION = "./SSDMobilenetV2COCO/frozen_inference_graph.xml"
MODEL_FACE_DETECTION = "./FaceDetection/face-detection-retail-0004.xml"
MODEL_AGE_GENDER_PREDICTION = "./modelsGenderAge/age-gender-recognition-retail-0013.xml"


def intelNeuralStick_modelRead(PATH_MODEL_XML, DEVICE_NAME, ie):
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    model_xml = PATH_MODEL_XML
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    log.info("Loading Inference Engine...")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)
    #Read model
    #input
    for blob_name in net.inputs: 
        if len(net.inputs[blob_name].shape) == 4: #net size [batch, depth, height, width]
            input_blob = blob_name
        elif len(net.inputs[blob_name].shape) == 2:
            img_info_input_blob = blob_name
        else:
            raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                               .format(len(net.inputs[blob_name].shape), blob_name))
    #output
    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_net = ie.load_network(network=net, device_name=DEVICE_NAME)
    n, c, h, w = net.inputs[input_blob].shape  #[batch, depth, height, width]
    info_NeuralComputeStick = [net, input_blob, out_blob, exec_net, n, c, h, w]
    return info_NeuralComputeStick


def intelNeuralStick_init():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    ie = IECore()
    log.info("Device info:")
    versions = ie.get_versions(DEVICE_NAME)
    print("{}{}".format(" "*10, DEVICE_NAME))
    print("{}Plugin version ......... {}.{}".format(" "*10, versions[DEVICE_NAME].major, versions[DEVICE_NAME].minor))
    print("{}Build ........... {}".format(" "*10, versions[DEVICE_NAME].build_number))
    
    # Read IR
    objectDetection = intelNeuralStick_modelRead(MODEL_OBJECT_DETECTION, DEVICE_NAME, ie)
    faceDetection  = intelNeuralStick_modelRead(MODEL_FACE_DETECTION, DEVICE_NAME, ie)
    genderAgeDetection = intelNeuralStick_modelRead(MODEL_AGE_GENDER_PREDICTION, DEVICE_NAME, ie)

    return objectDetection, faceDetection, genderAgeDetection


def intelNeuralStick_preprocess_object_image(img, info_NeuralComputeStick):
   
    n, c, h, w = info_NeuralComputeStick[4], info_NeuralComputeStick[5], info_NeuralComputeStick[6], info_NeuralComputeStick[7]
    exec_net = info_NeuralComputeStick[3]
    input_blob = info_NeuralComputeStick[1]
    ih, iw = img.shape[:-1]
    if (ih, iw) != (h, w):
        img = cv2.resize(img, (w, h))
    img = img.transpose((2, 0, 1))   # Change data layout from HWC to CHW   [height, width, channel] to [channel, height, width, ]
    img = img.reshape((n, c, h, w))  # [batch, channel, height, width]
    res = exec_net.infer(inputs = {input_blob: img})
    
    return res   
