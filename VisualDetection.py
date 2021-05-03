
import cv2
from fastai.vision.all import load_learner
import numpy as np
import time
import moviepy.editor as mp
import math
import os
import re

import multiprocessing

# @ToBeRefactor------------------------some constant
# Result of the image classifier model
high = []
high_t = []
low = []
low_t = []

# Result of the yolo model
book = []
book_t = []
laptop = []
laptop_t = []
cell = []
cell_t = []
person = []
person_t = []
person_count = []
person_count_t = []
objects = []
objects_t = []

learn_inf = load_learner('/home/ese440/PycharmProjects/ESE440/models/risk_v3.pkl', cpu=True)
# video_path = "/home/ese440/PycharmProjects/ESE440/resources/test-video-voice.mp4"


# font
font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 2
# Line thickness of 2 px
thickness = 1

risk_color = {"low_risk": (0, 255, 0), "high_risk": (0, 0, 255)}
risk_counter = {"low_risk": 0, "high_risk": 0}

sample_rate = 1

# dnn face detection configuration -----------------------------------------------------
modelFile = "/home/ese440/PycharmProjects/ESE440/models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "/home/ese440/PycharmProjects/ESE440/models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# yolov3 configuration---------------------------------------------------------------------
modelConfiguration = '/home/ese440/PycharmProjects/ESE440/yolov3-config/yolov3-tiny.cfg'
modelWeights = '/home/ese440/PycharmProjects/ESE440/yolov3-config/yolov3-tiny.weights'

darkNet = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
darkNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
darkNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

whT = 320
confidenceThreshHold = 0.5
nmsThreshold = 0.3
classesFile = '/home/ese440/PycharmProjects/ESE440/yolov3-config/coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


# @ToBeRefactor------------------------some constant



def findObjects(outputs, img, frame_time):
    hT, wT, cT = img.shape
    boundingBox = []
    classIds = []
    confs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confidenceThreshHold:
                w = int(detection[2] * wT)
                h = int(detection[3] * hT)
                x = int((detection[0] * wT) - (w / 2))
                y = int((detection[1] * hT) - (h / 2))
                boundingBox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boundingBox, confs, confidenceThreshHold, nmsThreshold)

    personCount = 0
    # book, laptop, cellphone, and person
    for i in indices:
        i = i[0]
        box = boundingBox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        personCount = personCount + 1 if classNames[classIds[i]].upper() == 'PERSON' else personCount

        object_class = classNames[classIds[i]].upper()
        # combine all objects into single class
        if object_class == 'CELL PHONE' or object_class == 'LAPTOP' or object_class == 'BOOK':
            # add text to existing object detection description
            if frame_time in objects_t:
                objects[len(objects) - 1] = objects[len(objects) - 1] + object_class
            else:
                objects.append(object_class)
                objects_t.append(frame_time)
    person_count.append(personCount)
    person_count_t.append(f)


def yoloObjectDetection(img, f):
    yoloBlog = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    darkNet.setInput(yoloBlog)
    layerNames = darkNet.getLayerNames()
    outputNames = [layerNames[i[0] - 1] for i in darkNet.getUnconnectedOutLayers()]

    outputs = darkNet.forward(outputNames)
    findObjects(outputs, img, f)
    return


def imageClassify(img, h, w, f):
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    imageWidth = img.shape[1]
    imageHeight = img.shape[0]
    paddingX = math.floor(imageWidth * 0.01)
    paddingY = math.floor(imageHeight * 0.01)

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # show the indicator only when confidence is above 50%
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # add paddings to sides
            x1 = x1 if x1 - paddingX < 0 else x1 - paddingX
            x2 = x2 if x2 + paddingX > imageWidth else x2 + paddingX
            y1 = y1 if y1 - paddingY < 0 else y1 - paddingY
            y2 = y2 if y2 + paddingY > imageHeight else y2 + paddingY
            face_img = img[y1:y2, x1:x2]

            predict_result, label, accuracy = learn_inf.predict(face_img)
            label = label.data.numpy()
            accuracy = (accuracy.data[label]).numpy() * 100

            if predict_result == "high_risk":
                high.append(accuracy)
                high_t.append(f)
            else:
                low.append(accuracy)
                low_t.append(f)

            risk_counter[predict_result] = risk_counter[predict_result] + 1

            message = predict_result + " " + str(round(accuracy, 2)) + "%"
            # drawing a rectangle and coloring the rectangle based on prediction result
            cv2.rectangle(img, (x1, y1), (x2, y2), risk_color[predict_result], 2)

            cv2.putText(img, message, (x1, y1), font, fontScale,
                        risk_color[predict_result], thickness, cv2.LINE_AA)

            print("F:", f, " ", predict_result, " ", accuracy)


def run(path):
    # path ="/home/ese440/PycharmProjects/ESE440/resources/test-video_70s.mp4"
    cap = cv2.VideoCapture(path)
    start = time.time()
    prev_time = -1
    processed_frames = 0
    while True:
        status, img = cap.retrieve(cap.grab()) # time that the frame was captured
        fr = cap.get(cv2.CAP_PROP_POS_MSEC)
        (h, w) = img.shape[:2]
        if status:

            # adjust the denominator to skip frames, higher denominator number =>>>  more frame skipping
            time_s = fr / 1000
            if int(time_s) > int(prev_time):
                processed_frames += 1
                # imageClassify(img, h, w, fr/1000)
                yoloObjectDetection(img, fr/1000)

                cv2.imshow('Video', cv2.resize(img, (round(img.shape[1] / 2), round(img.shape[0] / 2))))

                if cv2.waitKey(1) & 0XFF == ord('q'):
                    break

            prev_time = time_s
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    end = time.time()

    return high, high_t, low, low_t, book, book_t, laptop, laptop_t, cell, cell_t, person, person_t,person_count, person_count_t;


def run_test(image_folder_path):
    sorted_file_list = sorted(os.listdir(image_folder_path), key=lambda x: int(x.split(".")[0]))
    start = time.time()
    print("Start:", start)
    # read image from folder
    for input_filename in sorted_file_list:
        #  with open(image_folder_path + '/' + input_filename, 'rb') as image:
        image = cv2.imread(image_folder_path + input_filename)
        yoloObjectDetection(image, int(re.search('(.+)\\.png', input_filename).groups()[0]) / 1000)
    end = time.time()
    print("End:", end)
    print(end-start)
    return objects, objects_t, person_count, person_count_t


image_folder = "/home/ese440/PycharmProjects/ESE440/image_samples/"
if __name__ == "__main__":
    run_test(image_folder)
