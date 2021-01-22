# Combining code from cheatModel.py and yoloModel.py

import cv2
from fastai.vision.all import load_learner
import numpy as np
import time
import multiprocessing
# @ToBeRefactor------------------------some constant
learn_inf = load_learner('/home/ese440/PycharmProjects/ESE440/models/risk_v3.pkl', cpu=True)
cap = cv2.VideoCapture("/home/ese440/PycharmProjects/ESE440/resources/testvideo.mp4")

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

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def face_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    boundingBox = []
    classIds = []
    confs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            # print(confidence)
            if confidence > confidenceThreshHold:
                w = int(detection[2] * wT)
                h = int(detection[3] * hT)
                x = int((detection[0] * wT) - (w / 2))
                y = int((detection[1] * hT) - (h / 2))
                boundingBox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boundingBox, confs, confidenceThreshHold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = boundingBox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


def yoloObjectDetection(img):
    yoloBlog = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    darkNet.setInput(yoloBlog)
    layerNames = darkNet.getLayerNames()
    outputNames = [layerNames[i[0] - 1] for i in darkNet.getUnconnectedOutLayers()]

    outputs = darkNet.forward(outputNames)
    findObjects(outputs, img)
    return


def imageClassify(img, h, w):
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # show the indicator only when confidence is above 50%
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            face_img = img[y1 - 50:y2 + 50, x1 - 50:x2 + 50]

            # cv2.imshow('face',face_img)   --- use to show face portion
            predict_result, label, accuracy = learn_inf.predict(face_img)
            label = label.data.numpy()
            accuracy = (accuracy.data[label]).numpy() * 100

            risk_counter[predict_result] = risk_counter[predict_result] + 1

            message = predict_result + " " + str(round(accuracy, 2)) + "%"
            # drawing a rectangle and coloring the rectangle based on prediction result
            cv2.rectangle(img, (x1 - 50, y1 - 50), (x2 + 50, y2 + 50), risk_color[predict_result], 2)
            # cv2.rectangle(img, (x1,y1), (x2, y2), risk_color[predict_result], 2)
            cv2.putText(img, message, (x1, y1 - 50), font, fontScale,
                        risk_color[predict_result], thickness, cv2.LINE_AA)

            f = cap.get(cv2.CAP_PROP_POS_MSEC)
            print("F:", f, " ", predict_result, " ", accuracy)


def run():
    start = time.time()
    prev_time = -1
    processed_frames = 0
    while True:
        status, img = cap.retrieve(cap.grab())
        (h, w) = img.shape[:2]
        if status:

            # adjust the denominator to skip frames, higher denominator number =>>>  more frame skipping
            time_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 200
            # time_s = cap.get(cv2.CAP_PROP_POS_MSEC)
            if int(time_s) > int(prev_time):
                processed_frames += 1

                # multiprocessing version

                # p1 = multiprocessing.Process(imageClassify(img, h, w))
                # p2 = multiprocessing.Process(yoloObjectDetection(img))
                # p1.start()
                # p2.start()
                # p1.join()
                # p2.join()

                # regular

                imageClassify(img,h,w)
                yoloObjectDetection(img)

                cv2.imshow('Video', cv2.resize(img, (round(img.shape[1] / 2), round(img.shape[0] / 2))))

                if cv2.waitKey(1) & 0XFF == ord('q'):
                    break

            prev_time = time_s
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    end = time.time()
    print("Low risk frames:", risk_counter["low_risk"])
    print("High risk frames:", risk_counter["high_risk"])
    print("Total time taken to process: {0} second".format(round(end - start, 2)))
    print("Processed total: {0} frames".format(processed_frames))
    print("Average fps:", processed_frames / 10)


if __name__ == "__main__":
    run()
