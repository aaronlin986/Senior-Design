import cv2
import numpy as np

# yolov3 config -------------------------------------------------------------------
classesFile = '/home/ese440/PycharmProjects/ESE440/yolov3-config/coco.names'
classNames = []
with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = '/home/ese440/PycharmProjects/ESE440/yolov3-config/yolov3-tiny.cfg'
modelWeights = '/home/ese440/PycharmProjects/ESE440/yolov3-config/yolov3-tiny.weights'

darkNet = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
darkNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
darkNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
whT = 320

cap = cv2.VideoCapture("/home/ese440/PycharmProjects/ESE440/resources/testvideo.mp4")


confidenceThreshHold = 0.5
nmsThreshold = 0.3
def findObjects(outputs,img):
    hT,wT,cT = img.shape
    boundingBox = []
    classIds = []
    confs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            print(confidence)
            if confidence > confidenceThreshHold:
                w = int(detection[2] * wT)
                h = int(detection[3] * hT)
                x = int((detection[0] * wT) - (w / 2))
                y = int((detection[1] * hT) - (h / 2))
                boundingBox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boundingBox,confs,confidenceThreshHold,nmsThreshold)

    for i in indices:
        i = i[0]
        box = boundingBox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

while True:
    status, img = cap.retrieve(cap.grab())
    # img = cv2.imread('/home/ese440/PycharmProjects/ESE440/resources/person.jpeg',0)
    #yoloBlob
    yoloBlog = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    darkNet.setInput(yoloBlog)
    layerNames = darkNet.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in darkNet.getUnconnectedOutLayers()]

    outputs = darkNet.forward(outputNames)
    findObjects(outputs,img)

    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break