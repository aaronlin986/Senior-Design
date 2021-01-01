import cv2
from fastai.vision.all import load_learner
import numpy as np
import dlib
import time

start = time.time()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/hang/PycharmProjects/MaskDetector/venv/models"
                                 "/shape_predictor_68_face_landmarks.dat")
learn_inf = load_learner('/home/hang/PycharmProjects/MaskDetector/venv/models/risk_v3.pkl')
cap = cv2.VideoCapture("/home/hang/PycharmProjects/MaskDetector/venv/Resources/demo.mp4")

# font
font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 2
# Line thickness of 2 px
thickness = 1

risk_color = {"low_risk": (0, 255, 0), "high_risk": (0, 0, 255)}
risk_counter = {"low_risk": 0, "high_risk": 0}

prev_time = -1
processed_frames = 0
sample_rate = 1

# FACIAL_LANDMARKS_IDXS = OrderedDict([
# 	("mouth", (48, 68)),
# 	("right_eyebrow", (17, 22)),
# 	("left_eyebrow", (22, 27)),
# 	("right_eye", (36, 42)),
# 	("left_eye", (42, 48)),
# 	("nose", (27, 35)),
# 	("jaw", (0, 17))
# ])

modelFile = "/home/hang/PycharmProjects/MaskDetector/venv/models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "/home/hang/PycharmProjects/MaskDetector/venv/models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def face_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


pause = True

while True:

    status, img = cap.retrieve(cap.grab())
    # resize image for better processing speed
    (h, w) = img.shape[:2]
    if status:
        # img2= cv2.resize(img, (round(img.shape[1] / 2), round(img.shape[0] / 2)))
        # adjust the denominator to skip frames, higher number =  more frame skipping
        time_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 150
        # time_s = cap.get(cv2.CAP_PROP_POS_MSEC)
        if int(time_s) > int(prev_time):

            processed_frames += 1
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))

            net.setInput(blob)
            detections = net.forward()
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")

                    face_img = img[y1-50:y2+50, x1-50:x2+50]
                    # face_img = img[y1:y2, x1:x2]
                    cv2.imshow('face',face_img)
                    predict_result, label, accuracy = learn_inf.predict(face_img)
                    label = label.data.numpy()
                    accuracy = (accuracy.data[label]).numpy() * 100

                    risk_counter[predict_result] = risk_counter[predict_result] + 1
                    message = predict_result + " " + str(round(accuracy, 2)) + "%"
                    # drawing a rectangle and coloring the rectangle based on prediction result
                    cv2.rectangle(img, (x1-50, y1-50), (x2+50, y2+50), risk_color[predict_result], 2)
                    # cv2.rectangle(img, (x1,y1), (x2, y2), risk_color[predict_result], 2)
                    cv2.putText(img, message, (x1, y1-50), font, fontScale,
                                risk_color[predict_result], thickness, cv2.LINE_AA)

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
print("Average fps:", processed_frames / 20)
