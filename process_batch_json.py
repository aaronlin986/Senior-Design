import json
import os, glob
import re
from google.cloud import vision


# this file process the json output download from google cloud storage
# and output frame_time, number of faces, and angles of faces
def configure():
    os.environ[
        "GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ese440/PycharmProjects/ESE440/resources/ese440-ee15808ee7b1.json"


def detect_face(image, max_results=4):
    client = vision.ImageAnnotatorClient()
    # [END vision_face_detection_tutorial_client]
    content = image.read()
    image = vision.Image(content=content)
    return client.face_detection(
        image=image, max_results=max_results).face_annotations


def process_face_data(faces):
    number_of_faces = len(faces)
    faces_data = {}
    # only expect single image in calibration frame
    for face in faces:
        detection_confidence = round(face.detection_confidence, 2)
        face_pan_angle = round(face.pan_angle, 2)
        face_tilt_angle = round(face.tilt_angle, 2)
        face_roll_angle = round(face.roll_angle, 2)
        faces_data = {"pan_angle": face_pan_angle, "tilt_angle": face_tilt_angle, "roll_angle": face_roll_angle}
    return faces_data


def calibration_frame_data(image_path):
    # use a proper posture image to calibrate
    # ex. facing the camera
    calibration_data = {}
    with open(image_path, 'rb') as image:
        faces = detect_face(image, 10)
        calibration_data = process_face_data(faces)
    print('calibration is',calibration_data)
    return calibration_data


def head_pose_flag(angles):
    # maximum thresh that trigger flag
    PAN_MAX = 20
    TILT_UP_MAX = 15
    TILT_DOWN_MAX = -8
    ROLL_MAX = 20
    pan_angle_text = None
    tilt_angle_text = None
    roll_angle_text = None

    # angles in order of pan, tilt and roll
    if angles[0] > PAN_MAX:
        pan_angle_text = "Right"
    elif angles[0] < -PAN_MAX:
        pan_angle_text = "Left"

    if angles[1] > TILT_UP_MAX:
        tilt_angle_text = "Up"
    elif angles[1] < TILT_DOWN_MAX:
        tilt_angle_text = "Down"

    if angles[2] > ROLL_MAX:
        roll_angle_text = "Roll Right"
    elif angles[2] < -ROLL_MAX:
        roll_angle_text = "Roll Left"

    return [
        pan_angle_text,
        tilt_angle_text,
        roll_angle_text
    ]


def process_json(folder_path,calibration):

    # use a reference frame to calibrate head pose - should give better result
    #subtract if positive, add otherwise
    PAN_ADJUST = 0 if not calibration["pan_angle"] else calibration["pan_angle"]
    TILT_ADJUST = 0 if not calibration["tilt_angle"] else calibration["tilt_angle"]
    ROLL_ADJUST = 0 if not calibration["roll_angle"] else calibration["roll_angle"]
    #add a calibration here

    files = glob.glob(os.path.join(folder_path, "*"))

    frame_data = []
    number_of_faces = []
    number_of_faces_time = []
    frame_times = []
    head_poses = []
    head_poses_time = []

    for file in files:
        with open(file) as json_file:
            data = json.load(json_file)
            # loop over each frame
            for face_annotations in data["responses"]:
                frame_time = int(re.search('.+image_samples/(.+)\\.png',
                                           face_annotations["context"]["uri"]).groups()[0]) / 1000
                # # each frame may contain multiple faces
                head_pose_in_image = []

                for face_annotation in face_annotations["faceAnnotations"]:
                    # return a dictionary
                    # maybe store only known faces

                    head_pose_single = head_pose_flag([face_annotation['panAngle'] + PAN_ADJUST,
                                                       face_annotation['tiltAngle'] - TILT_ADJUST,
                                                       face_annotation['rollAngle'] + ROLL_ADJUST])
                    head_pose_in_image.append(head_pose_single)
                    print("time",frame_time,head_pose_single)
                frame_data.append({'frame_time': frame_time,
                                   'number_of_faces': len(face_annotations["faceAnnotations"]),
                                   "head_poses": head_pose_in_image})

    # sort data based on time, this is not needed if results in json file are sorted
    frame_data.sort(key=lambda e: e['frame_time'])

    # split data to time array, number of faces array and head poses array
    for data in frame_data:
        if data['number_of_faces'] != 1:
            number_of_faces.append(data['number_of_faces'])
            number_of_faces_time.append(data['frame_time'])

        if data['head_poses'][0][0] or data['head_poses'][0][1] or data['head_poses'][0][2]:
            head_pose = data['head_poses'][0][0] or data['head_poses'][0][1] or data['head_poses'][0][2]
            head_poses.append(head_pose)
            head_poses_time.append(data['frame_time'])

    return number_of_faces, number_of_faces_time, head_poses, head_poses_time


def run(calibration):
    configure()
    path_to_json = "/home/ese440/PycharmProjects/ESE440/annotated_json/ese440_batch_output/"
    return process_json(path_to_json,calibration)


if __name__ == "__main__":
    run()