from google.cloud import vision
from PIL import Image, ImageDraw
import os


# this file is for reference only
def configure():
    os.environ[
        "GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ese440/PycharmProjects/ESE440/resources/ese440-ee15808ee7b1.json"


def read_images(image_folder_path):
    sorted_file_list = sorted(os.listdir(image_folder_path), key=lambda x: int(x.split(".")[0]))
    for input_filename in sorted_file_list:
        with open(image_folder_path+'/'+input_filename, 'rb') as image:
            print("-----------------------------")
            print("Info of ", input_filename)
            faces = detect_face(image, 10)
            process_face_data(faces)
            print('Found {} face{}'.format(
                len(faces), '' if len(faces) == 1 else 's'))


def detect_face(image,max_results=4):
    client = vision.ImageAnnotatorClient()
    # [END vision_face_detection_tutorial_client]
    content = image.read()
    image = vision.Image(content=content)
    return client.face_detection(
        image=image, max_results=max_results).face_annotations


def process_face_data(faces):
    number_of_faces = len(faces)
    faces_data = [] # contains data of all faces
    for face in faces:
        detection_confidence = round(face.detection_confidence,2)
        face_pan_angle = round(face.pan_angle, 2)
        face_tilt_angle = round(face.tilt_angle, 2)
        face_roll_angle = round(face.roll_angle, 2)
        faces_data.append([detection_confidence,face_pan_angle, face_tilt_angle, face_roll_angle])
    print("Face Detection Confidence, Pan Angle, Tilt Angle, Roll Angle")
    print(faces_data)


def highlight_faces(image,faces):
    im = Image.open(image)
    draw = ImageDraw.Draw(im)
    for face in faces:
        print(face)
        box = [(vertex.x, vertex.y)
               for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=5, fill='#00ff00')
        # Place the confidence value/score of the detected faces above the
        # detection box in the output image
        draw.text(((face.bounding_poly.vertices)[0].x,
                   (face.bounding_poly.vertices)[0].y - 30),
                  str(format(face.detection_confidence, '.3f')) + '%',
                  fill='#FF0000')
        draw.text(((face.bounding_poly.vertices)[0].x + 30,
                   (face.bounding_poly.vertices)[0].y - 70),
                  "Pan Angle:"+str(format(face.pan_angle, '.3f')) + 'degrees',fill='#FF0000')

        draw.text(((face.bounding_poly.vertices)[0].x + 30,
                 (face.bounding_poly.vertices)[0].y - 90),
                "Roll Angle: " + str(format(face.roll_angle, '.3f')) + 'degrees',fill='#FF0000')

        draw.text(((face.bounding_poly.vertices)[0].x + 30,
                 (face.bounding_poly.vertices)[0].y - 110),
                "Tilt Angle: " + str(format(face.tilt_angle, '.3f')) + 'degrees',fill='#FF0000')
    im.show(draw)


def main(input_filename, max_results):

    configure()
    read_images(input_filename)


file_name = "/home/ese440/PycharmProjects/ESE440/image_samples/0.png"
folder_name = "/home/ese440/PycharmProjects/ESE440/image_samples"
main(folder_name,10)