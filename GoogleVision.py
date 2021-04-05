from google.cloud import vision
from PIL import Image, ImageDraw


def configure():
    import os
    os.environ[
        "GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ese440/PycharmProjects/ESE440/resources/ese440-ee15808ee7b1.json"

def detect_face(face_file,max_results=4):
    client = vision.ImageAnnotatorClient()
    content = face_file.read()
    image = vision.Image(content=content)
    faceAnnotation = client.face_detection(image=image, max_results=max_results).face_annotations
    return faceAnnotation



def highlight_faces(image,faces):
    im = Image.open(image)
    draw = ImageDraw.Draw(im)
    # Sepecify the font-family and the font-size
    for face in faces:
        print(faces)
        print("---------------------------------------")
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
    im.show()

def main(input_filename, max_results):
    configure()
    with open(input_filename, 'rb') as image:
        faces = detect_face(image, max_results)
        print('Found {} face{}'.format(
            len(faces), '' if len(faces) == 1 else 's'))

        # Reset the file pointer, so we can read the file again
        image.seek(0)
        highlight_faces(image, faces)

file_name = "/home/ese440/PycharmProjects/ESE440/resources/Left0.PNG"
main(file_name,4)