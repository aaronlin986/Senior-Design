import tkinter as tk
from tkinter import filedialog, Toplevel
import VisualDetection
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import moviepy.editor as mp
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
import process_batch_json
from file_uploader import upload_blob, parallel_upload_google_cloud, json_file_download
from AudioDetection import transcribe_gcs
from VideoToImages import video_to_frames
from batch_image_annotation import async_batch_annotate_images
import VisualDetection

app = dash.Dash()


# high = []
# high_t = []
# low = []
# low_t = []
#
# book = []
# book_t = []
# laptop = []
# laptop_t = []
# cell = []
# cell_t = []
# person = []
# person_t = []
# person_count = []
# person_count_t = []
# word = []
# word_t = []


# @app.callback(Output('tabular', 'figure'),
#               Output('videoclip', 'src'),
#               Input('graph', 'clickData'), prevent_initial_call=True)
# def clickCallback(clickData):
#     # @TODO show all data instead of confidence
#     if clickData is None:
#         table = go.Figure(data=[go.Table(
#             header=dict(values=['Time (secs)', 'Confidence (%)'], font_size=18),
#             cells=dict(values=[[], []], font_size=14)
#         )])
#     else:
#         table = go.Figure(data=[go.Table(
#             header=dict(values=['Time', 'Confidence'], font_size=18),
#             cells=dict(values=[[round(clickData['points'][0]['x'], 2)],
#                                [round(clickData['points'][0]['y'], 2)]], font_size=14
#                        )
#         )])
#
#     # video = VideoFileClip("/home/ese440/PycharmProjects/ESE440/resources/test-video-voice.mp4")
#     # t1 = 0
#     # t2 = video.duration
#     # if clickData['points'][0]['x'] > 3 :
#     #     t1 = clickData['points'][0]['x'] - 3
#     # if clickData['points'][0]['x'] + 3 < video.duration :
#     #     t2 = clickData['points'][0]['x'] + 3
#     # # ffmpeg_extract_subclip("/home/ese440/PycharmProjects/ESE440/resources/test-video-voice.mp4", t1, t2, targetname="clip.mp4")
#     # print(t1)
#     # print(t2)
#     # video.cutout(t1, t2).write_videofile("clip.mp4")
#
#
#     return table, "clip.mp4"


@app.callback(Output('graph', 'figure'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              prevent_initial_call=True)
def updateGraph(list_of_contents, list_of_names):
    configure()  # import environmental variable for google cloud apis

    id_filename = list_of_names[0]  # assume photo file is chosen first
    filename = list_of_names[1]

    project_path = "/home/ese440/PycharmProjects/ESE440/"
    file_full_path = project_path + "resources/" + filename
    id_file_full_path = project_path + "resources/" + id_filename

    image_folder = "/home/ese440/PycharmProjects/ESE440/image_samples/"

    calibration_data = process_batch_json.calibration_frame_data(id_file_full_path)

    data = data_generator(project_path, file_full_path, filename,calibration_data)
    text_data = data[0]
    head_pose_data = data[1]
    words, words_start, words_end = text_data
    number_of_faces, number_of_faces_time, head_poses, head_poses_time = head_pose_data

    # combine objects into one array
    objects, objects_t, person_count, person_count_t = VisualDetection.run_test(image_folder)

    face_counter = []  # len will be the same
    for index, value in enumerate(number_of_faces):
        face_counter.append(max(value,person_count[index]))

    f = make_subplots(rows=4, cols=1,shared_xaxes=True,subplot_titles=("Number of Faces", 'Head Orientation', 'Speech Detection',
                                                      'Object Detection'))

    # show # of faces and person
    f.add_trace(
        go.Scatter(x=number_of_faces_time, y=face_counter, mode='markers+text', name='Number of faces', marker_size=10,
                   text=face_counter,textposition="top center",
                   marker_line_width=2, marker_color='red'), row=1, col=1)

    # head poses
    f.add_trace(go.Scatter(x=head_poses_time, y=[0] * len(head_poses_time),
                           text=head_poses, mode="markers+text", name='Head Orientation',
                           marker_size=10, marker_line_width=2, marker_color='blue', textposition="bottom center"),
                row=2, col=1)

    # speech text
    f.add_trace(go.Scatter(x=words_start, y=[0] * len(words),
                           text=words, mode="markers+text", name='Speech Detection',
                           marker_size=10, marker_line_width=2, marker_color='yellow', textposition="bottom center"),
                row=3, col=1)

    # Object detection
    f.add_trace(go.Scatter(x=objects_t, y=[0] * len(objects),
                           text=objects, mode="markers+text", name='Object Detection',
                           marker_size=10, marker_line_width=2, marker_color='pink', textposition="bottom center"),
                row=4, col=1)

    f.update_xaxes(range=[-1, number_of_faces_time[len(number_of_faces_time) - 1]])
    f.update_xaxes(title_text="Time in second", row=4, col=1)
    f.update_yaxes(showticklabels=False)  # turn off all y axis
    f.update_yaxes(showticklabels=True, row=1, col=1)  # turn on selective y axis
    f.update_layout(title_text="Detection Result", height=800)
    f.update_layout(legend_orientation="h", xaxis2_rangeslider_visible=True, xaxis2_rangeslider_thickness=0.1)
    return go.FigureWidget(f)


def buildLayout():
    layout = html.Div([
        dcc.Upload(id='upload-data',
                   style={
                       'textAlign': 'center'
                   },
                   children=html.Button(id='button_id',
                                        children='Upload Id and Video',
                                        style={
                                            'width': '150px',
                                            'height': '75px',
                                            'margin': '25px',
                                            'padding': '0px'
                                        }), multiple=True),
        dcc.Loading(children=dcc.Graph(id='graph')),
        dcc.Graph(id='tabular'),
        html.Video(id='videoclip', autoPlay=True,controls=True, src='/home/ese440/PycharmProjects/ESE440/resources/test-video-voice.mp4')
    ])
    return layout


def extractAudio(fileName):
    my_clip = mp.VideoFileClip(fileName)
    audio_file_name = fileName.replace('.mp4', '.mp3')
    my_clip.audio.write_audiofile(audio_file_name)
    return audio_file_name


def generate_text_data(filename, file_full_path):
    #  1.extract audio
    audio_file_full_path = extractAudio(file_full_path)

    #  2. upload audio to Google Storage
    destination_blob_name = upload_blob(filename.replace('.mp4', '.mp3'), audio_file_full_path)
    #  3 and 4. process audio file in the Google Storage and store text
    # @TODO this takes long time,result is from google cloud, maybe process other function while waiting
    words, words_start, words_end = transcribe_gcs("gs://ese440_audios/" + destination_blob_name)
    return words, words_start, words_end


def generate_head_poses_data(project_path, file_full_path,calibration_data):
    #  5. extract video to images
    video_to_frames(file_full_path, './image_samples/', 2)

    #  6. upload images to Google Storage
    gcs_images_bucket = "ese440_test_images"
    parallel_upload_google_cloud(project_path + "image_samples/", "ese440_test_images")

    #  7. process image files in the Google Storage
    async_batch_annotate_images(gcs_images_bucket, "gs://" + gcs_images_bucket + "/")
    # high, high_t, low, low_t, book, book_t, laptop, laptop_t, cell, cell_t, person, person_t, person_count, person_count_t = VisualDetection.run(file)

    #  8. download the json files from google cloud storage
    json_src_folder = "ese440_batch_output"
    json_des_name = "/home/ese440/PycharmProjects/ESE440/annotated_json/"
    json_file_download(json_des_name, json_src_folder)
    #  9. process the json files and get headposes
    (number_of_faces, number_of_faces_time, head_poses, head_poses_time) = process_batch_json.run(calibration_data)
    return number_of_faces, number_of_faces_time, head_poses, head_poses_time


def data_generator(project_path, file_full_path, file_name,calibration_data):
    words, words_start, words_end = generate_text_data(file_name, file_full_path)
    number_of_faces, number_of_faces_time, head_poses, head_poses_time = generate_head_poses_data(project_path,
                                                                                                        file_full_path,calibration_data)
    return [(words, words_start, words_end), (number_of_faces, number_of_faces_time, head_poses, head_poses_time)]


def configure():
    import os
    os.environ[
        "GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ese440/PycharmProjects/ESE440/resources/ese440-ee15808ee7b1.json"


if __name__ == '__main__':
    app.layout = buildLayout()
    app.run_server(debug=True)
