import tkinter as tk
from tkinter import filedialog, Toplevel
import VisualDetection
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import moviepy.editor as mp
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import json

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

@app.callback(Output('tabular', 'figure'),
               Input('graph', 'clickData'))
def clickCallback(clickData):
    if clickData is None:
        table = go.Figure(data=[go.Table(
        header=dict(values=['Time (secs)', 'Confidence (%)'], font_size=18),
        cells=dict(values=[[],[]], font_size=14)
        )])
    else:
        table = go.Figure(data=[go.Table(
            header=dict(values=['Time', 'Confidence'], font_size=18),
            cells=dict(values=[[round(clickData['points'][0]['x'], 2)],
                               [round(clickData['points'][0]['y'], 2)]], font_size=14
                       )
        )])

    return table

@app.callback(Output('graph', 'figure'),
              Input('upload', 'filename'), prevent_initial_call=True)
def updateGraph(filename):
    file = "/home/ese440/PycharmProjects/ESE440/resources/" + filename
    high, high_t, low, low_t, book, book_t, laptop, laptop_t, cell, cell_t, person, person_t, person_count, person_count_t = VisualDetection.run(file)
    f = make_subplots(rows=2, cols=1)
    f.add_trace(go.Scatter(x=high_t, y=high, mode='markers', name='high risk', marker_size=10, marker_line_width=2, marker_color='red'), row=1, col=1)
    f.add_trace(go.Scatter(x=book_t, y=book, mode='markers', name='book'), row=2, col=1)
    f.add_trace(go.Scatter(x=laptop_t, y=laptop, mode='markers', name='laptop'), row=2, col=1)
    f.add_trace(go.Scatter(x=cell_t, y=cell, mode='markers', name='cell'), row=2, col=1)
    f.add_trace(go.Scatter(x=person_t, y=person, mode='markers', name='person'), row=2, col=1)
    return go.FigureWidget(f)



def buildLayout():
    layout = html.Div([
        dcc.Upload(id='upload',
                   style={
                       'textAlign': 'center'
                   },
                   children=html.Button(id='button',
                                        children='Upload File',
                                        style={
                                            'width': '150px',
                                            'height': '75px',
                                            'margin': '50px',
                                            'padding': '0px'
                                        })
        ),
        dcc.Loading(children=dcc.Graph(id='graph')),
        dcc.Graph(id='tabular')
    ])
    return layout

def extractAudio (fileName):
    my_clip = mp.VideoFileClip(fileName)
    audio_file_name = fileName.replace('.mp4', '.mp3')
    my_clip.audio.write_audiofile()
    return audio_file_name

if __name__ == '__main__':
    app.layout = buildLayout()
    app.run_server(debug=True)
