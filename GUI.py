import tkinter as tk
from tkinter import filedialog, Toplevel
import VisualDetection
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import moviepy.editor as mp
import dash
import dash_core_components as dcc
import dash_html_components as html
import json
import threading

app = dash.Dash()
high = []
high_t = []
low = []
low_t = []

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
word = []
word_t = []

fileName = ""
tabWindow = None
scatter = go.FigureWidget()

def changeTabularState():
    if fileName != "":
        tabButton['state'] = tk.NORMAL

def tabularWindow():
    global tabWindow
    if tabWindow == None or not tk.Toplevel.winfo_exists(tabWindow):
        tabWindow = Toplevel(root)
        tabWindow.title("Tabular")
    else:
        tabWindow.lift()

@app.callback(dash.dependencies.Output('header', 'children'),
               [dash.dependencies.Input('graph', 'clickData')], prevent_initial_call=True)
def clickCallback(clickData):
    tabularWindow()
    return json.dumps(clickData)

def buildLayout(scatter):
    layout = html.Div([
        html.H1(id='header', children="HELLO"),
        dcc.Graph(id='graph', figure=scatter)
    ])
    return layout

def extractAudio (fileName):
    my_clip = mp.VideoFileClip(fileName)
    audio_file_name = fileName.replace('.mp4', '.mp3')
    my_clip.audio.write_audiofile()
    return audio_file_name

def runServer(app):
    app.run_server()

def selectFile() :
    global fileName
    fileName = filedialog.askopenfilename(initialdir="/", title="Select File", filetypes=(("videos", "*.mp4"), ("all files", "*.*")))
    #extractAudio(fileName)
    high, high_t, low, low_t, book, book_t, laptop, laptop_t, cell, cell_t, person, person_t,person_count, person_count_t = VisualDetection.run(fileName)

    f = make_subplots(rows=2, cols=1)
    f.add_trace(go.Scatter(x=high_t, y=high, mode='markers', name='high risk', marker_size=10, marker_line_width=2, marker_color='red'), row=1, col=1)
    f.add_trace(go.Scatter(x=book_t, y=book, mode='markers', name='book'), row=2, col=1)
    f.add_trace(go.Scatter(x=laptop_t, y=laptop, mode='markers', name='laptop'), row=2, col=1)
    f.add_trace(go.Scatter(x=cell_t, y=cell, mode='markers', name='cell'), row=2, col=1)
    f.add_trace(go.Scatter(x=person_t, y=person, mode='markers', name='person'), row=2, col=1)
    global scatter
    scatter = go.FigureWidget(f)

    changeTabularState()

root = tk.Tk()
canvas = tk.Canvas(root, height=500, width=800)
canvas.pack()

openFile = tk.Button(root, text="Select file", padx=10, pady=5, fg="black", command=selectFile)
tabButton = tk.Button(root, text="Tabular", padx=10, pady=5, fg="black", state=tk.DISABLED, command=tabularWindow)
openFile.pack()
tabButton.pack()

# scatter = go.FigureWidget(go.Scatter(y=[1, 2, 3, 4]))
app.layout = buildLayout(scatter)
t1 = threading.Thread(target=runServer, args=(app,))
t1.start()
root.mainloop()




