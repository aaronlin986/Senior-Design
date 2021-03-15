import tkinter as tk
from tkinter import filedialog, Text
import VisualDetection
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import moviepy.editor as mp


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


def extractAudio (fileName):
    my_clip = mp.VideoFileClip(fileName)
    audio_file_name = fileName.replace('.mp4', '.mp3')
    my_clip.audio.write_audiofile()
    return audio_file_name

def selectFile() :
    fileName = filedialog.askopenfilename(initialdir="/", title="Select File", filetypes=(("videos", "*.mp4"), ("all files", "*.*")))
    # extractAudio(fileName)
    high, high_t, low, low_t, book, book_t, laptop, laptop_t, cell, cell_t, person, person_t,person_count, person_count_t = VisualDetection.run(fileName)

    fig = make_subplots(rows=2, cols=1) #go.Figure()
    fig.add_trace(go.Scatter(x=low_t, y=low, mode='markers', name='low risk'), row=1, col=1)
    fig.add_trace(go.Scatter(x=high_t, y=high, mode='markers', name='high risk', marker_size=10, marker_line_width=2), row=1, col=1)
    fig.show()


def run():
    root = tk.Tk()
    canvas = tk.Canvas(root, height=500, width=800)
    canvas.pack()

    openFile = tk.Button(root, text="Select file", padx=10, pady=5, fg="black", command=selectFile)
    openFile.pack()

    root.mainloop()

if __name__ == "__main__":
    run()



