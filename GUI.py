import tkinter as tk
from tkinter import filedialog, Text

def selectFile() :
    fileName = filedialog.askopenfilename(initialdir="/", title="Select File", filetypes=(("videos", "*.mpg"), ("all files", "*.*")))

def run():
    root = tk.Tk()
    canvas = tk.Canvas(root, height=500, width=800)
    canvas.pack()

    openFile = tk.Button(root, text="Select file", padx=10, pady=5, fg="black", command=selectFile)
    openFile.pack()

    root.mainloop()

if __name__ == "__main__":
    run()



