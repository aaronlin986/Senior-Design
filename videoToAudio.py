import moviepy.editor as mp


def extractAudio (videoName):
    my_clip = mp.VideoFileClip(r"/home/ese440/PycharmProjects/ESE440/resources/test_3.mp4")
    my_clip.audio.write_audiofile(r"test.mp3")