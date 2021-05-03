import moviepy.editor as mp
from pydub import AudioSegment

def extract_audio_test (videoName=""):
    my_clip = mp.VideoFileClip(r"/home/ese440/PycharmProjects/ESE440/resources/test-video-no-voice.mp4")
    my_clip.audio.write_audiofile(r"test.mp3")
