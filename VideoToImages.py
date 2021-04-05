import cv2
import os, glob

# This file writes images from video to the folder

path = "/home/ese440/PycharmProjects/ESE440/resources/demo_3-28.mp4"
output_folder_path = './image_samples/'


def delete_files(folder_path):
    files = glob.glob(os.path.join(folder_path,"*"))
    for f in files:
        os.remove(f)


def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    prev_time = -1
    processed_frames = 0
    vidcap = cv2.VideoCapture(video)
    while True:
        success, image = vidcap.read()
        f = vidcap.get(cv2.CAP_PROP_POS_MSEC)
        if success:
            time_s = f / 1
            if int(time_s) > int(prev_time):
                processed_frames += 1

                image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
                cv2.imwrite(os.path.join(path_output_dir, '%d.png') % f, image)

                if cv2.waitKey(1) & 0XFF == ord('q'):
                    break
            prev_time = time_s
        else:
            break
    print('total frames',processed_frames)
    cv2.destroyAllWindows()
    vidcap.release()


video_to_frames(path, output_folder_path)
