import cv2
import numpy as np

cap = cv2.VideoCapture('/Users/frankwang/Desktop/Visual Media REU/OpenMV Camera Recordings/MendelssohnVideo.mp4')
path = '/Users/frankwang/Desktop/Visual Media REU/OpenMV Camera Recordings/MendelssohnVideo.mp4'

pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
total_frames = 1687 # Replace 1687 w/   int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   for all other videos
# the video I am using is messed up; opencv thinks there are more frames than what there actually is
# most likely because I trimmed the video using the QuickTime editor which messes up the code because
# frame_ready will be False, storing the images[] with None values.

a, frame = cap.read()
images = np.zeros((total_frames, frame.shape[0], frame.shape[1], frame.shape[2]))

for count in range(0, total_frames):
    frame_ready, frame = cap.read()
    if not frame_ready:
        #print('frame_ready is False, broken. Stopping at frame', count)
        break
    images[count, :, :, :] = frame
