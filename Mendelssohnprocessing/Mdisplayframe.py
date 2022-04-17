from Mvideoframestoarray import *
import matplotlib.pyplot as plt

frame_index = 1688

frame = images[int(frame_index), :, :, :]
frame = cv2.normalize(src = frame, dst = None, alpha = 0,
                      beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#print(images.shape)
plt.imshow(frame)
plt.show()