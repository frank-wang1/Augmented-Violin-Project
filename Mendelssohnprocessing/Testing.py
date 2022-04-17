from Mvideoframestoarray import *
import matplotlib.pyplot as plt
import math


def siftKeypoints(frameNumber):
    gray = vidFrametoGray(frameNumber)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints


def vidFrametoGray(frameNumber):
    image = images[frameNumber, :, :, :]
    image = cv2.normalize(src=image, dst=None, alpha=0,
                          beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray, cmap = "gray")
    # plt.show()
    return gray


framerate = 30
kparray = np.array([280, 282, 304, np.inf, 4, 5, np.inf, np.inf, np.inf, np.inf])
velarray = np.full(10, np.inf)
i = 0
j = 1

while i != kparray.size - 1:
    if kparray[i] == np.inf:
        i += 1
        continue
    print('i is', i)
    while j != kparray.size:
        print('j is', j)
        if kparray[j] == np.inf:
            j += 1
            continue
        else:
            secondx = siftKeypoints(j)[int(kparray[j])].pt[0]
            secondy = siftKeypoints(j)[int(kparray[j])].pt[1]
            firstx = siftKeypoints(i)[int(kparray[i])].pt[0]
            firsty = siftKeypoints(i)[int(kparray[i])].pt[1]
            time = (j - i) / framerate
            distance = math.sqrt(((secondx - firstx) ** 2) + ((secondy - firsty) ** 2))
            velarray[j] = distance / time
            print(distance/time)
            i = j
            j = i + 1
            break
velarray[0] = 0
#print(velarray)
yvalues = []
for velovalue in velarray:
    if velovalue != np.inf:
        yvalues.append(velovalue)
print(yvalues)
