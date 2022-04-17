import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

from Mvideoframestoarray import *



# Total frames in images array: 1878

# success = True

# Helper function
def vidFrametoGray(frameNumber):
    image = images[frameNumber, :, :, :]
    image = cv2.normalize(src=image, dst=None, alpha=0,
                          beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray, cmap = "gray")
    # plt.show()
    return gray


def showSiftFunction(frameNumber):
    gray = vidFrametoGray(frameNumber)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    # print(np.shape(keypoints)) #returns 298    [277:281], testing 278:279
    img = cv2.drawKeypoints(gray, keypoints, None)  # Have to edit keypoints index
    # print('Descriptors shape: ', descriptors.shape)   #(403, 128)
    # print('Keypoints shape: ', np.shape(keypoints))
    # print((keypoints[0].pt)[0])
    # print(descriptors[369])
    plt.imshow(img)
    plt.show()
    # print(descriptors.shape)


def showSpecficFeature(frameNumber, kplocation):
    gray = vidFrametoGray(frameNumber)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(gray, keypoints[kplocation : kplocation + 1], None)
    plt.imshow(img, cmap = 'gray')
    plt.show()


def siftDescriptors(frameNumber):
    gray = vidFrametoGray(frameNumber)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


def siftKeypoints(frameNumber):
    gray = vidFrametoGray(frameNumber)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints


def trackNextFrame(initialframe, frametotrack, kplocation):
    sourcegray = vidFrametoGray(initialframe)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(sourcegray, None)
    sourcedes = descriptors[kplocation]  # For the 0th frame, the bow tip is from 277:281

    targetframenum = frametotrack
    targetgray = vidFrametoGray(targetframenum)
    secondSift = cv2.SIFT_create()
    secondKeypoints, secondDescriptors = secondSift.detectAndCompute(targetgray, None)
    diff = np.linalg.norm(np.full(128, np.inf))
    indexcount = 0
    finalcount = 0
    # global success
    for i in secondDescriptors:
        # first loop finds the smallest difference
        if np.linalg.norm(sourcedes - i) < diff:
            diff = np.linalg.norm(sourcedes - i)
    if diff > 300:
        return False
        # success = False
        # return kplocation
    for j in secondDescriptors:
        # global finalcount, count
        # second loop finds the index of that smallest difference
        if np.linalg.norm(sourcedes - j) == diff:
            finalcount = indexcount
            break
        else:
            indexcount += 1
    secondx = secondKeypoints[finalcount].pt[0]
    secondy = secondKeypoints[finalcount].pt[1]
    firstx = keypoints[kplocation].pt[0]
    firsty = keypoints[kplocation].pt[1]

    if math.sqrt(((secondx - firstx) ** 2) + (
            (secondy - firsty) ** 2)) > 100:
        return False
        # success = False
        # return kplocation
    image = cv2.circle(targetgray, (int(secondKeypoints[finalcount].pt[0]),
                                    int(secondKeypoints[finalcount].pt[1])), 6, (255, 0, 0), 2)
    plt.imshow(image, cmap = 'gray')
    plt.show()
    #print("Diff: ", diff, "Distance: ",
    #      math.sqrt(((secondKeypoints[finalcount].pt[0] - keypoints[kplocation].pt[0]) ** 2) + (
    #              (secondKeypoints[finalcount].pt[1] - keypoints[kplocation].pt[1]) ** 2)))
    return finalcount


def trackNextFrameSuccess(initialframe, frametotrack, kplocation):
    sourcegray = vidFrametoGray(initialframe)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(sourcegray, None)
    sourcedes = descriptors[kplocation]  # For the 0th frame, the bow tip is from 277:281
    targetframenum = frametotrack
    targetgray = vidFrametoGray(targetframenum)
    secondSift = cv2.SIFT_create()
    secondKeypoints, secondDescriptors = secondSift.detectAndCompute(targetgray, None)
    diff = np.linalg.norm(np.full(128, np.inf))
    indexcount = 0
    finalcount = 0
    # global success
    for i in secondDescriptors:
        # first loop finds the smallest difference
        if np.linalg.norm(sourcedes - i) < diff:
            diff = np.linalg.norm(sourcedes - i)
    if diff > 300:
        return False
    for j in secondDescriptors:
        if np.linalg.norm(sourcedes - j) == diff:
            finalcount = indexcount
            break
        else:
            indexcount += 1
    secondx = secondKeypoints[finalcount].pt[0]
    secondy = secondKeypoints[finalcount].pt[1]
    firstx = keypoints[kplocation].pt[0]
    firsty = keypoints[kplocation].pt[1]
    if math.sqrt(((secondx - firstx) ** 2) + (
            (secondy - firsty) ** 2)) > 100:
        return False
    return True


# def track(kparray, frametotrack):
#     targetframenum = frametotrack
#     targetgray = vidFrametoGray(targetframenum)
#     targetSift = cv2.SIFT_create()
#     targetKeypoints, targetDescriptors = targetSift.detectAndCompute(targetgray, None)
#     diff = np.linalg.norm(np.full(128, np.inf))
#
#     indexcount = 0
#     finalcount = 0
#
#     for i in kparray:
#         for j in targetDescriptors:
#             # first loop finds the smallest difference
#             if np.linalg.norm(j - i) < diff:
#                 diff = np.linalg.norm(j - i)
#     if diff > 300:
#         return False
#         # success = False
#         # return kplocation
#     for k['descriptor'] in kparray:
#         for h in targetDescriptors:
#             if np.linalg.norm(h - k) == diff:
#                 finalcount = indexcount
#                 break
#             else:
#                 indexcount += 1
#     if frametotrack != 0:
#         secondx = targetKeypoints[finalcount].pt[0]
#         secondy = targetKeypoints[finalcount].pt[1]
#         firstx = siftKeypoints(frametotrack - 1)[track(kparray, frametotrack - 1)].pt[0]
#         firsty = siftKeypoints(frametotrack - 1)[track(kparray, frametotrack - 1)].pt[1]
#
#         if math.sqrt(((secondx - firstx) ** 2) + (
#                 (secondy - firsty) ** 2)) > 100:
#             return False
#     else:
#         pass
#         # success = False
#         # return kplocation
#     image = cv2.circle(targetgray, (int(targetKeypoints[finalcount].pt[0]),
#                                     int(targetKeypoints[finalcount].pt[1])), 4, (255, 0, 0), 2)
#     plt.imshow(image)
#     plt.show()
#     if frametotrack != 0:
#         print("second point is", secondx, secondy, "first point is", firstx, firsty, "count is ", finalcount, "diff is "
#               , diff, "distance is ", math.sqrt(((secondx - firstx) ** 2) + ((secondy - firsty) ** 2)))
#     return finalcount
#
#
# def trackSuccess(kparray, frametotrack):
#     targetframenum = frametotrack
#     targetgray = vidFrametoGray(targetframenum)
#     targetSift = cv2.SIFT_create()
#     targetKeypoints, targetDescriptors = targetSift.detectAndCompute(targetgray, None)
#     diff = np.linalg.norm(np.full(128, np.inf))
#     indexcount = 0
#     finalcount = 0
#     for i in kparray:
#         for j in targetDescriptors:
#             if np.linalg.norm(j - i) < diff:
#                 diff = np.linalg.norm(j - i)
#     if diff > 300:
#         return False
#         # success = False
#         # return kplocation
#     for k in kparray:
#         for h in targetDescriptors:
#             if np.linalg.norm(h - k) == diff:
#                 finalcount = indexcount
#                 break
#             else:
#                 indexcount += 1
#     if frametotrack != 0:
#         secondx = targetKeypoints[finalcount].pt[0]
#         secondy = targetKeypoints[finalcount].pt[1]
#         firstx = siftKeypoints(frametotrack - 1)[track(kparray, frametotrack - 1)].pt[0]
#         firsty = siftKeypoints(frametotrack - 1)[track(kparray, frametotrack - 1)].pt[1]
#
#         if math.sqrt(((secondx - firstx) ** 2) + (
#                 (secondy - firsty) ** 2)) > 100:
#             return False
#     else:
#         return True

############################################################################
############################################################################
# CODE FOR RUNNING FUNCTIONS:


############################################################################
# WORKING CODE FOR TRACKING 1 FEATURE

# kplocation = 280
def ftrack(kplocation):
    #count = 1
    successindex = 0
    start = 0
    result = np.full(total_frames, np.inf)
    result[0] = kplocation
    if not trackNextFrameSuccess(0, 1, kplocation):
        for i in range(2, total_frames):
            if trackNextFrameSuccess(0, i, kplocation):
                start = trackNextFrame(0, i, kplocation)
                result[i] = start
                successindex = i
    else:
        start = trackNextFrame(0, 1, kplocation)
        result[1] = start
        successindex += 1

    while successindex != total_frames:
        if not trackNextFrameSuccess(successindex, successindex + 1, start):
            if successindex == total_frames - 1:
                break
            for j in range(successindex + 2, total_frames + 1):
                if j == total_frames - 1 and not trackNextFrameSuccess(successindex, j, start):
                    # print('j is', j)
                    #print(count)
                    return result
                if trackNextFrameSuccess(successindex, j, start):
                    start = trackNextFrame(successindex, j, start)
                    result[j] = start
                    successindex = j
                    #print(successindex)
                    #count += 1
        else:
            start = trackNextFrame(successindex, successindex + 1, start)
            result[successindex + 1] = start
            successindex += 1
            #print(successindex)
            #count += 1


def pveloplot(kplocation):
    framerate = cap.get(cv2.CAP_PROP_FPS)
    kparray = ftrack(kplocation)
    velarray = np.full(total_frames, np.inf)

    def yvalues():
        i = 0
        j = 1
        while i != kparray.size - 1:
            if kparray[i] == np.inf:
                i += 1
                continue
            # print('i is', i)
            while j != kparray.size:
                # print('j is', j)
                if j == kparray.size - 1 and kparray[j] == np.inf:
                    velarray[0] = 0
                    # print(velarray)
                    yvalues = []
                    for velovalue in velarray:
                        if velovalue != np.inf:  # and velovalue != 0:
                            yvalues.append(velovalue)
                    return yvalues
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
                    # print(distance / time)
                    i = j
                    j = i + 1
                    break


    yvalues = yvalues()
    xvalues = []
    # real is used to remove the points with velocities of 0.
    realyvalues = []
    realxvalues = []
    for i in range(0, velarray.size):
        if velarray[i] != np.inf:
            xvalues.append(i / framerate)
    for i in range(0, len(yvalues)):
        if yvalues[i] != 0:
            realyvalues.append(yvalues[i])
            realxvalues.append(xvalues[i])
    plt.plot(xvalues, yvalues, 'ro')
    plt.ylabel('Pixel Velocity (Pixels / Second)')
    plt.xlabel('Time (Seconds)')
    plt.title('Velocity-Time Graph of Bow')
    plt.show()
    # velarray[0] = 0
    # print(velarray)
    # yvalues = []
    # for velovalue in velarray:
    #     if velovalue != np.inf:
    #         yvalues.append(velovalue)
    # return yvalues

    #
    #
    #
    # framerate = cap.get(cv2.CAP_PROP_FPS)
    # kparray = ftrack(kplocation)
    # velarray = np.full(total_frames, np.inf)
    #
    # for i in range(0, kparray.size - 1):
    #     if kparray[i] == np.inf:
    #         continue
    #     #print('i is', i)
    #     for j in range(i + 1, kparray.size):
    #         #print('j is', j)
    #         if kparray[j] == np.inf:
    #             continue
    #         secondx = siftKeypoints(j)[int(kparray[j])].pt[0]
    #         secondy = siftKeypoints(j)[int(kparray[j])].pt[1]
    #         firstx = siftKeypoints(i)[int(kparray[i])].pt[0]
    #         firsty = siftKeypoints(i)[int(kparray[i])].pt[1]
    #         time = (j - i) / framerate
    #         distance = math.sqrt(((secondx - firstx) ** 2) + ((secondy - firsty) ** 2))
    #         velarray[j] = distance / time
    # velarray[0] = 0
    # yvalues = []
    # for velovalue in velarray:
    #     if velovalue != np.inf:
    #         yvalues.append(velovalue)
    # return yvalues


pveloplot(280)
# print(ftrack(280))


# kparray = ftrack(280)
# image = vidFrametoGray(0)
# #image = image = cv2.circle(targetgray, (int(secondKeypoints[finalcount].pt[0]), int(secondKeypoints[finalcount].pt[1])), 6, (255, 0, 0), 2)
# for i in range(0, kparray.size):
#     if kparray[i] == np.inf:
#         pass
#     image = cv2.circle(image, (siftKeypoints(i)[int(kparray[i])].pt[0], siftKeypoints(i)[int(kparray[i])].pt[0]), 6, (255, 0, 0), 2)
# plt.imshow(image, cmap = 'gray')
# plt.show()

############################################################################
# NON-FUNCTIONING CODE FOR TRACKING WITH MULTIPLE KEYPOINTS


# dtype = [('keypoint', cv2.KeyPoint), ('descriptor', 'f8', (1, 128))]
# kparray = np.array([(siftKeypoints(0)[280], siftDescriptors(0)[280]), (siftKeypoints(0)[279], siftDescriptors(0)[279])], dtype = dtype)
# print(kparray['keypoint'][0])
# print(siftDescriptors(0)[280])


# for i in range(0, total_frames):
#    track(kparray, i)


# successindex = 0
# start = 0
#
# if not trackSuccess(kparray, 0):
#     for i in range(2, total_frames):
#         if trackSuccess(0, i, 280):
#             start = trackNextFrame(0, i, 280)
#             successindex = i
# else:
#     start = trackNextFrame(0, 1, 280)
#     successindex += 1
#
# while successindex != total_frames:
#     if not trackNextFrameSuccess(successindex, successindex + 1, start):
#         for j in range(successindex + 2, total_frames):
#             if trackNextFrameSuccess(successindex, j, start):
#                 start = trackNextFrame(successindex, j, start)
#                 successindex = j
#                 print(successindex)
#     else:
#         start = trackNextFrame(successindex, successindex + 1, start)
#         successindex += 1


############################################################################
# CODE FOR PRINTING A KEYPOINT'S KEYPOINT ARRAY


# targetframenum = 3
# targetgray = vidFrametoGray(targetframenum)
# targetSift = cv2.SIFT_create()
# targetKeypoints, targetDescriptors = targetSift.detectAndCompute(targetgray, None)
# print((targetKeypoints[4]))


##################################################################################
# for i in range(6, 10):
#     showSiftFunction(i)
#     image = vidFrametoGray(i)
#     plt.imshow(image)
#     plt.show()

# Diff on 8th frame is 351.17
# The keypoint I track is all black, which may be why it moves to the hair.
# Need to choose a better keypoint to track

# Use distance to determine the tracking now since just measuring diff doesnt work
# showSpecficFeature(0, 280)
# showSiftFunction(0)
# showSiftFunction(100)
# showSpecficFeature(7, 300)
# showSpecficFeature(8, 225)

# Does not plot the 0th frame; starts with the 1st.

# start = trackNextFrame(0, 1, 280)
# print(start)


# successindex = 0
# start = 0
#
# if isinstance(trackNextFrame(0, 1, 280), int):
#     start = trackNextFrame(0, 1, 280)
#     successindex += 1
# else:
#     for i in range(2, total_frames):
#         if isinstance(trackNextFrame(0, i, 280), int):
#             start = trackNextFrame(0, i, 280)
#             successindex = i
#
# while successindex != total_frames:
#     if isinstance(trackNextFrame(successindex, successindex + 1, start), int):
#         start = trackNextFrame(successindex, successindex + 1, start)
#         successindex += 1
#     else:
#         for j in range(successindex + 2, total_frames):
#             if isinstance(trackNextFrame(successindex, j, start), int):
#                 start = trackNextFrame(successindex, j, start)
#                 successindex = j

# trackNextFrame(0, 2, 280)


# while trackNextFrame(0, 1, 280) == False:
#    trackNextFrame


# i = 1
# start = trackNextFrame(0, 1, 280)
# #for i in range(1, 1879):
# while i < 1879:  #8th frame starts to screw up   |   siftDescriptors(0).shape[0]
#    if not success:
#          start = trackNextFrame(i, i + 2, start)
#          i += 2
#     else:
#         start = trackNextFrame(i, i + 1, start)
#     #print('drawn')
#         i = i + 1
