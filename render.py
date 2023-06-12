import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

previous_frame = None
current_frame = None

previous_frame_kp = None
current_frame_kp = None

previous_frame_dsc = None
current_frame_dsc = None


def get_match_pose(matches, prev, curr):
    list_prev = []
    list_curr = []

    for match in matches:
        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = prev[match.queryIdx].pt
        (x2, y2) = curr[match.trainIdx].pt

        # Append to each list
        # list_kp1.append((x1, y1))
        list_curr.append((x2, y2))

    return list_curr


def change_frame():
    global previous_frame, current_frame
    global previous_frame_kp, current_frame_kp
    global previous_frame_dsc, current_frame_dsc

    previous_frame = current_frame
    previous_frame_kp = current_frame_kp
    previous_frame_dsc = current_frame_dsc

def get_match_pixel():
    # print('pre cicle///')
    global previous_frame, current_frame
    global previous_frame_kp, current_frame_kp
    global previous_frame_dsc, current_frame_dsc
    orb = cv.ORB_create()
    current_frame_kp, current_frame_dsc = orb.detectAndCompute(current_frame,None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(previous_frame_dsc,current_frame_dsc)
    match = cv.drawMatches(previous_frame,
                            previous_frame_kp,
                            current_frame,
                            current_frame_kp,
                            matches,
                            None,
                            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # print('its trying')
    plt.imshow(match)
    plt.show(block=False)
    plt.pause(0.2)
    # plt.close()
    kpoints = get_match_pose(matches, previous_frame_kp, current_frame_kp)

    change_frame()
    return kpoints

    cv.destroyAllWindows()


def video_render():
    global current_frame, previous_frame
    cap = cv.VideoCapture('/home/lidia/Downloads/sfm/lune.mp4')
    
    for i in range(20):
        ret, frame = cap.read()  # Take each frame
        # plt.imshow(frame),plt.show()
        # current_frame = cv.imread(cv.samples.findFile(f"/home/lidia/Downloads/sfm/BOX/{index}.jpg"))
        current_frame = frame
        if i == 0:
            previous_frame = current_frame

        print(i)
        yield get_match_pixel()

def img_render():
    global current_frame, previous_frame
    for index in [2,3,4,5,6,7]:
        current_frame = cv.imread(cv.samples.findFile(f"/home/lidia/Downloads/sfm/BOX/{index}.jpg"))

        if index == 2:
            previous_frame = cv.imread(cv.samples.findFile(f"/home/lidia/Downloads/sfm/BOX/{1}.jpg"))

        # print(get_match_pixel())
        print(index)
        yield get_match_pixel()



