from sklearn.cluster import KMeans

import argparse
from cv2 import cv2
import numpy as np
import os
import time
import math
import statistics 
import colorsys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True, help="Path to the Video")
    ap.add_argument("-c", "--clusters", required=True, type=int, help="# of clusters", default=3)
    ap.add_argument("-o", "--output", help="Output folder", default="Outputo")
    ap.add_argument("-m", "--mls", help="Offset Miliseconds", type=int, default=0)
    ap.add_argument("-s", "--sec", help="Output time", type=int, default=-1)
    args = vars(ap.parse_args())
    directory = args["output"]

    if not os.path.exists(directory):
        os.makedirs(directory)

    vidcap = cv2.VideoCapture(args["video"])
    success, image = vidcap.read()
    count = 0
    success = True

    # for x in range(1000):
    #     success, image = vidcap.read()

    width = image.shape[1]
    add_width = int(width/4)
    height = image.shape[0]
    add_height = width - height # int(height/2)

    if add_height == 0:
        print("Video is Square")
        return
    cluster_size = args["clusters"]
    x_offset = int(height/cluster_size/8)
    y_offset = int(width/cluster_size/8)

    milis = args["mls"]

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print("Fps : {0}".format(fps))

    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)
    if(milis > 0):
        frame_count = math.ceil(frame_count / (milis/1000*fps))
    frame_calc_time = time.time()

    output_time = args["sec"]

    new_video_fps = fps if output_time < 0 else frame_count / output_time
    
    writer = cv2.VideoWriter(directory + "/video.avi", cv2.VideoWriter_fourcc(*"MJPG"), new_video_fps, (width, height+add_height))
    print("Wait")

    frame_calc_list = []
    hsv_list = []
    # for x in range(10):
    frame_step = 1
    while success:
        if(milis > 0):
            vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * milis))

        frame_calc_time = time.time()

        
        img = np.zeros((add_height,width, 3), np.uint8)
        img.fill(30)

        count = count + 1

        _cury = 0
        _curx = 0
        _addy = int(height / cluster_size)
        _addx = int(width / cluster_size)

        # if len(hsv_list)==0:
        if(count % frame_step == 0 or len(hsv_list)==0):

            k_image = image.reshape((width * height, 3))
            clt = KMeans(n_clusters=int(math.ceil(cluster_size)), max_iter=1,random_state=0,n_init=1)
            clt.fit(k_image)

            # print(clt.cluster_centers_)
            
            new_colors = [c for i, c in enumerate(clt.cluster_centers_)]
            new_colors.sort(key=lambda x: step(x)[1])

            hsv_list = new_colors
            # if(len(hsv_list) == 0):
            #     hsv_list = new_colors
            # else:
            #     hsv_list = orderList(hsv_list, new_colors)
        # # else:

        
        # else:
        #     for i in range(cluster_size):
        #         hsv_list[i] = [(hsv_list[i][j]+new_colors[i][j])/2 for j in range(3)]


        # hsv_list.sort(key=lambda x: max(x), reverse=True)
        # hsv_list.sort(key=lambda x: rgb_to_hsv(x)[0]+rgb_to_hsv(x)[2])
        # hsv_list.sort(key=lambda x: rgb_to_hsv(x[0])[2], reverse=True)
        # hsv_list.sort(key=lambda x: step(x[2],x[1],x[0],4)[1], reverse=True)
        # hsv_list.sort(key=lambda x: lum(x[2],x[1],x[0]))

        # print("HSV ")
        # print(hsv_list)


        for i in range(cluster_size):
            color = hsv_list[i]
            # color = sorto[i]
            # cv2.rectangle(img, (x_offset, y_offset + _cury), (add_width - x_offset, _cury + _addy - y_offset), (color[0], color[1], color[2]), -1)
            # _cury = _cury + _addy
            cv2.rectangle(img, (x_offset + _curx, y_offset ), (_curx + _addx - x_offset, add_height - y_offset), (color[0], color[1], color[2]), -1)
            _curx = _curx + _addx
            #/ (1 if i % (cluster_size-1) == 0 else 2)

        out_image = np.concatenate((image, img), axis=0)
        
        # out_image = image
        cv2.imwrite("%s/frame%d.jpg" % (directory, count), out_image)
        writer.write(out_image)

        frame_calc_time = time.time() - frame_calc_time
        frame_calc_list.append(frame_calc_time)
        remaining_count = frame_count - count
        print('Remaining Frames: ', remaining_count)
        print('Remaining Time: %d seconds' % (remaining_count*statistics.median(frame_calc_list)))

        success, image = vidcap.read()
        # if(count > 5):
        #     break

    writer.release()

def orderList(a,b):
    ret = []
    old = a
    new = b
    for i in range(len(old)):
        b_val = min(new, key=lambda x:abs(step(x)[1]-step(old[i])[1]))
        ret.append(b_val)
        for _j in range(len(new)):
            j = new[_j]
            if(j[0] ==b_val[0] and j[1] == b_val[1] and j[2] == b_val[2]):
            # if(j == b_val):
                new.pop(_j)
                break
    return ret

def lum (rgb):
    r = rgb[2]
    g = rgb[1]
    b = rgb[0]
    return math.sqrt( .241 * r + .691 * g + .068 * b)

def step(rgb, repetitions=1):
    r = rgb[2]
    g = rgb[1]
    b = rgb[0]
    
    val =  .241 * r + .691 * g + .068 * b
    if(val < 0):
        return (0,0,0)
    lum = math.sqrt(val)

    h, s, v = colorsys.rgb_to_hsv(r,g,b)

    h2 = int(h * repetitions)
    lum2 = int(lum * repetitions)
    v2 = int(v * repetitions)

    # if h2 % 2 == 1:
    #     v2 = repetitions - v2
    #     lum = repetitions - lum

    return (h2, lum, v2)

def rgb_to_hsv(rgb):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]

    r = float(r)
    g = float(g)
    b = float(b)
    high = max(r, g, b)
    low = min(r, g, b)
    h, s, v = high, high, high

    d = high - low
    s = 0 if high == 0 else d/high

    if high == low:
        h = 0.0
    else:
        h = {
            r: (g - b) / d + (6 if g < b else 0),
            g: (b - r) / d + 2,
            b: (r - g) / d + 4,
        }[high]
        h /= 6

    return h, s, v

if __name__ == '__main__':
    main()
