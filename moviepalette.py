from sklearn.cluster import KMeans

import argparse
from cv2 import cv2
import numpy as np
import os
import time
import math
import statistics 
import colorsys

# import ffmpeg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True, help="Path to the Video")
    ap.add_argument("-c", "--clusters", type=int, help="# of clusters", default=5)
    ap.add_argument("-o", "--output", required=True, help="Output folder", )
    ap.add_argument("-k", "--skip", help="Skip frame in Input video", type=int, default=0)
    ap.add_argument("-f", "--step", help="Calculation Step", type=int, default=1)
    ap.add_argument("-s", "--sec", help="Output time", type=int, default=-1)
    ap.add_argument("-a", "--alpha", help="Background alpha", type=int, default=30,choices=range(0, 256))
    ap.add_argument("-q", "--quality", help="Calculation quality (0,1)", type=float, default=0.5)
    ap.add_argument("-cl", "--cliplength", help="Length of Clips", type=float, default=0)
    ap.add_argument("-cc", "--clipcount", help="How many clips do you want?", type=int, default=1)
    ap.add_argument("-st", "--start", help="Start timestamp", type=int, default=0)
    ap.add_argument("-e", "--end", help="End timestamp", type=int, default=0)

    args = vars(ap.parse_args())
    directory = args["output"]

    if not os.path.exists(directory):
        os.makedirs(directory)

    vidcap = cv2.VideoCapture(args["video"])
    success, image = vidcap.read()
    count = 0
    success = True


    width = image.shape[1]
    height = image.shape[0]
    add_width = height-width
    add_height = width - height 
    cluster_size = args["clusters"]
    x_offset = int(height/cluster_size/8)
    y_offset = int(width/cluster_size/8)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print("Fps : {0}".format(fps))


    original_frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = original_frame_count
    print(frame_count)

    start_skip_frame = args["start"] * fps
    end_skip_frame = args["end"] * fps
    frame_count = frame_count - start_skip_frame - end_skip_frame

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_skip_frame)


    frame_calc_time = time.time()

    output_time = args["sec"]



    frame_calc_list = []
    hsv_list = []
    frame_colors = []
    # for x in range(10):
    frame_step = args["step"]
    quality = args["quality"]
    max_iter = int(300*quality)
    if max_iter <=0:
        max_iter = 1
    n_init = int(10*quality)
    if n_init <=0:
        n_init = 1

    c_len = args["cliplength"]
    c_count = args["clipcount"]
    c_frame_count = c_len * fps
    c_frame_count = frame_count if c_frame_count <= 0 and not c_frame_count > frame_count else c_frame_count
    c_frame_skip_count = math.floor((frame_count - c_count * c_frame_count) / (c_count))
    print("C_FRAME")
    print(c_frame_count)
    print(c_frame_skip_count)
    # vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # return
    clip_writer = cv2.VideoWriter(directory + "/clip.mp4", cv2.VideoWriter_fourcc(*"MP4V"), fps, (width, height))

    clip_count = 0
    move = start_skip_frame
    while True:
        success, image = vidcap.read()

        if not success or move + count > original_frame_count-end_skip_frame:
            break
        
        # if mode == 0:
        if count < c_frame_count:
            clip_writer.write(image)
            count = count + 1
            print(count)
        else:
            # mode = 1
            clip_count = clip_count + 1
            count = 0
            move = math.ceil((clip_count * c_frame_count + (clip_count) * c_frame_skip_count) + start_skip_frame)
            print("MOVE")
            print(move)
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, move)

    # else:
    #     if count < c_frame_skip_count:
    #         count = count + 1
    #     else:
    #         mode = 0
    #         count = 0

    clip_writer.release()
    # return

    vidcap = cv2.VideoCapture(directory + "/clip.mp4")
    success, image = vidcap.read()
    count = 0
    success = True
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
   
    
    while success:
        frame_calc_time = time.time()

        # if(milis > 0):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, (count * frame_step))

        k_image = image.reshape((width * height, 3))
        clt = KMeans(n_clusters=int(math.ceil(cluster_size)), max_iter=max_iter,random_state=0,n_init=n_init)
        clt.fit(k_image)

        new_colors = [c for i, c in enumerate(clt.cluster_centers_)]
        new_colors.sort(key=lambda x: lum(x))
        frame_colors.append((count*frame_step, new_colors))

       
        count = count + 1

        success, image = vidcap.read()
         
        frame_calc_time = time.time() - frame_calc_time
        frame_calc_list.append(frame_calc_time)
        remaining_count = int(frame_count/frame_step) - count
        print('Remaining Frames: ', remaining_count)
        print('Remaining Time: %d seconds' % (remaining_count*statistics.median(frame_calc_list)))


    vidcap.set(cv2.CAP_PROP_POS_MSEC, 0)
    success, image = vidcap.read()

    print("DONE CALCULATING")
    
    new_video_fps = fps if output_time < 0 else frame_count / output_time
    writer = cv2.VideoWriter(directory + "/video.mp4", cv2.VideoWriter_fourcc(*"MP4V"), new_video_fps, (width, (height+add_height)))
    count = 0
    frame_calc_list = []

    from_colors = frame_colors.pop(0)[1]

    to_frame = frame_colors.pop(0)
    to_colors = to_frame[1]
    to_frame_index = to_frame[0]

    skip_frame = args['skip']
    
    while success:
        frame_calc_time = time.time()

        count = count + 1
      
        if count == to_frame_index:
            # print('next')
            # print(count)
            from_colors = to_colors
            if(len(frame_colors)>0):
                to_frame = frame_colors.pop(0)
                to_frame_index = to_frame[0]
                to_colors = to_frame[1]
            else:
                break

            frame_calc_time = time.time() - frame_calc_time
            frame_calc_list.append(frame_calc_time)
            remaining_count = int(frame_count/(skip_frame+1)) - count
            print('Remaining Frames: ', remaining_count)
            print('Remaining Time: %d seconds' % (remaining_count*statistics.median(frame_calc_list)))


        img = np.zeros((add_height,width, 3), np.uint8)
        img.fill(args['alpha'])


        _cury = 0
        _curx = 0
        _addy = int(height / cluster_size)
        _addx = int(width / cluster_size)

    
        _step = ((count % frame_step) /frame_step)

        for i in range(cluster_size):
            color = colorLerp(from_colors[i],to_colors[i],_step)
            # cv2.rectangle(img, (x_offset, y_offset + _cury), (add_width - x_offset, _cury + _addy - y_offset), color, -1)
            # _cury = _cury + _addy
            cv2.rectangle(img, (x_offset + _curx, y_offset ), (_curx + _addx - x_offset, add_height - y_offset), color, -1)
            _curx = _curx + _addx

        out_image = np.concatenate((image, img), axis=0)
        
        # out_image = image
        cv2.imwrite("%s/frame%d.jpg" % (directory, count), out_image)
        writer.write(out_image)

        

        success, _im = vidcap.read()
        if count % (skip_frame+1) == 0:
            image = _im

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

def lerp(a, b, f):
    return a + f * (b - a);

def colorLerp(cfrom, cto, step):
    return [lerp(cfrom[i],cto[i],step) for i in range(3)]

if __name__ == '__main__':
    main()
