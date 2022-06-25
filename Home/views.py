from django.shortcuts import render
from Home.models import Videos
from VideoColorizer.settings import MEDIA_ROOT

import io
import eel
import os
import numpy as np
import cv2
import subprocess
from moviepy.editor import *

import cv2
from posixpath import splitext
import numpy as np
import os
from os.path import isfile, join
import shutil


def index(request):
    if request.method == "POST":
        video_obj = Videos()
        video_file = request.FILES['video_file']
        video_obj.video = video_file
        video_obj.save()
        video_name_ext = video_obj.video.name
        video_path = f"{MEDIA_ROOT}{video_name_ext}"
        video_name = video_name_ext.split(".")[0]

        # .............................................................................
        os.mkdir(f'{MEDIA_ROOT}temp')
        vidcap = cv2.VideoCapture(f'{MEDIA_ROOT}{video_name_ext}')
        success,image = vidcap.read()
        count = 0
        while success:
            net = cv2.dnn.readNetFromCaffe(f'{MEDIA_ROOT}model/colorization_deploy_v2.prototxt', f'{MEDIA_ROOT}model/colorization_release_v2.caffemodel')
            pts = np.load(f'{MEDIA_ROOT}model/pts_in_hull.npy')
            class8 = net.getLayerId("class8_ab")
            conv8 = net.getLayerId("conv8_313_rh")
            pts = pts.transpose().reshape(2, 313, 1, 1)
            net.getLayer(class8).blobs = [pts.astype("float32")]
            net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
            scaled = image.astype("float32") / 255.0
            lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
            resized = cv2.resize(lab, (224, 224))
            L = cv2.split(resized)[0]
            L -= 50
            net.setInput(cv2.dnn.blobFromImage(L))
            ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
            ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
            L = cv2.split(lab)[0]
            colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
            colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
            colorized = np.clip(colorized, 0, 1)
            colorized = (255 * colorized).astype("uint8")
            # cv2.imwrite(f"output_{fname}", colorized)
            
            cv2.imwrite(f"{MEDIA_ROOT}temp/frame%d.jpg" % count, colorized)     # save frame as JPEG file      
            success,image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1
        pathIn= f'{MEDIA_ROOT}temp/'
        # pathOut = 'video.avi'
        pathOut = f'{MEDIA_ROOT}temp/output.mp4'
        fps = 30.0

        frame_array = []
        files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

        #for sorting the file names properly
        files.sort(key = lambda x: int(x[5:-4]))

        for i in range(len(files)):
            filename=pathIn + files[i]
            #reading each files
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            print(filename)
            #inserting the frames into an image array
            frame_array.append(img)

        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

        for i in range(len(frame_array)):
            # writing to a image array
            out.write(frame_array[i])
        out.release()
        import moviepy.editor as mp
        my_clip = mp.VideoFileClip(f"{MEDIA_ROOT}{video_name_ext}")
        my_clip.audio.write_audiofile(f"{MEDIA_ROOT}temp/audio.mp3")
        audio_clip = mp.AudioFileClip(f"{MEDIA_ROOT}temp/audio.mp3")
        coloured_clip = mp.VideoFileClip(f"{MEDIA_ROOT}temp/output.mp4")
        final_clip = coloured_clip.set_audio(audio_clip)
        final_clip.write_videofile(f"{MEDIA_ROOT}final_{video_name}.mp4")
        shutil.rmtree(f'{MEDIA_ROOT}temp')
        final_video_ext = f"final_{video_name}.mp4"

        # .............................................................................

        return render(request,'home/index.html',{'video_name_ext': video_name_ext, 'final_video_ext': final_video_ext})

    return render(request,'home/index.html')
