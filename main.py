import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import tasks

import render
import gfxmath
import cv2
import time
import numpy as np


camImage = None
facePose = None


def current_milli_time():
    return round(time.time() * 1000)

def GetFacePoseResult(result: vision.FaceLandmarkerOptions, output_image: mp.Image, timestamp_ms: int):
    global facePose

    if len(result.facial_transformation_matrixes) > 0:
        facePose = result.facial_transformation_matrixes[0]


def InitMediapipe():
    model_path = "./Models/face_landmarker_v2_with_blendshapes.task"
    options = vision.FaceLandmarkerOptions(
        base_options=tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.LIVE_STREAM,
        output_facial_transformation_matrixes = True,
        result_callback=GetFacePoseResult)
    return vision.FaceLandmarker.create_from_options(options)
    #return options

# https://stackoverflow.com/questions/57104921/cv2-addweighted-except-some-color
def composite(image, overlay, ignore_color=[0,0,0]):
    ignore_color = np.full(overlay.shape, ignore_color)
    mask = ~(overlay==ignore_color).all(-1)
    out = image.copy()
    out[mask] = image[mask] * 0.5 + overlay[mask] * 0.5
    return out
    

def doAr():
    global camImage
    global scene
    global sceneObjNodes

    faceLandmarker = InitMediapipe()

    # 3d renderer init
    scale = 0.5
    width = 1280 * scale
    height = 720 * scale
    render.SetupRenderer(width, height)
    render.SetupScene()

    # ar effect mesh loading and configuration
    sunglass_path = './objs/sunglass/model.obj'
    render.LoadObj(sunglass_path)
    sunglassScale = 8
    sunglassTransform = gfxmath.makePose(translation=[-0.5,2,4], rotation=[10,180,0], scale=[sunglassScale, sunglassScale, sunglassScale])
    
    # start a camera and apply ar filter
    try: 
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        startTime = current_milli_time()
        while True:
            ret, image = cap.read()
            image = cv2.flip(image, 1)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            timeStampMs = current_milli_time() - startTime
            faceLandmarker.detect_async(mp_image, timeStampMs)

            outImage = image
            if facePose is not None:
                objImg = render.DrawArObj(sunglass_path, facePose, sunglassTransform)
                outImage = composite(image, objImg, [0,0,0])

            cv2.imshow('Input', outImage)

            c = cv2.waitKey(1)
            if c == 27:
                break
    finally:
        print("release camera")
        if cap is not None and cap.isOpened(): 
            cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    doAr()


