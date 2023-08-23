import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import tasks

import render
import gfxmath
import cv2
import time
import numpy as np

facePose = None


def current_milli_time():
    return round(time.time() * 1000)

# this function will get called when mediapipe has run it's landmark detection on the output_image
def GetFacePoseResult(result: vision.FaceLandmarkerOptions, output_image: mp.Image, timestamp_ms: int):
    global facePose

    if len(result.facial_transformation_matrixes) > 0:
        facePose = result.facial_transformation_matrixes[0]

# initialize mediapipe for live stream video feed (e.g. webcam)
# and set it to return the face pose matrix
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
# puts overlay on top of image to make a ocmposite image, treating 'ignore_color' as transparent
def composite(image, overlay, ignore_color=[0,0,0]):
    ignore_color = np.full(overlay.shape, ignore_color)
    mask = ~(overlay==ignore_color).all(-1)
    out = image.copy()
    out[mask] = image[mask] * 0.5 + overlay[mask] * 0.5
    return out
    

def doAr():

    # initialize mediapipe
    faceLandmarker = InitMediapipe()

    # 3d renderer init
    scale = 0.5
    width = 1280 * scale
    height = 720 * scale
    render.SetupRenderer(width, height)
    render.SetupScene()

    # load a 3d model of a sunglass.
    sunglass_path = './objs/sunglass/model.obj'
    render.LoadObj(sunglass_path)
    sunglassScale = 8

    # also make a transformation matrix. This will move, rotate and scale the sunglass so that it
    # appears in the right place on the user's face, i.e. in front of the eyes. 
    sunglassTransform = gfxmath.makePose(translation=[-0.5,2,4], rotation=[10,180,0], scale=[sunglassScale, sunglassScale, sunglassScale])
    
    # start a camera and apply ar filter
    try: 
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        
        startTime = current_milli_time()
        while True:

            # get webcam image and resize it
            ret, image = cap.read()
            image = cv2.flip(image, 1)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

            # convert the image to mediapipe readable format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            timeStampMs = current_milli_time() - startTime
            # and submit it for pose detection
            faceLandmarker.detect_async(mp_image, timeStampMs)

            # we'll show the captured image 
            outImage = image
            # but first see if mediapipe gave us a face pose (i.e. what direction the user is facing)
            if facePose is not None:
                # yes it did, so let's draw a 3d sunglass
                objImg = render.DrawArObj(sunglass_path, facePose, sunglassTransform)
                # and combine it with the image from the webcam
                outImage = composite(image, objImg, [0,0,0])

            # and show
            cv2.imshow('Input', outImage)

            # did someone press escape?
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


