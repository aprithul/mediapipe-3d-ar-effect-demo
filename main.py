import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import tasks
from mediapipe.framework.formats import landmark_pb2

import render
import gfxmath
import cv2
import time
import numpy as np
from Entities import Entity
from enum import Enum

class State(Enum):
    ON_HEAD = 0
    IN_HAND = 1
    FREE = 2


MinPinchDist = 6
MinSnapDist = 10

facePose = None
indexPos = None
thumbPos = None
matPsudoCam = None
distortionPsudoCam = None
width = None
height = None
state = State.FREE
hasHand = False


'''
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
'''



def current_milli_time():
    return round(time.time() * 1000)

# this function will get called when mediapipe has run it's landmark detection on the output_image
def GetFacePoseResult(result: vision.FaceLandmarkerOptions, output_image: mp.Image, timestamp_ms: int):
    global facePose

    if len(result.facial_transformation_matrixes) > 0:
        facePose = result.facial_transformation_matrixes[0]

def GetHandLandMarkResult(result:vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global indexPos
    global thumbPos
    global hasHand

    if len(result.handedness) > 0 and result.handedness[0][0].category_name == "Right":
        world_landmarks = result.hand_world_landmarks[0]
        hand_landmarks = result.hand_landmarks[0]
        
        #print(hand_landmarks)
        model_points = np.float32([[-l.x, -l.y, -l.z] for l in world_landmarks])
        image_points = np.float32([[l.x * width, l.y * height] for l in hand_landmarks])

        #print(type(model_points), type(image_points))
        world_points = gfxmath.GetWorldPoints(model_points, image_points, matPsudoCam, distortionPsudoCam)
        indexPos = world_points[8]
        thumbPos = world_points[4]
        
        indexPos[0] *= -100
        indexPos[1] *= 100
        indexPos[2] *= 50

        thumbPos[0] *= -100
        thumbPos[1] *= 100
        thumbPos[2] *= 50
        hasHand = True
        #print(world_points[8])

        #print(len(result.handedness), ' ', len(result.hand_landmarks[0]))

# initialize mediapipe for live stream video feed (e.g. webcam)
# and set it to return the face pose matrix
def GetFacialLandmarker():
    face_model_path = "./Models/face_landmarker_v2_with_blendshapes.task"

    options = vision.FaceLandmarkerOptions(
        base_options=tasks.BaseOptions(model_asset_path=face_model_path),
        running_mode=vision.RunningMode.LIVE_STREAM,
        output_facial_transformation_matrixes = True,
        result_callback=GetFacePoseResult)
    return vision.FaceLandmarker.create_from_options(options)
    #return options

def GetHandLandmarker():
    # STEP 2: Create an HandLandmarker object.
    hand_model_path = "./Models/hand_landmarker.task"
    base_options = tasks.BaseOptions(model_asset_path=hand_model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        running_mode=vision.RunningMode.LIVE_STREAM,
                                        result_callback=GetHandLandMarkResult)
    return vision.HandLandmarker.create_from_options(options)

# https://stackoverflow.com/questions/57104921/cv2-addweighted-except-some-color
# puts overlay on top of image to make a ocmposite image, treating 'ignore_color' as transparent
def composite(image, overlay, ignore_color=[0,0,0]):
    ignore_color = np.full(overlay.shape, ignore_color)
    mask = ~(overlay==ignore_color).all(-1)
    out = image.copy()
    out[mask] = image[mask] * 0.5 + overlay[mask] * 0.5
    return out
    
def isPinching():
    if hasHand is False:
        print("Error")
        return False, None
    
    dist = gfxmath.VecDist(indexPos, thumbPos)
    pos = [(px + qx)/ 2.0 for px, qx in zip(indexPos, thumbPos)]
    print(current_milli_time(), state,dist)
    if  dist < MinPinchDist:
        return True, pos
    return False, pos

def doAr():

    global matPsudoCam
    global distortionPsudoCam
    global width
    global height
    global indexPos
    global thumbPos
    global state

    # initialize mediapipe
    faceLandmarker = GetFacialLandmarker()
    handLandmarker = GetHandLandmarker()

    # 3d renderer init
    scale = 0.5
    width = 1280 * scale
    height = 720 * scale
    render.SetupRenderer(width, height)
    render.SetupScene()

    # calculate camera matrices using camera intrinsics
    matPsudoCam, distortionPsudoCam = gfxmath.GetPsudoCamera(width, height)        

    # load a 3d model of a sunglass.
    sunglassOffset = gfxmath.makePose(translation=[-0.5,2,4], rotation=[0,180,0], scale=[8, 8, 8])
    sunglassEnt =  Entity('./objs/sunglass/model.obj', sunglassOffset)
    sunglassEnt.SetTransform(np.dot(gfxmath.makePose(translation=[20,0,-35], rotation=[0,180,0], scale=[1,1,1]), sunglassOffset))
    render.LoadObj(sunglassEnt)

    headOffset = gfxmath.makePose(scale=[25,25,25])
    headEnt = Entity('./objs/head/head.obj', headOffset)
    render.LoadObj(headEnt)
    '''
    LoadObj(_headPath)
    headScale = 25
    _headPose = gfxmath.makePose(translation=[0,0,0], rotation=[0,0,0], scale=[headScale, headScale, headScale])
    '''

    # start a camera and apply ar filter
    try: 
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        
        startTime = current_milli_time()
        while True:
            
            '''Get image and do mediapipe detection'''
            # get webcam image and resize it
            ret, image = cap.read()
            image = cv2.flip(image, 1)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

            # convert the image to mediapipe readable format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            timeStampMs = current_milli_time() - startTime
            # and submit it for pose detection
            faceLandmarker.detect_async(mp_image, timeStampMs)
            handLandmarker.detect_async(mp_image, timeStampMs)
            ''''''
            '''depending on state run interaction logic'''
            if state is State.FREE:
                res, pos = isPinching()
                if res is True:
                    state = State.IN_HAND
            elif state is State.IN_HAND:
                res, pos = isPinching()
                if res is True:
                    pinchPosTransform = gfxmath.makePose(pos)
                    sunglassEnt.SetTransform(np.dot(pinchPosTransform, sunglassOffset))
                else:
                    if facePose is not None:
                        headPos = (facePose[0,3], facePose[1,3], facePose[2,3])
                        distToHead = gfxmath.VecDist(sunglassEnt.GetPos(), headPos)
                        if  distToHead <= MinSnapDist:
                            state = State.ON_HEAD
                        else:
                            state = State.FREE
                    else:
                        state = State.FREE
            elif state is State.ON_HEAD:
                headEnt.SetTransform(np.dot(facePose, headOffset))
                render.DrawDepth(headEnt)
                
                sunglassEnt.SetTransform(np.dot(facePose, sunglassOffset))
                res, pos = isPinching()
                if res is True:
                    state = State.IN_HAND


            #render.Draw(sunglassEnt)

            # we'll show the captured image 
            outImage = composite(image, render.colorBuffer, [0,0,0])

            # and show
            outImage = cv2.resize(outImage, (1024, 576))
            cv2.imshow('Input', render.depthBuffer/255)
            render.Clear()

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


