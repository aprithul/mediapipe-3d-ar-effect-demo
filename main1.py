import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import tasks
import cv2
import time
from landmarkDrawingUtil import draw_landmarks_on_image

_processedImage = None

def current_milli_time():
    return round(time.time() * 1000)


# this function will get called when mediapipe has run it's landmark detection on the output_image
def GetResult(result: vision.FaceLandmarkerOptions, output_image: mp.Image, timestamp_ms: int):
    global _processedImage
    #print('face landmarker result: {}'.format(result))
    _processedImage = draw_landmarks_on_image(output_image.numpy_view(), result)


# initialize mediapipe for live stream video feed (e.g. webcam)
# and set it to return the face pose matrix
def InitMediapipe():
    model_path = "./Models/face_landmarker_v2_with_blendshapes.task"
    options = vision.FaceLandmarkerOptions(
        base_options=tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.LIVE_STREAM,
        output_facial_transformation_matrixes = True,
        result_callback=GetResult)
    return vision.FaceLandmarker.create_from_options(options)


def main():

    # initialize mediapipe
    faceLandmarker = InitMediapipe()

    # start a camera and apply ar filter
    try: 
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        startTime = current_milli_time()
        while True:
            # get webcam image
            ret, image = cap.read()
            image = cv2.flip(image, 1)
            
            # convert the image to mediapipe readable format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            timeStampMs = current_milli_time() - startTime
            
            # and submit it for landmark detection
            faceLandmarker.detect_async(mp_image, timeStampMs)
            
            # do we have a _processedImage returned by mediapipe?
            # this image will have the landmarks drawn on the face
            if _processedImage is not None:
                # then show it
                cv2.imshow('Input', _processedImage)
            
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
    main()
