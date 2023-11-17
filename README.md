# 3D AR effects with Google Mediapipe in Python
![python-mediapipe-ar](https://github.com/aprithul/mediapipe-3d-ar-effect-demo/assets/8151229/c2076166-d8b6-43c7-9ef2-0bcc758f7538)

### Description
This is a Python application that demonstrates the use of Google's Mediapipe to create 3D AR effects. Mediapipe solution for Python doesn't have the capability to easily render 3D AR effects. The basic landmark detection works in the screen space cooridnate system which isn't ideal for 3D effects. However, it's also possible to get the face from the landmark detection results. This application does the following:
1. Gets the face pose from the webcam footage using Mediapipe
2. Sets up a virtual scene:
   1. Sets up a virtual camera from the pov of the physical camera
   2. Renders the 3D AR effect (e.g. a virtual sun glass) with this face pose.
   3. Renders a canonical face model to the depth buffer. This depth data is then used to mask out parts of the 3D AR effect that should be occluded by the head.
3. Composites the webcam image with the 3D effect image to come up with the final image
   
### Dependencies:
Install the following Python libraries first before trying to run the app:
1. Python 3
2. opencv-python
3. mediapipe -> for face landmark detection
4. pyrender -> for rendering 3d AR effect
5. numpy

### How to run:
run the main.py script:
`python3 main.py`


### Limitations
1. The virtual camera may need to be tweaked to match with your webcam to get good results
2. Not the most optimized application at the moment
