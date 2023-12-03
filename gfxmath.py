import numpy as np
import cv2
import math

'''
make and return a 4x4 3d transformation matrix (also known as a pose)
'''
def makePose(translation =[0,0,0], rotation = [0,0,0], scale=[1,1,1]):
    # Translation matrix to place the object at the origin
    translation_matrix = np.array([
        [1, 0, 0, translation[0]],
        [0, 1, 0, translation[1]],
        [0, 0, 1, translation[2]],
        [0, 0, 0, 1]
    ])
    
    rotation = [i * (np.pi/180.0) for i in rotation]
    rot_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rotation[0]), -np.sin(rotation[0]), 0],
        [0, np.sin(rotation[0]), np.cos(rotation[0]), 0],
        [0, 0, 0, 1]
    ])

    rot_y = np.array([
        [np.cos(rotation[1]), 0, np.sin(rotation[1]), 0],
        [0, 1, 0, 0],
        [-np.sin(rotation[1]), 0, np.cos(rotation[1]), 0],
        [0, 0, 0, 1]
    ])

    rot_z = np.array([
        [np.cos(rotation[2]), -np.sin(rotation[2]), 0, 0],
        [np.sin(rotation[2]), np.cos(rotation[2]), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    rotation_matrix = np.dot(np.dot(rot_x, rot_y), rot_z)

    scaling_matrix = np.array([
        [scale[0], 0, 0, 0],
        [0, scale[1], 0, 0],
        [0, 0, scale[2], 0],
        [0, 0, 0, 1]
    ])

    transformation_matrix = np.dot(translation_matrix, np.dot(rotation_matrix, scaling_matrix))
    return transformation_matrix

def GetPsudoCamera(frame_width, frame_height):
    # pseudo camera internals
    focal_length = frame_width
    center = (frame_width/2, frame_height/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
    distortion = np.zeros((4, 1))
    return camera_matrix, distortion

def VecDist(vecA, vecB):
    return math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(vecA, vecB)))

#
def GetWorldPoints(model_points, image_points, camera_matrix, distortion):
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, distortion, flags=cv2.SOLVEPNP_SQPNP)
    transformation = np.eye(4)  # needs to 4x4 because you have to use homogeneous coordinates
    transformation[0:3, 3] = translation_vector.squeeze()
    # the transformation consists only of the translation, because the rotation is accounted for in the model coordinates. Take a look at this (https://codepen.io/mediapipe/pen/RwGWYJw to see how the model coordinates behave - the hand rotates, but doesn't translate
    # transform model coordinates into homogeneous coordinates
        
    model_points_hom = np.concatenate((model_points, np.ones((21, 1))), axis=1)
    # apply the transformation
    world_points = model_points_hom.dot(np.linalg.inv(transformation).T)
    
    return world_points