import numpy as np

'''
make and return a 4x4 3d transformation matrix (also known as a pose)
'''
def makePose(translation, rotation, scale):
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
