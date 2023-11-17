import pyrender
import trimesh
import pyrender.constants as RenderFlags
import gfxmath
import numpy as np
from Entities import Entity

_sceneObjNodes = {}
_headPath = './objs/head/head.obj'
_scene = None
_camera = None
_headPose = None
_renderer = None

depthBuffer = None
colorBuffer = None


'''
Create a renderer. width and height should match that of the physical camera feed.
'''
def SetupRenderer(width, height):
    global _renderer
    global colorBuffer
    global depthBuffer

    _renderer = pyrender.OffscreenRenderer(width, height)
    colorBuffer = np.ndarray(shape=(int(height), int(width), 3), dtype=float)
    depthBuffer = np.ndarray(shape=(int(height), int(width)), dtype=float)
    print("Color buffer: ",colorBuffer.shape)
    print("Depth buffer: ",depthBuffer.shape)


'''
sets up scene with a camera facing negative z direction and ambient light.
also loads up the canonical head mesh used for masking out ar effect 3d meshes behind the head
'''
def SetupScene():
    global _scene
    global _camera
    global _headPose

    _scene = pyrender.Scene()
    _scene.ambient_light = [0.3,0.3,0.3, 1.0]
    _camera = pyrender.PerspectiveCamera(yfov=np.pi/3, aspectRatio=1.778, znear=0.05, zfar=50)
    camera_pose = gfxmath.makePose([0.0,0.0,0.0], [0.0,0.0,0.0], [1,1,1])
    _scene.add(_camera, pose=camera_pose)
    
    return _scene, _camera

'''
load the obj file at objPath (with material) as a node and save a reference to it.
this doesn't add the node to the scene.
'''
def LoadObj(entity:Entity):
    global _sceneObjNodes

    obj = trimesh.load(entity.path, force='mesh')
    objMesh = pyrender.Mesh.from_trimesh(obj)
    objNode = pyrender.Node(mesh=objMesh, scale=[1, 1, 1])
    _sceneObjNodes[entity.path] = objNode


'''
masks the colorImg by depthM. The result is the colorImg with depth test applied.
'''
def maskWithDepth(depthHead, colorObj, depthObj):
    # set all depth values close to 0 to 50 (we set 50 as the camera zfar distance)
    # this is because pyrender sets 0 to undefined depths (where there no object to render)
    # instead we set it to the farthest distance seen by the camera
    depthObj[depthObj < 0.05 ] = 50
    depthHead[depthHead < 0.05 ] = 50

    # create mask from the two depth buffers. 
    # mask will have 0 where head has a higher depth than the objs depth (i.e. obj is behind head)
    mask = np.maximum(depthHead - depthObj, 0)

    # set 1 to mask where value is greater than 0 (i.e. head is behind obj)
    mask[mask > 0] = 1
    mask = mask.astype('uint8')
    mask = np.expand_dims(mask, axis=-1)

    # use the mask
    return colorObj * mask


'''
helper function to add the node, render it with the render 'flags' and removing the node. 
This lets us draw one image at a time to a offscreen buffer with that drawcall specific render flags.
'''
def addAndDrawRemoveNode(node, transform, flags):
    global _scene

    _scene.add_node(node)
    _scene.set_pose(node, transform) 
    renderedImgs = _renderer.render(_scene, flags)
    _scene.remove_node(node)
    return renderedImgs

'''
draws the the already loaded obj from objpath on the user head, 
maintaining depth buffer as created by a canonical head model
def DrawArObjOnHead(objPath, facePose, transform = np.identity(4)):
    # retrieve the head model's node
    headNode = _sceneObjNodes[_headPath]
    # add the head to the scene, draw it to a depth buffer, and remove it
    #depthBuffer = addAndDrawRemoveNode(headNode, np.dot(facePose, _headPose), RenderFlags.RenderFlags.DEPTH_ONLY | RenderFlags.RenderFlags.OFFSCREEN)
    DrawDepth(_headPath, np.dot(facePose, _headPose))
    # we can now use this depth buffer for drawing the obj
    # if part of the obj is occluded by the head, we will skip drawing that part

    # retrieve the obj's node
    objNode = _sceneObjNodes[objPath]
    # add the obj to the to the scene, draw it to both color and depth buffer, and remove it
    #colorObj, depthObj = addAndDrawRemoveNode(objNode, np.dot(facePose, transform), RenderFlags.RenderFlags.ALL_SOLID | RenderFlags.RenderFlags.OFFSCREEN)
    return Draw(objPath, np.dot(facePose, transform))

    # now mask parts of the obj's color buffer using the head's depth buffer and obj's depth buffer
    # depthCorrectedImage = maskWithDepth(depthBuffer, colorObj, depthObj)
    # return depthCorrectedImage
    return colorBuffer
'''

'''
def DrawHand(objPath, transform = np.identity(4)):
    objNode = _sceneObjNodes[objPath]
    colorObj, depthObj = addAndDrawRemoveNode(objNode, transform, RenderFlags.RenderFlags.ALL_SOLID | RenderFlags.RenderFlags.OFFSCREEN)
    return colorObj
'''



def Draw(entity:Entity):
    
    global colorBuffer
    global depthBuffer

    objNode = _sceneObjNodes[entity.path]
    color, depth = addAndDrawRemoveNode(objNode, entity.Transform, RenderFlags.RenderFlags.ALL_SOLID | RenderFlags.RenderFlags.OFFSCREEN)
    #fix depth
    depth[depth < 0.05] = 50
    depth = 50 - depth
    mask = depth > depthBuffer
    depthBuffer = np.where(mask, depth, depthBuffer)
    colorBuffer = np.where(mask[..., None], color, colorBuffer)
    return colorBuffer

def DrawDepth(entity:Entity):
    
    global depthBuffer

    objNode = _sceneObjNodes[entity.path]
    depth = addAndDrawRemoveNode(objNode, entity.Transform, RenderFlags.RenderFlags.DEPTH_ONLY | RenderFlags.RenderFlags.OFFSCREEN)
    
    #fix depth
    depth[depth < 0.05] = 50
    depth = 50 - depth
    mask = depth > depthBuffer
    depthBuffer = np.where(mask, depth, depthBuffer)

def Clear():
    global colorBuffer
    global depthBuffer
    
    if colorBuffer is None or depthBuffer is None:
        return
    
    colorBuffer = np.zeros(colorBuffer.shape)
    depthBuffer = np.zeros(depthBuffer.shape)