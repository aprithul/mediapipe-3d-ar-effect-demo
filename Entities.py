import numpy as np
import gfxmath
import render

class Entity:

    def __init__(self, path, Offset) -> None:
        self.path = path
        self.Transform = np.identity(n=4, dtype=float)


    def SetTransform(self, Transform) -> None:
        if Transform is None:
            return
        self.Transform = Transform

    def GetPos(self):
        return (self.Transform[0][3], self.Transform[1][3], self.Transform[2][3])