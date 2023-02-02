from src.Line import Line
import numpy as np
from scipy.spatial.transform import Rotation as R

class Plane:

    def __init__(self, xpoint, phiz, phiy, sizez, sizey):

        self.phiz = phiz
        self.phiy = phiy
        self.sizez = sizez
        self.sizey = sizey
        self.point = np.asarray([xpoint, 0, 0])
        self.r = (R.from_euler('zy', [phiz, phiy], degrees=True)).as_matrix()
        self.normalVector = self.r.dot(np.asarray([1,0,0]))
        self.origin = self.r.dot(np.asarray([0, -sizey/2.0, -sizez/2.0])) 
        self.origin[0] += xpoint
        self.zedge = self.r.dot(np.asarray([0, -sizey/2.0, sizez/2.0])) 
        self.zedge[0] += xpoint
        self.yedge = self.r.dot(np.asarray([0, sizey/2.0, -sizez/2.0])) 
        self.yedge[0] += xpoint
        lz = self.zedge - self.origin
        ly = self.yedge - self.origin
        self.uz = (self.zedge - self.origin) / np.linalg.norm(lz)
        self.uy = (self.yedge - self.origin) / np.linalg.norm(ly)


    def intersect(self, line):

        mu = np.vdot(self.point - line.point, self.normalVector) / np.vdot(line.vector, self.normalVector)
        return line.point + mu * line.vector


    def isInside(self, p):

        v = (p - self.origin)
        zscore = np.vdot(v, self.uz)
        yscore = np.vdot(v, self.uy)
        if zscore < 0 or zscore > self.sizez:
            return False
        if yscore < 0 or yscore > self.sizey:
            return False
        return True
        
