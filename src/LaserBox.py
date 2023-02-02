from src.Line import Line
from src.Plane import Plane
import numpy as np
from scipy.spatial.transform import Rotation as R

class LaserBox:

    def __init__(self, lVector, lAngles, phires, thetares, dres):

        self.phires = phires
        self.thetares = thetares
        self.dres = dres
        self.lVector = np.asarray(lVector)
        self.lAngles = lAngles
        self.r = (R.from_euler('zy', lAngles, degrees=True)).as_matrix()
        xvector = np.asarray([1, 0, 0])
        self.laserDirectionInBox = self.r.dot(xvector)

    def intersect(self, phi, theta, plane):
        
        rotMatrix = (R.from_euler('zy', [phi, theta], degrees=True)).as_matrix()
        laserPosition = rotMatrix.dot(self.lVector)
        laserDirection = rotMatrix.dot(self.laserDirectionInBox)
        line = Line(laserPosition, laserDirection)
        point = plane.intersect(line)
        d = np.linalg.norm(laserPosition - point) 
        return [point, [phi, theta, d]]         

    def point(self, phi, theta, d):

        rotMatrix = (R.from_euler('zy', [phi, theta], degrees=True)).as_matrix()
        laserPosition = rotMatrix.dot(self.lVector)
        laserDirection = rotMatrix.dot(self.laserDirectionInBox)
        return laserPosition + d * laserDirection


    def error(self, phi, theta, d):

        pointPhiPlus = self.point(phi + self.phires, theta, d)
        pointPhiMinus = self.point(phi - self.phires, theta, d)
        #Not really the derivative but speeds up calculation
        derivativePhi = (pointPhiPlus - pointPhiMinus) / 2.0 

        pointThetaPlus = self.point(phi, theta + self.thetares, d)
        pointThetaMinus = self.point(phi, theta - self.thetares, d)
        #Not really the derivative but speeds up calculation
        derivativeTheta = (pointThetaPlus - pointThetaMinus) / 2.0
        
        pointDPlus = self.point(phi, theta, d + self.dres)
        pointDMinus = self.point(phi, theta, d - self.dres)
        #Not really the derivative but speeds up calculation
        derivativeD = (pointDPlus - pointDMinus) / 2.0

        sigmax2 = (derivativePhi[0] * derivativePhi[0]) + (derivativeTheta[0] * derivativeTheta[0]) + (derivativeD[0] * derivativeD[0]) 
        sigmay2 = (derivativePhi[1] * derivativePhi[1]) + (derivativeTheta[1] * derivativeTheta[1]) + (derivativeD[1] * derivativeD[1])
        sigmaz2 = (derivativePhi[2] * derivativePhi[2]) + (derivativeTheta[2] * derivativeTheta[2]) + (derivativeD[2] * derivativeD[2]) 
        

        return np.asarray([sigmax2, sigmay2, sigmaz2])
        

