import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
import math
from optparse import OptionParser
import matplotlib.pyplot as plt
import sys
from scipy.spatial.transform import Rotation as R
from src.Line import Line
from src.Plane import Plane
from src.LaserBox import LaserBox

##################################################################################################
###################################Likelihood class################################################
##################################################################################################
class MyLikelihoodAll(GenericLikelihoodModel):

    def __init__(self, endog, exog, sigmad, sigmaphi, sigmatheta, **kwds):
        self.sigmad = sigmad
        self.sigmaphi = sigmaphi
        self.sigmatheta = sigmatheta
        super(MyLikelihoodAll, self).__init__(endog, exog, **kwds)

    def loglike(self, params):

        lx, ly, lz, phiz, phiy = params
        d = self.endog
        x = self.exog
        sigmad = self.sigmad
        sigmaphi = self.sigmaphi
        sigmatheta = self.sigmatheta
        laserbox = LaserBox([lx, ly, lz], [phiz, phiy], sigmaphi, sigmatheta, sigmad)
        chi2 = 0
        for i, di in enumerate(d):
            phi1 = x[i][0]
            theta1 = x[i][1]
            d01 = x[i][2]
            phi2 = x[i][3]
            theta2 = x[i][4]
            d02 = x[i][5]
            point1 = laserbox.point(phi1, theta1, d01)                                        
            error12 = laserbox.error(phi1, theta1, d01)                                        
            point2 = laserbox.point(phi2, theta2, d02)
            error22 = laserbox.error(phi2, theta2, d02)
            dist, edist2 = self.distanceAndError(point1, point2, error12, error22)
            chi2 -= (di - dist)*(di - dist) / edist2
        print(lx, ly, lz, phiz, phiy, chi2)
        return chi2

    #def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
    #    # we have one additional parameter and we need to add it for summary
    #    self.exog_names.append('alpha')
    #    return super(MyLikelihood, self).fit(start_params=start_params,
    #                                 maxiter=maxiter, maxfun=maxfun,
    #                                 **kwds)

    def distanceAndError(self, p1, p2, e12, e22):

        err2 = [e12[0] + e22[0], e12[1] + e22[1], e12[2] + e22[2]]
        p = p1 - p2
        distance = np.linalg.norm(p)
        errorDistance2 = ((p[0] * p[0] * err2[0]) + (p[1] * p[1] * err2[1]) + (p[2] * p[2] * err2[2])) / (distance * distance)
        return [distance, errorDistance2]
        

##################################################################################################
##################################################################################################

##################################################################################################
###################################Likelihood class################################################
##################################################################################################
class MyLikelihoodPosition(GenericLikelihoodModel):

    def __init__(self, endog, exog, sigmad, sigmaphi, sigmatheta, phiz, phiy, **kwds):
        self.sigmad = sigmad
        self.sigmaphi = sigmaphi
        self.sigmatheta = sigmatheta
        self.phiz = phiz
        self.phiy = phiy
        super(MyLikelihoodPosition, self).__init__(endog, exog, **kwds)

    def loglike(self, params):

        #lx, ly, lz, phix, phiy, phiz = params
        lx, ly, lz = params
        d = self.endog
        x = self.exog
        sigmad = self.sigmad
        sigmaphi = self.sigmaphi
        sigmatheta = self.sigmatheta
        laserbox = LaserBox([lx, ly, lz], [self.phiz, self.phiy], sigmaphi, sigmatheta, sigmad)
        chi2 = 0
        for i, di in enumerate(d):
            phi1 = x[i][0]
            theta1 = x[i][1]
            d01 = x[i][2]
            phi2 = x[i][3]
            theta2 = x[i][4]
            d02 = x[i][5]
            point1 = laserbox.point(phi1, theta1, d01)                                        
            error12 = laserbox.error(phi1, theta1, d01)                                        
            point2 = laserbox.point(phi2, theta2, d02)
            error22 = laserbox.error(phi2, theta2, d02)
            dist, edist2 = self.distanceAndError(point1, point2, error12, error22)
            chi2 -= (di - dist)*(di - dist) / edist2
        print(lx, ly, lz, self.phiz, self.phiy, chi2)
        return chi2

    #def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
    #    # we have one additional parameter and we need to add it for summary
    #    self.exog_names.append('alpha')
    #    return super(MyLikelihood, self).fit(start_params=start_params,
    #                                 maxiter=maxiter, maxfun=maxfun,
    #                                 **kwds)

    def distanceAndError(self, p1, p2, e12, e22):

        err2 = [e12[0] + e22[0], e12[1] + e22[1], e12[2] + e22[2]]
        p = p1 - p2
        distance = np.linalg.norm(p)
        errorDistance2 = ((p[0] * p[0] * err2[0]) + (p[1] * p[1] * err2[1]) + (p[2] * p[2] * err2[2])) / (distance * distance)
        return [distance, errorDistance2]
        

##################################################################################################
##################################################################################################

##################################################################################################
###################################Likelihood class################################################
##################################################################################################
class MyLikelihoodAngle(GenericLikelihoodModel):

    def __init__(self, endog, exog, sigmad, sigmaphi, sigmatheta, lx, ly, lz, **kwds):
        self.sigmad = sigmad
        self.sigmaphi = sigmaphi
        self.sigmatheta = sigmatheta
        self.lx = lx
        self.ly = ly
        self.lz = lz
        super(MyLikelihoodAngle, self).__init__(endog, exog, **kwds)

    def loglike(self, params):

        phiz, phiy = params
        d = self.endog
        x = self.exog
        sigmad = self.sigmad
        sigmaphi = self.sigmaphi
        sigmatheta = self.sigmatheta
        laserbox = LaserBox([self.lx, self.ly, self.lz], [phiz, phiy], sigmaphi, sigmatheta, sigmad)
        chi2 = 0
        for i, di in enumerate(d):
            phi1 = x[i][0]
            theta1 = x[i][1]
            d01 = x[i][2]
            phi2 = x[i][3]
            theta2 = x[i][4]
            d02 = x[i][5]
            point1 = laserbox.point(phi1, theta1, d01)                                        
            error12 = laserbox.error(phi1, theta1, d01)                                        
            point2 = laserbox.point(phi2, theta2, d02)
            error22 = laserbox.error(phi2, theta2, d02)
            dist, edist2 = self.distanceAndError(point1, point2, error12, error22)
            chi2 -= (di - dist)*(di - dist) / edist2
        print(self.lx, self.ly, self.lz, phiz, phiy, chi2)
        return chi2

    #def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
    #    # we have one additional parameter and we need to add it for summary
    #    self.exog_names.append('alpha')
    #    return super(MyLikelihood, self).fit(start_params=start_params,
    #                                 maxiter=maxiter, maxfun=maxfun,
    #                                 **kwds)

    def distanceAndError(self, p1, p2, e12, e22):

        err2 = [e12[0] + e22[0], e12[1] + e22[1], e12[2] + e22[2]]
        p = p1 - p2
        distance = np.linalg.norm(p)
        errorDistance2 = ((p[0] * p[0] * err2[0]) + (p[1] * p[1] * err2[1]) + (p[2] * p[2] * err2[2])) / (distance * distance)
        return [distance, errorDistance2]
        

##################################################################################################
##################################################################################################



##################################################################################################
##################################################################################################
def generateCalibrationTable(targetPlane, npoints, laserbox):

    maxphi = np.arctan((targetPlane.sizey/2.0) / targetPlane.point[0])
    maxtheta = np.arctan((targetPlane.sizez/2.0) / targetPlane.point[0])

    points = []
    numberOfPoints = 0
    while numberOfPoints < npoints:
        phi = 180.0 * np.random.uniform(-maxphi, maxphi, 1)[0] / np.pi
        theta = 180.0 * np.random.uniform(-maxtheta, maxtheta, 1)[0] / np.pi
        point = laserbox.intersect(phi, theta, targetPlane)
        if targetPlane.isInside(point[0]):
            points.append(point)
            numberOfPoints += 1

    return points
##################################################################################################
##################################################################################################


##################################################################################################
##################################################################################################
def randomizeMeasurements(idealMeasurements, phires, etares, dres):

    points = []
    for meas in idealMeasurements:
        p = meas[0]
        v = meas[1]
        v[0] = v[0] + np.random.normal(0, phires, 1)[0]
        v[1] = v[1] + np.random.normal(0, etares, 1)[0]
        v[2] = v[2] + np.random.normal(0, dres, 1)[0]
        points.append([p, v])

    return points
##################################################################################################
##################################################################################################


##################################################################################################
##################################################################################################
def merge(dData, xData):

    d = dData[0]
    x = xData[0]
    for i in range(1, len(dData)):
        d = np.concatenate((d, dData[i]), axis=0)
        x = np.concatenate((x, xData[i]), axis=0)
    return d, x



##################################################################################################
##################################################################################################
def produceData(idealPoints, realPoints):

    d = []
    x = []
    for i in range(0, len(idealPoints)):
        for j in range(0, len(idealPoints)):
            if j >= i:
                continue
            d.append(np.linalg.norm(idealPoints[i][0] - idealPoints[j][0]))
            v = [realPoints[i][1][0], realPoints[i][1][1], realPoints[i][1][2], realPoints[j][1][0], realPoints[j][1][1], realPoints[j][1][2]]
            x.append(v)            
    xpoints = np.asarray(x)
    points = np.reshape(xpoints, (xpoints.shape[0], xpoints.shape[1]))
    return [np.asarray(d), points]
##################################################################################################
##################################################################################################


##################################################################################################
########################################### Main #################################################
##################################################################################################
if __name__ == "__main__":


    parser = OptionParser(usage="%prog --help")
    parser.add_option("-n", "--ntrials",            dest="ntrials",              type=int,          default=5,               help="Number of times to repeat the measurement")
    parser.add_option("-N", "--number",             dest="number",               type=int,          default=5,               help="Number of points per side of the calib table")
    parser.add_option("-c", "--calibTableSize",     dest="calibTableSize",       type='string',     default='10,10',         help="Size of the calib table")
    parser.add_option("-x", "--calibTableX",        dest="calibTableX",          type='string',     default='1,1',           help="X of the calib table")
    parser.add_option("-a", "--calibTableAngle",    dest="calibTableAngle",      type='string',     default='45,70',         help="Angle of the calib table")
    parser.add_option("-l", "--lvector",            dest="lvector",              type='string',     default='10,10,10',      help="Position of laser point")
    parser.add_option("-L", "--lrot",               dest="lrot",                 type='string',     default='10,10',         help="Rotation of laser point")
    parser.add_option("-s", "--svector",            dest="svector",              type='string',     default='10,10,10',      help="Starting position of laser point")
    parser.add_option("-S", "--Srot",               dest="srot",                 type='string',     default='10,10',         help="Starting rotation of laser point")
    (options, args) = parser.parse_args()



    ############### Laser box information #############
    lvector = [float(options.lvector.split(",")[0]), float(options.lvector.split(",")[1]), float(options.lvector.split(",")[2])]
    lrot = [float(options.lrot.split(",")[0]), float(options.lrot.split(",")[1])]
    svector = [float(options.svector.split(",")[0]), float(options.svector.split(",")[1]), float(options.svector.split(",")[2])]
    srot = [float(options.srot.split(",")[0]), float(options.srot.split(",")[1])]
    phires = 0.1
    thetares = 0.1
    dres = 0.1
    laserbox = LaserBox(lvector, lrot, phires, thetares, dres)  

    ############## Target planes ##############
    npoints = options.number
    nplanes = len(options.calibTableX.split(","))
    xtable = []
    tableanglez = []
    tableangley = []
    sizez = []
    sizey = []
    targetPlane = []
    idealMeasurements = []
    realMeasurements = []
    dData = []
    xData = []
    for i in range(0, nplanes):
        xtable.append(float(options.calibTableX.split(",")[i]))
        tableanglez.append(float(options.calibTableAngle.split(",")[i*2]))
        tableangley.append(float(options.calibTableAngle.split(",")[i*2+1]))
        sizez.append(float(options.calibTableSize.split(",")[i*2]))
        sizey.append(float(options.calibTableSize.split(",")[i*2+1]))
        targetPlane.append(Plane(xtable[i], tableanglez[i], tableangley[i], sizez[i], sizey[i]))
        idealMeasurements.append(generateCalibrationTable(targetPlane[i], npoints, laserbox))
        realMeasurements.append(randomizeMeasurements(idealMeasurements[i], phires, thetares, dres))
        d, x = produceData(idealMeasurements[i], realMeasurements[i])
        dData.append(d)
        xData.append(x)  

    d, x = merge(dData, xData)
    
    #First position iteration
    l_pos1 = MyLikelihoodPosition(d, x, dres, phires, thetares, 0, 0)
    res_pos1 = l_pos1.fit(start_params=np.asarray([svector[0], svector[1], svector[2]]))
    #First angular iteration 
    l_ang1 = MyLikelihoodAngle(d, x, dres, phires, thetares, res_pos1.params[0], res_pos1.params[1], res_pos1.params[2])
    res_ang1 = l_ang1.fit(start_params=np.asarray([lvector[0], lvector[1]]))
    
    #Second position iteration
    l_pos2 = MyLikelihoodPosition(d, x, dres, phires, thetares, res_ang1.params[0], res_ang1.params[1])
    res_pos2 = l_pos2.fit(start_params=np.asarray([res_pos1.params[0], res_pos1.params[1], res_pos1.params[2]]))
    #Second angular iteration 
    l_ang2 = MyLikelihoodAngle(d, x, dres, phires, thetares, res_pos2.params[0], res_pos2.params[1], res_pos2.params[2])
    res_ang2 = l_ang2.fit(start_params=np.asarray([res_ang1.params[0], res_ang1.params[1]]))
    
    #Third position iteration
    l_pos3 = MyLikelihoodPosition(d, x, dres, phires, thetares, res_ang2.params[0], res_ang2.params[1])
    res_pos3 = l_pos3.fit(start_params=np.asarray([res_pos2.params[0], res_pos2.params[1], res_pos2.params[2]]))
    #Third angular iteration 
    l_ang3 = MyLikelihoodAngle(d, x, dres, phires, thetares, res_pos3.params[0], res_pos3.params[1], res_pos3.params[2])
    res_ang3 = l_ang3.fit(start_params=np.asarray([res_ang2.params[0], res_ang2.params[1]]))
    
    #Final iteration 
    lfinal = MyLikelihoodAll(d, x, dres, phires, thetares)
    rfinal = lfinal.fit(start_params=np.asarray([res_pos3.params[0], res_pos3.params[1], res_pos3.params[2], res_ang3.params[0], res_ang3.params[1]]))

    print('Fit to positions only. Iter 1.')
    print(res_pos1.params)
    print(res_pos1.bse)
    print('Fit to angles only. Iter 1')
    print(res_ang1.params)
    print(res_ang1.bse)

    print('Fit to positions only. Iter 2.')
    print(res_pos2.params)
    print(res_pos2.bse)
    print('Fit to angles only. Iter 2')
    print(res_ang2.params)
    print(res_ang2.bse)

    print('Fit to positions only. Iter 3.')
    print(res_pos3.params)
    print(res_pos3.bse)
    print('Fit to angles only. Iter 3')
    print(res_ang3.params)
    print(res_ang3.bse)

    print('Global fit')
    print(rfinal.params)
    print(rfinal.bse)
    




 

