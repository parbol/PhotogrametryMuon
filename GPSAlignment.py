import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
import math
from optparse import OptionParser
import matplotlib.pyplot as plt
import sys


##################################################################################################
###################################Likelihood class################################################
##################################################################################################
class MyLikelihood(GenericLikelihoodModel):

    def __init__(self, endog, exog, sigmad, sigmax, **kwds):
        self.sigmad = sigmad
        self.sigmax = sigmax 
        super(MyLikelihood, self).__init__(endog, exog, **kwds)

    def loglike(self, params):

        x0, y0, z0 = params
        d = self.endog
        x = self.exog
        sigmad = self.sigmad
        sigmax = self.sigmax
        sigma2 = sigmad * sigmad + sigmax*sigmax 
        chi2 = 0
        for i, di in enumerate(d):
            ri =  math.sqrt((x[i][0] - x0) * (x[i][0] - x0) + (x[i][1] - y0) * (x[i][1] - y0) + (x[i][2] - z0) * (x[i][2] - z0))
            chi2 += (di - ri)*(di - ri)/sigma2   
        return -chi2
##################################################################################################
##################################################################################################




##################################################################################################
################################Some methods I need###############################################
##################################################################################################
def generatePoints(N, xorig, sigmax):

    thepoints = []
    for i in range(0, N):
        x = np.random.uniform(-xorig[0]/2.0, xorig[0]/2.0, 1)
        y = np.random.uniform(-xorig[1]/2.0, xorig[1]/2.0, 1)
        z = np.random.uniform(-xorig[2]/2.0, xorig[2]/2.0, 1)
        p = np.asarray([x, y, z])
        thepoints.append(p)
    xpoints = np.asarray(thepoints)
    points = np.reshape(xpoints, (xpoints.shape[0], xpoints.shape[1]))
    return applyPointUncertainty(points, sigmax)

def applyPointUncertainty(points, sigmax):
    return points + np.random.normal(0, sigmax, points.shape)  

def generateTarget(target):
    
    
    x = np.random.uniform(-target[0]/2.0, target[0]/2.0, 1)[0]
    y = np.random.uniform(-target[1]/2.0, target[1]/2.0, 1)[0]
    z = target[2]
    P = np.asarray([x, y, z])
    return P

def generateDistances(points, target, sigmad):

    diffx = points[:, 0] - target[0]
    diffy = points[:, 1] - target[1]
    diffz = points[:, 2] - target[2]

    d = np.sqrt(diffx * diffx + diffy* diffy + diffz * diffz)
    
    return d + np.random.normal(0, sigmad, d.shape)

def runOneGo(ntrials, number, xorig, ddist, sigmax, sigmad, name):

    dx = []
    dy = []
    dz = []
    for measurement in range(0, ntrials):

        X = generatePoints(number, xorig, sigmax) 
        P = generateTarget(ddist)
        d = generateDistances(X, P, sigmad)

        #print('###############Selected points###############')
        #print(X)
        #print('###############Selected target###############')
        #print(P)
        #print('###############   Distances   ###############')
        #print(d)
        l = MyLikelihood(d, X, sigmad, sigmax)
        res = l.fit()
        if abs(res.params[0] - P[0]) > 10 or abs(res.params[1] - P[1]) > 10 or abs(res.params[2] - P[2]) > 10:
            continue
        dx.append(res.params[0] - P[0])
        dy.append(res.params[1] - P[1])
        dz.append(res.params[2] - P[2])


    means = [np.mean(dx), np.mean(dy), np.mean(dz)]
    stdss = [np.std(dx), np.std(dy), np.std(dz)]

    plt.subplot(1, 3, 1)
    plt.hist(dx, bins=100, range=(-5,5))
    plt.title('X coordinate. Mean: ' + str(np.mean(dx)) + ' Std: ' + str(np.std(dx)))
    plt.ylabel('N. Entries')
    plt.xlabel('x [cm]')
    plt.subplot(1, 3, 2)
    plt.hist(dy, bins=100, range=(-5,5))
    plt.title('Y coordinate. Mean: ' + str(np.mean(dy)) + ' Std: ' + str(np.std(dy)))
    plt.ylabel('N. Entries')
    plt.xlabel('y [cm]')
    plt.subplot(1, 3, 3)
    plt.hist(dz, bins=100, range=(-5,5))
    plt.title('Z coordinate. Mean: ' + str(np.mean(dz)) + ' Std: ' + str(np.std(dz)))
    plt.ylabel('N. Entries')
    plt.xlabel('z [cm]')
    plt.savefig(name)

    return means, stdss     

##################################################################################################
##################################################################################################

    



##################################################################################################
########################################### Main #################################################
##################################################################################################
if __name__ == "__main__":


    parser = OptionParser(usage="%prog --help")
    parser.add_option("-n", "--ntrials",   dest="ntrials",      type=int,        default='5',           help="Number of times to repeat the measurement")
    #parser.add_option("-N", "--number",    dest="number",       type=int,        default='5',           help="Number of points")
    parser.add_option("-O", "--origin",    dest="origin",       type="string",   default='10,10,10',    help="Dimensions of the origin volume: Lx,Ly,Lz")
    parser.add_option("-T", "--target",    dest="target",       type="string",   default='1,1,2',       help="Dimensions of the target surface and distance: Lx,Ly,dz")
    #parser.add_option("-s", "--sigmax",    dest="sigmax",       type=float,      default=1,             help="Precision for origin points")
    #parser.add_option("-S", "--sigmad",    dest="sigmad",       type=float,      default=1,             help="Precision for distancemeter")
    (options, args) = parser.parse_args()


    xorig = []
    for i in options.origin.split(","):
        xorig.append(float(i))
    ddist = []
    for i in options.target.split(","):
        ddist.append(float(i))
    

    theXmean = []
    theYmean = []
    theZmean = []
    theXstds = []
    theYstds = []
    theZstds = []
    
    for number in range(5, 10):
        theSigmaX = np.arange(0.05, 1.55, 0.1)
        theSigmaY = np.arange(0.05, 1.05, 0.1)
        X, Y = np.meshgrid(theSigmaX, theSigmaY)
        XmeanVal = np.zeros( (len(theSigmaX), len(theSigmaY)) )
        YmeanVal = np.zeros( (len(theSigmaX), len(theSigmaY)) )
        ZmeanVal = np.zeros( (len(theSigmaX), len(theSigmaY)) )
        XstdVal = np.zeros( (len(theSigmaX), len(theSigmaY)) )
        YstdVal = np.zeros( (len(theSigmaX), len(theSigmaY)) )
        ZstdVal = np.zeros( (len(theSigmaX), len(theSigmaY)) )
        for i, sigmax in enumerate(theSigmaX):
            for j, sigmad in enumerate(theSigmaY):
                name = 'GPS_number' + str(number) + '_sigmax' + str(sigmax) + '_sigmad' + str(sigmad) + '.png'
                means, stds = runOneGo(options.ntrials, number, xorig, ddist, sigmax, sigmad, name) 
                XmeanVal[i][j] = means[0]
                YmeanVal[i][j] = means[1]
                ZmeanVal[i][j] = means[2]
                XstdVal[i][j] = stds[0]
                YstdVal[i][j] = stds[1]
                ZstdVal[i][j] = stds[2]
                 
        XmeanVal = XmeanVal[:-1, :-1]
        YmeanVal = YmeanVal[:-1, :-1]
        ZmeanVal = ZmeanVal[:-1, :-1]
        XstdVal = XstdVal[:-1, :-1]
        YstdVal = YstdVal[:-1, :-1]
        ZstdVal = ZstdVal[:-1, :-1]

        name = 'sigmaX_' + str(number) + '.png'
        figx, axx = plt.subplots()
        meshx = axx.pcolormesh(X, Y, np.swapaxes(XstdVal,0,1), cmap='viridis')
        cbarx = figx.colorbar(meshx)
        cbarx.set_label('Std Dev x [cm]')
        plt.xlabel('Sigma(x) [cm]')
        plt.ylabel('Sigma(d) [cm]')
        plt.title('X Std Dev vs. x and d resolution')
        figx.savefig(name)


        name = 'sigmaY_' + str(number) + '.png'
        figy, axy = plt.subplots()
        meshy = axy.pcolormesh(X, Y, np.swapaxes(YstdVal,0,1), cmap='viridis')
        cbary = figy.colorbar(meshy)
        cbary.set_label('Std Dev y [cm]')
        plt.xlabel('Sigma(x) [cm]')
        plt.ylabel('Sigma(d) [cm]')
        plt.title('Y Std Dev vs. x and d resolution')
        figy.savefig(name)

        name = 'sigmaZ_' + str(number) + '.png'
        figz, axz = plt.subplots()
        meshz = axz.pcolormesh(X, Y, np.swapaxes(ZstdVal,0,1), cmap='viridis')
        cbarz = figz.colorbar(meshz)
        cbarz.set_label('Mean z [cm]')
        plt.xlabel('Sigma(x) [cm]')
        plt.ylabel('Sigma(d) [cm]')
        plt.title('Z Std Dev vs. x and d resolution')
        figz.savefig(name)


 

