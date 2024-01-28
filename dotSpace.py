import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from time import sleep
from scipy.interpolate import interp1d

import warnings
warnings.filterwarnings("ignore", message="Conversion of an array with ndim > 0 to a scalar is deprecated", category=DeprecationWarning)

class Donut:
    def __init__(self, nZ, nTheta, innerRadius, outerRadius):
        self.name="Donut"

        assert innerRadius<=outerRadius

        self._innerRad = innerRadius
        self._outerRad = outerRadius

        self._nz = nZ
        self._nTheta = nTheta
        self._zRange = (outerRadius-innerRadius)/2
        self.zPoints = np.linspace(-self._zRange, self._zRange, self._nz)

        self.nPoints = 2*nZ*nTheta

        # Coordinates of sampled points
        self.coordinates = np.zeros(shape=(2*self._nz*self._nTheta, 3), dtype=np.float32)

        # Gradients of each point
        self.pointsGrad = np.zeros(shape=(2*self._nz*self._nTheta, 3), dtype=np.float64)

        _counter = 0
        for idx, point in enumerate(self.zPoints):
            r1 = (self._innerRad+self._outerRad)/2 + math.sqrt((self._outerRad-self._innerRad)**2 - 4*(point**2))/2
            r2 = (self._innerRad+self._outerRad)/2 - math.sqrt((self._outerRad-self._innerRad)**2 - 4*(point**2))/2

            for idxT, angle in enumerate(np.linspace(0, 2*math.pi, self._nTheta)):

                # Innner and Outer Circle
                for r in [r1, r2]:
                    if (_counter>=self.nPoints): break
                    _x = r*math.cos(angle)
                    _y = r*math.sin(angle)
                    self.coordinates[_counter][0]=_x
                    self.coordinates[_counter][1]=_y
                    self.coordinates[_counter][2]=point
                    self.pointsGrad[_counter] = self.getPointGradient(_x, _y, point)
                    _counter+=1




    def getPointGradient(self, _x, _y, _z):
        grad = np.array([(2*_x - (self._innerRad+self._outerRad)*_x/math.dist([_x, _y], [0, 0])), (2*_y - 4*_y/math.dist([_x, _y], [0, 0])), 2*_z])
        unitGrad = grad/np.linalg.norm(grad)
        return unitGrad

    def plotDonut(self):
        fig = plt.figure(figsize=(7, 7), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1], self.coordinates[:, 2])
        ax.set_title("Donut Points 3D plot")
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_xlim(-self._outerRad-1, self._outerRad+1)
        ax.set_ylim(-self._outerRad-1, self._outerRad+1)
        ax.set_zlim(-self._outerRad-1, self._outerRad+1)
        plt.show()

    def getParams(self):
        return {
            "name": self.name,
            "innerRadius": self._innerRad,
            "outerRadius": self._outerRad,
            "num_Z": self._nz,
            "num_Theta": self._nTheta
        } 
    
    def getCoordinates(self):
        return self.coordinates
    
    def getAllGradients(self):
        return self.pointsGrad
    

class Camera:
    def __init__(self, cx, cy, cz):
        self.position = np.array([
            [cx],
            [cy],
            [cz]
        ])

        # angles for rotation along Y-axis and Z-axis
        self.thetaY=-math.acos(cx/(math.dist([cx, cz], [0, 0])))
        self.thetaZ = math.acos(math.dist([cx, cz], [0, 0])/(math.dist([cx, cy, cz], [0, 0, 0])))

        self._cameraCoordinates = None
        self.rotationMatrix = self._getRotationMatrix()

    def _getRotationMatrix(self):
        R1 = np.matrix([
            [math.cos(self.thetaY), 0, math.sin(self.thetaY)],
            [0, 1, 0],
            [-math.sin(self.thetaY), 0, math.cos(self.thetaY)]
        ], dtype=np.float64)
        R2 = np.matrix([
            [ math.cos(self.thetaZ), - math.sin(self.thetaZ), 0],
            [math.sin(self.thetaZ), math.cos(self.thetaZ), 0],
            [0, 0, 1]
        ], dtype=np.float64)

        R = np.linalg.inv(np.matmul(R1, R2))
        return R
    

    def setCameraCoordinates(self, coordinates):
        _m=(np.matmul(self.rotationMatrix, coordinates.T-self.position)).T
        _m[:, 0] = -1*_m[:, 0]
        self._cameraCoordinates = _m

    def getCameraCoordinates(self):
        return self._cameraCoordinates


class Screen():
    def __init__(self, reso, dist, i, j, k):
        self.resolution = reso
        self._screen = np.zeros(shape=(reso, reso), dtype=np.float32)

        self._screenBuffer = None

        self._mask = np.zeros(shape=(reso, reso), dtype=np.float32)
        self._dist = dist
        self.lightSource = np.array([i, j, k], dtype=np.float32)/np.linalg.norm([i, j, k])
        self.paintParams = {
            0.90: "@ ",
            0.80: "$ ",
            0.70: "# ",
            0.60: "* ",
            0.50: "; ",
            0.40: ": ",
            0.30: "~ ",
            0.20: "- ",
            0.10: ", ",
            0: "  "
        }

    def generateScreenBuffer(self, cameraCoorindates, grad):
        self._screenBuffer = np.zeros(shape=(cameraCoorindates.shape[0], 4), dtype=np.float32)

        for idx, coordinates in enumerate(cameraCoorindates):
            magnification_factor = 10
            _x, _y, _z = coordinates.T[0], coordinates.T[1], coordinates.T[2]
            _scaler = self._dist/_x
            _y = _y*_scaler*magnification_factor
            _z = _z*_scaler*magnification_factor

            brightness = -np.dot(self.lightSource, grad[idx])
            self._screenBuffer[idx][0]=_z
            self._screenBuffer[idx][1]=_y
            self._screenBuffer[idx][2]=_x
            self._screenBuffer[idx][3]=brightness

    def getScreenBuffer(self):
        return self._screenBuffer

    def mapBufferOnScreen(self):
        _min1, _max1 = self._screenBuffer[:, 0].min(), self._screenBuffer[:, 0].max()
        _min2, _max2 = self._screenBuffer[:, 1].min(), self._screenBuffer[:, 1].max()

        print(_min1, _max1, _min2, _max2)
        interpolatorX = interp1d([_min1, _max1], [0, self.resolution-1])
        interpolatorY = interp1d([_min2, _max2], [0, self.resolution-1])

        for idx, item in enumerate(self._screenBuffer):
            camX, camY = int(interpolatorX(item[0])), int(interpolatorY(item[1]))
            if (item[2]>self._mask[camX][camY] and item[3]>self._screen[camX][camY]):
                self._screen[camX][camY]=item[3]


    def paintScreen(self):
        string_screen = ""
        for idx_row, row in enumerate(self._screen):
            for idx_col, col in enumerate(row):
                for params in self.paintParams.keys():
                    if col>=params:
                        string_screen+=self.paintParams[params]
                        break
            string_screen+="\n"
            
        return string_screen
    
    def getMask(self):
        return self._mask
                    

    
    def getScreen(self):
        return self._screen
            


def main():
    donut = Donut(40, 60, 1, 3)

    cameraVector = [10, 10, 10]

    camera = Camera(cameraVector[0], cameraVector[1], cameraVector[2])
    camera.setCameraCoordinates(donut.getCoordinates())
    cameraCoordinates = camera.getCameraCoordinates()

    canvas = Screen(50, 7, cameraVector[0], cameraVector[1], cameraVector[2])
    canvas.generateScreenBuffer(cameraCoordinates, donut.getAllGradients())
    canvas.mapBufferOnScreen()

    print(canvas.paintScreen())

if __name__=='__main__':
    main()