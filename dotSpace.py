import numpy as np
import math
from scipy.interpolate import interp1d
from Entity.surfaces import Donut, MbStrip


import warnings
warnings.filterwarnings("ignore", message="Conversion of an array with ndim > 0 to a scalar is deprecated", category=DeprecationWarning)
  


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

    def focus(self, Object):
        self._cameraCoordinates = None
        self.setCameraCoordinates(Object.getCoordinates())


class LightSource:
    def __init__(self, lx, ly, lz):
        self._vector = np.array([
            [lx],
            [ly],
            [lz]
        ])/np.linalg.norm([lx, ly, lz])
    
    def getLightVector(self):
        return self._vector

    def getSourceGradAlignment(self, Object):
        return -np.dot(Object.getAllGradients(), self._vector)


class Screen():
    def __init__(self, reso, dist):
        assert reso>0
        self.resolution = reso

        self._screen = np.zeros(shape=(reso, reso), dtype=np.float32)
        self._screenBuffer = None
        self._mask = np.zeros(shape=(reso, reso), dtype=np.float32)
        self._dist = dist

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

    # generates the position, z-index, point illumination for all points in the scene
    
    def clearToDefault(self):
        self._screen = np.zeros(shape=(self.resolution, self.resolution), dtype=np.float32)
        self._screenBuffer = None
        self._mask = np.zeros(shape=(self.resolution, self.resolution), dtype=np.float32)


    def generateScreenBuffer(self, cameraCoordinates, alignmentVector):
        self._screenBuffer = np.zeros(shape=(cameraCoordinates.shape[0], 4), dtype=np.float32)
        magnification_factor = 10
        _scaler = magnification_factor*(self._dist/np.dot(cameraCoordinates, np.array([
            [1],
            [0],
            [0]
        ])))
        for idx, row in enumerate(_scaler):
            scaled_position = cameraCoordinates[idx]*float(row[0])
            self._screenBuffer[idx][0] = scaled_position.T[2]
            self._screenBuffer[idx][1] = scaled_position.T[1]
            self._screenBuffer[idx][2] = float(cameraCoordinates[idx].T[0])
            self._screenBuffer[idx][3] = alignmentVector[idx]

    def getScreenBuffer(self):
        return self._screenBuffer

    # for all possible points in the scene, map all points that lie within camera view
    def mapBufferOnScreen(self):
        _min1, _max1 = self._screenBuffer[:, 0].min(), self._screenBuffer[:, 0].max()
        _min2, _max2 = self._screenBuffer[:, 1].min(), self._screenBuffer[:, 1].max()

        minA = min(_min1, _min2)
        maxA = max(_max1, _max2)
        
        interpolator = interp1d([minA, maxA], [0, self.resolution-1])

        for idx, item in enumerate(self._screenBuffer):
            camX, camY = int(interpolator(item[0])), int(interpolator(item[1]))
            if (item[2]>self._mask[camX][camY] and item[3]>self._screen[camX][camY]):
                self._screen[camX][camY]=item[3]


    def paintScreen(self):
        string_screen = ""
        for idx_row, row in enumerate(self._screen):
            string_screen+="\n"
            for idx_col, col in enumerate(row):
                for params in self.paintParams.keys():
                    if col>=params:
                        string_screen+=self.paintParams[params]
                        break
            
        return string_screen
    
    def getMask(self):
        return self._mask
                    

    
    def getScreen(self):
        return self._screen
            


# Helper Function
def focusAndPlot(Object, camera, canvas, lightSource):
    camera.focus(Object)
    canvas.generateScreenBuffer(camera.getCameraCoordinates(), lightSource.getSourceGradAlignment(Object))
    canvas.mapBufferOnScreen()

    paintedCanvas = canvas.paintScreen()
    print(paintedCanvas, end='\r')


def main():
    donut = Donut(50, 60, 1, 3)
    strip = MbStrip(4, 2, 50, 30)

    cameraVector = [5, 0, 10]
    lightSource = LightSource(cameraVector[0], cameraVector[1], cameraVector[2])
    camera = Camera(cameraVector[0], cameraVector[1], cameraVector[2])
    canvas = Screen(50, 7)
    focusAndPlot(donut, camera, canvas, lightSource)
    canvas.clearToDefault()
    focusAndPlot(strip, camera, canvas, lightSource)


if __name__=='__main__':
    main()