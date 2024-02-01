import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
                    self.pointsGrad[_counter] = self._getPointGrad(_x, _y, point)
                    _counter+=1




    def _getPointGrad(self, _x, _y, _z):
        grad = np.array([(2*_x - (self._innerRad+self._outerRad)*_x/math.dist([_x, _y], [0, 0])), (2*_y - 4*_y/math.dist([_x, _y], [0, 0])), 2*_z])
        unitGrad = grad/np.linalg.norm(grad)
        return unitGrad

    def plotSurface(self):
        fig = plt.figure(figsize=(7, 7), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1], self.coordinates[:, 2])
        ax.set_title(f"{self.name.title()} Points 3D plot")
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
  

class MbStrip:
    def __init__(self, midCircleRad, halfWidth,nTheta, nW):
        self.name = "Mobius Strip"
        self._Rad = midCircleRad
        self._halfWidth = halfWidth
        self._ntheta = nTheta
        self._nw = nW

        self.nPoints = self._ntheta*self._nw

        # coordinates of sampled points
        self.coordinates = np.zeros((self.nPoints, 3),dtype=np.float32)

        # gradients array of all points
        self.pointsGrad = np.zeros((self.nPoints, 3), dtype=np.float32)

        idx = 0
        for s in np.linspace(-self._halfWidth, self._halfWidth, self._nw):
            for angle in np.linspace(0, 2*math.pi, self._ntheta):
                _x = (self._Rad + s*math.cos(0.5*angle))*math.cos(angle)
                _y = (self._Rad + s*math.cos(0.5*angle))*math.sin(angle)
                _z = s*math.sin(0.5*angle)
                self.coordinates[idx][0] = _x
                self.coordinates[idx][1] = _y
                self.coordinates[idx][2] = _z
                self.pointsGrad[idx]=self._getPointGrad(_x, _y, _z)
                idx+=1 

    def getCoordinates(self):
        return self.coordinates
    
    def _getPointGrad(self, x, y, z):
        grad = np.array([2*x*(y-2*z)-2*self._Rad*z, (x**2)+y*(3*y-4*z)+(z**2)-self._Rad**2, -2*x*(self._Rad+x)-2*y*(y-z)])
        grad = grad/np.linalg.norm(grad)
        return grad
    
    def getAllGradients(self):
        return self.pointsGrad
    
    def plotSurface(self):
        fig = plt.figure(figsize=(7, 7), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1], self.coordinates[:, 2])
        ax.set_title(f"{self.name.title()} Points 3D plot")
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_xlim(-self._Rad-1, self._Rad+1)
        ax.set_ylim(-self._Rad-1, self._Rad+1)
        # ax.set_zlim(-self._outerRad-1, self._outerRad+1)
        plt.show()




        

# Object Defination for your custom surface
class SurfaceTemplate:
    def __init__(self, *args): 
        self.name = "surface"
        self._params = args
    
    def getArgs(self):
        return self._params
    




def main():
    strip = MbStrip(4, 2, 40, 40)
    strip.plotSurface()

if __name__=="__main__":
    main()