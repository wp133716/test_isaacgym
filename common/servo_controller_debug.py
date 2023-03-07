import numpy as np
from scipy.spatial.transform import Rotation as R
M_PI = 3.14159265358979323846
from math import asin, acos, atan2, sqrt, pi
np.set_printoptions(edgeitems=30, infstr='inf', linewidth=4000, nanstr='nan', precision=10, suppress=True, threshold=10, formatter=None)


class ServoExtPixelParam:
    def __init__(self, moveRoiCam=None, targetRoiCam=None, cameraMatrix=None, camAngle=None):
        self.width = 1280
        self.height = 760
        self.moveRoiCam = Rect(0, 0, 0, 0)
        self.targetRoiCam = Rect(0, 0, 0, 0)
        self.cameraMatrix = cameraMatrix
        self.camAngle = camAngle

class Rect:
    def __init__(self, x=None, y=None, width=None, height=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


def getCameraMatrix(width, height, widthMeter, focalDis):
    u0 = width/2
    v0 = height/2
    alpha = 1/widthMeter
    fxy = alpha * focalDis
    cameraMatrix = np.array([[fxy, 0, u0],
                             [0, fxy, v0],
                             [0, 0, 1]])
    print(cameraMatrix)
    return cameraMatrix


def getSimCameraMatrix(width, height, widthMeter, focalDis):
    u0 = width/2 + 0.5
    v0 = height/2 + 0.5
    alpha = width/widthMeter
    fxy = alpha * focalDis * 0.001
    cameraMatrix = np.array([[fxy, 0, u0],
                             [0, fxy, v0],
                             [0, 0, 1]])
    print(cameraMatrix)
    return cameraMatrix


def convertPixelToPhy(aRoi, aCameraMatrix):
    rotCoord = np.array([[0, 0, 1],
                         [1, 0, 0],
                         [0, 1, 0]])
    pixel_x = aRoi.x + aRoi.width / 2
    pixel_y = aRoi.y + aRoi.height / 2
    aPixelPoint = np.array([pixel_x, pixel_y, 1])
    print("aPixelPoint: ", aPixelPoint)
    print("aCameraMatrix: ", aCameraMatrix)
    aVector = np.linalg.inv(aCameraMatrix) @ aPixelPoint
    aVector = np.asarray(aVector)
    print("aVector: ", aVector, aVector.shape, type(aVector))
    print("aVector[0]: ", aVector[0], aVector[0].shape)
    print("aVector.squeeze(0): ", aVector.squeeze(), aVector.squeeze().shape)
    unitVector = rotCoord @ aVector.squeeze() / np.linalg.norm(aVector.squeeze())
    return unitVector


def convertPhyToPixel(aUnitVector, aCameraMatrix):
    rotCoord = np.array([[0, 1, 0],
                         [0, 0, 1],
                         [1, 0, 0]])
    aVector = rotCoord @ aUnitVector
    aPixelPoint = aCameraMatrix @ aVector
    aPixelPoint = aPixelPoint / aPixelPoint[2]
    roi = [aPixelPoint[0], aPixelPoint[1], 0, 0]
    return roi


def getRotMatrix(aCamAngleRad):
    rotPitchVector = np.array([[np.cos(aCamAngleRad[1]), 0, np.sin(aCamAngleRad[1])],
                               [0, 1, 0],
                               [-np.sin(aCamAngleRad[1]), 0, np.cos(aCamAngleRad[1])]])
    rotRollVector = np.array([[1, 0, 0],
                              [0, np.cos(aCamAngleRad[0]), -np.sin(aCamAngleRad[0])],
                              [0, np.sin(aCamAngleRad[0]), np.cos(aCamAngleRad[0])]])
    rotYawVector = np.array([[np.cos(aCamAngleRad[2]), -np.sin(aCamAngleRad[2]), 0],
                             [np.sin(aCamAngleRad[2]), np.cos(aCamAngleRad[2]), 0],
                             [0, 0, 1]])
    rotMatrix = rotYawVector @ rotRollVector @ rotPitchVector
    return rotMatrix


def getSimRotMatrix(aSimAngleRad):
    rotRollVector = np.array([[1, 0, 0],
                              [0, np.cos(aSimAngleRad[0]), -np.sin(aSimAngleRad[0])],
                              [0, np.sin(aSimAngleRad[0]), np.cos(aSimAngleRad[0])]])
    rotPitchVector = np.array([[np.cos(aSimAngleRad[1]), 0, np.sin(aSimAngleRad[1])],
                               [0, 1, 0],
                               [-np.sin(aSimAngleRad[1]), 0, np.cos(aSimAngleRad[1])]])
    rotYawVector = np.array([[np.cos(aSimAngleRad[2]), -np.sin(aSimAngleRad[2]), 0],
                             [np.sin(aSimAngleRad[2]), np.cos(aSimAngleRad[2]), 0],
                             [0, 0, 1]])
    rotMatrix = rotYawVector @ rotPitchVector @ rotRollVector
    return rotMatrix
    # return aSimAngleRad


'''
 @brief Caculation rotation of camera
 
 '''
def servoExtPixel(aServoExtPixelParam, xPixelMove, yPixelMove):

    aServoExtPixelParam.moveRoiCam.width = 0
    aServoExtPixelParam.moveRoiCam.height = 0
    aServoExtPixelParam.moveRoiCam.x = aServoExtPixelParam.width/2. + xPixelMove
    aServoExtPixelParam.moveRoiCam.y = aServoExtPixelParam.height/2. + yPixelMove
    
    aServoExtPixelParam.targetRoiCam.width = 0
    aServoExtPixelParam.targetRoiCam.height = 0
    aServoExtPixelParam.targetRoiCam.x = aServoExtPixelParam.width/2
    aServoExtPixelParam.targetRoiCam.y = aServoExtPixelParam.height/2
    
    camAngleRad = aServoExtPixelParam.camAngle# * M_PI/180

    unitVectorMove = convertPixelToPhy(aServoExtPixelParam.moveRoiCam, aServoExtPixelParam.cameraMatrix)
    unitVectorTarget = convertPixelToPhy(aServoExtPixelParam.targetRoiCam, aServoExtPixelParam.cameraMatrix)
    print("unitVectorMove: ", unitVectorMove)
    print("unitVectorTarget: ", unitVectorTarget)

    # unitVectorPosMove = getSimRotMatrix(camAngleRad) @ unitVectorMove
    unitVectorPosMove = camAngleRad @ unitVectorMove
    print("unitVectorPosMove: ", unitVectorPosMove)

    servoAngle = np.zeros(3)
    # Set roll as zero.
    servoAngle[0] = 0
    # Compute pitch for servo.
    servoAngle[1] = asin(unitVectorTarget[2]) - asin(unitVectorPosMove[2])
    # Compute yaw for servo.
    posUavOnYaw = np.array([unitVectorPosMove[0], unitVectorPosMove[1], 0])
    unitVectorPosMoveOnYaw = posUavOnYaw / np.linalg.norm(posUavOnYaw)
    servoAngle[2] = acos(unitVectorPosMoveOnYaw[0]) if (unitVectorPosMoveOnYaw[1] > 0) else -acos(unitVectorPosMoveOnYaw[0])
    print("servoAngle: ", servoAngle)

    # Compute the angle for unitVectorPosMove in coordinate system based on initial view.
    coordinateAngle = np.zeros(3)
    # Compute pitch for coordinateAngle.
    coordinateAngle[1] = -asin(unitVectorMove[2])
    # Compute yaw for coordinateAngle.
    posInitOnYaw = np.array([unitVectorMove[0], unitVectorMove[1], 0])
    unitVectorPosInitOnYaw = posInitOnYaw / np.linalg.norm(posInitOnYaw)
    coordinateAngle[2] = acos(unitVectorPosInitOnYaw[0]) if (unitVectorPosInitOnYaw[1] > 0) else -acos(unitVectorPosInitOnYaw[0])
    print("coordinateAngle: ", coordinateAngle)

    # unitYInit = np.dot(getSimRotMatrix(camAngleRad), np.array([0,1,0]))
    # unitZInit = np.dot(getSimRotMatrix(camAngleRad), np.array([0,0,1]))
    unitYInit = np.dot(camAngleRad, np.array([0, 1, 0]))
    unitZInit = np.dot(camAngleRad, np.array([0, 0, 1]))
    rotInitPitchVector = R.from_rotvec(coordinateAngle[1] * unitYInit).as_matrix()
    rotInitYawVector = R.from_rotvec(coordinateAngle[2] * unitZInit).as_matrix()
    unitMoveViewVector = rotInitYawVector @ rotInitPitchVector @ unitYInit
    unitRotViewVector = np.dot(getSimRotMatrix(servoAngle), np.array([0, 1, 0]))
    print("unitYInit: ", unitYInit, type(unitYInit))
    print("unitZInit: ", unitZInit, type(unitZInit))
    print("rotInitPitchVector: \n", rotInitPitchVector, type(rotInitPitchVector))
    print("rotInitYawVector: \n", rotInitYawVector, type(rotInitYawVector))
    print("unitMoveViewVector: \n", unitMoveViewVector, type(unitMoveViewVector))
    print("getSimRotMatrix(servoAngle): ", getSimRotMatrix(servoAngle), type(getSimRotMatrix(servoAngle)))
    print("unitRotViewVector: ", unitRotViewVector, type(unitRotViewVector))
    print("unitRotViewVector @ unitMoveViewVector: ", unitRotViewVector @ unitMoveViewVector, type(unitRotViewVector @ unitMoveViewVector))
    # Compute roll for servo.
    # servoAngle[0] = acos(unitRotViewVector.dot(unitMoveViewVector))
    servoAngle[0] = acos(np.clip(unitRotViewVector @ unitMoveViewVector, -1, 1))
    servoAngle[0] = -servoAngle[0] if (unitMoveViewVector[2]<0) else servoAngle[0]

    print("servoAngle: ", servoAngle)


    # Eigen::Vector3d errorVector = getSimRotMatrix(servoAngle) * unitVectorTarget - unitVectorPosMove;
    # if (errorVector.norm() < 1.e-6)
    # {
    #     std::cout << "*********Success!!!*********" << std::endl;
    # }
    # else
    # {
    #     std::cout << "*********Failure!!!*********" << std::endl;
    # }

    servoAngle = servoAngle * 180 / M_PI
    print("servoAngle: ", servoAngle)


    return servoAngle


if __name__ == "__main__":
    width = 1600 #1280
    height = 900 #760
    servoExtPixelParam = ServoExtPixelParam()
    servoExtPixelParam.width = width
    servoExtPixelParam.height = height
    # servoExtPixelParam.camAngle = np.array([-0, 90, 0])
    servoExtPixelParam.camAngle = R.from_euler('zyx', [-0, 90, 0], degrees=True).as_matrix()

    # servoExtPixelParam.cameraMatrix = np.array([[2586.12,       0,  width/2],
    #                                              [   0   , 2586.12, height/2],
    #                                              [   0   ,    0   ,   1]])

    servoExtPixelParam.cameraMatrix = np.array([[800. ,       0,  width/2],
                                                 [   0   , 800., height/2],
                                                 [   0   ,    0   ,   1]])
    
    servoAngle = servoExtPixel(servoExtPixelParam, 887.3743-width/2, 236.6615-height/2)
    # servoAngle = servoExtPixel(servoExtPixelParam, 25, -100)
    # print("servoAngle : ", servoAngle)