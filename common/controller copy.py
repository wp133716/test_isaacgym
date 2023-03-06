import math
from math import pi, sin, cos, tan, asin, acos, atan2
import numpy as np

def euler2quat(euler_angle):
    '''欧拉角->四元数'''
    roll, pitch, yaw = euler_angle
    q0 = (cos(roll/2)*cos(pitch/2)*cos(yaw/2) + sin(roll/2)*sin(pitch/2)*sin(yaw/2))
    q1 = (sin(roll/2)*cos(pitch/2)*cos(yaw/2) - cos(roll/2)*sin(pitch/2)*sin(yaw/2))
    q2 = (cos(roll/2)*sin(pitch/2)*cos(yaw/2) + sin(roll/2)*cos(pitch/2)*sin(yaw/2))
    q3 = (cos(roll/2)*cos(pitch/2)*sin(yaw/2) - sin(roll/2)*sin(pitch/2)*cos(yaw/2))
    return q1, q2, q3, q0

def quat2euler(quaternion):
    '''四元数->欧拉角'''
    q1, q2, q3, q0 = quaternion
    roll = atan2(2*(q2*q3+q0*q1), q0**2-q1**2-q2**2+q3**2)
    print("############## quaternion : ", quaternion)
    print("############## 2*(q1*q3-q0*q2) : ", 2*(q1*q3-q0*q2))
    pitch = -asin(np.clip(2*(q1*q3-q0*q2), -1, 1))
    yaw = atan2(2*(q1*q2+q0*q3), q0**2+q1**2-q2**2-q3**2)
    return roll, pitch, yaw

def cclvf(current_pos, target_pos, speed, radius):
    x_ = current_pos[0] - target_pos[0]
    y_ = current_pos[1] - target_pos[1]

    r = math.sqrt(x_ * x_ + y_ * y_)

    # print("CCLVF distance:", r)
    # print()
    if r < 0.01:
        r = 0.01

    c_ = 0.0

    rd = radius #* (np.cos(i*4*np.pi/10000)+2)
    # print("CCLVF distance:", r," ,rd: ",rd)
    # print()
    if r < rd:
        c_ = r / rd
    else:
        c_ = rd / r
    # print("c_:   ",c_)
    # print()

    #desired course angle
    r_rd_ = r * r - rd * rd
    
    factor = speed/(math.sqrt(pow(r,4)+(c_*c_-2)*rd*rd*r*r+pow(rd,4)))

    v = [-factor*(x_ * r_rd_ / r + c_ * rd * y_),-factor*(y_ * r_rd_ / r - c_ * rd * x_)]
    # v = [10, 10]
    # v = [-10, 10]
    # v = [10, -10]
    # v = [-10, -10]

    return v



class CameraController():
    def __init__(self) -> None:
        self.uav_angle = np.array([0., 0., 0.])
        self.cam_angle = np.array([0., 0., 0.])
        self.focal_dist = 50. # 焦距
        self.target_pos_world = np.asmatrix([0., 0., 0.])
        self.tra_uav2world = np.asmatrix([0., 0., 0.])
        self.tra_cam2uav = np.asmatrix([5., 0., 0.])
        
        '''
        calculate the camera's internal parameter matrix
        '''
        # 相机传感器尺寸(mm)
        self.__sensor_width = 16.
        self.__sensor_height = 9.
        
        # 相机分辨率
        self.__width = 1600.
        self.__height = 900.
        
        # pixel per meter in the x,y axis
        self.__width_meter =  self.__sensor_width * 0.001
        self.__height_meter =  self.__sensor_height * 0.001
        
        self.__alpha = self.__width / self.__width_meter
        self.__beta = self.__height / self.__height_meter
        self.__u0 = self.__width/2
        self.__v0 = self.__height/2

        self.__fx = self.__alpha * self.focal_dist * 0.001
        self.__fy = self.__beta * self.focal_dist * 0.001

        self.camera_matrix = np.asmatrix([[self.__fx, 0.,        self.__u0],
                                          [0.       , self.__fy, self.__v0],
                                          [0.       , 0.,        1.       ]])

    
    def set_params(self, ptz_angle, uav_angle, uav_location, car_location, car_angle, zoom):
        uav_location[2] *= -1
        car_location[2] *= -1
        # ptz_angle[1] *= -1
        self.uav_angle = np.asarray(uav_angle)
        self.cam_angle = np.asarray(ptz_angle)
        self.target_pos_world = np.asmatrix(car_location)
        self.tra_uav2world = np.asmatrix(uav_location)

        self.focal_dist = zoom * 1.1 * 0.001

        self.__fx = self.__alpha * self.focal_dist
        self.__fy = self.__beta * self.focal_dist
        self.camera_matrix = np.asmatrix([[self.__fx, 0.,       self.__u0],
                                          [0.,       self.__fy, self.__v0],
                                          [0.,       0.,        1.       ]])

    def get_rot_uav2world(self):
        roll = self.uav_angle[0] * pi / 180 # roll radians
        pitch = self.uav_angle[1] * pi / 180 # pitch radians
        yaw = self.uav_angle[2] * pi / 180 # yaw radians

        Rx = np.asmatrix([[1, 0        ,  0        ],
                          [0, cos(roll), -sin(roll)],
                          [0, sin(roll),  cos(roll)]])

        Ry = np.asmatrix([[ cos(pitch), 0, sin(pitch)],
                          [ 0         , 1, 0         ],
                          [-sin(pitch), 0, cos(pitch)]])

        Rz = np.asmatrix([[cos(yaw), -sin(yaw), 0],
                          [sin(yaw),  cos(yaw), 0],
                          [0,         0,        1]])
        
        rot_uav2world = Rz * Ry * Rx
        rot_uav2world += 1e-7*np.identity(3)

        return rot_uav2world

    def get_rot_cam2uav(self):
        roll = -self.cam_angle[0] * pi / 180 # roll radians
        pitch = -self.cam_angle[1] * pi / 180 # pitch radians
        yaw = self.cam_angle[2] * pi / 180# yaw radians

        Rx = np.asmatrix([[1, 0,           0         ],
                          [0, cos(pitch), -sin(pitch)],
                          [0, sin(pitch),  cos(pitch)]])

        Ry = np.asmatrix([[ cos(yaw), 0, sin(yaw)],
                          [ 0       , 1, 0       ],
                          [-sin(yaw), 0, cos(yaw)]])

        Rz = np.asmatrix([[cos(roll), -sin(roll), 0],
                          [sin(roll),  cos(roll), 0],
                          [0        ,  0        , 1]])
        
        # coordinate transformation from z,x,y to x,y,z
        rot_coord = np.asmatrix([[0, 0, 1],
                                 [1, 0, 0],
                                 [0, 1, 0]])


        # Rx = np.asmatrix([[1, 0        ,  0        ],
        #                   [0, cos(roll), -sin(roll)],
        #                   [0, sin(roll),  cos(roll)]])

        # Ry = np.asmatrix([[ cos(pitch), 0, sin(pitch)],
        #                   [ 0         , 1, 0         ],
        #                   [-sin(pitch), 0, cos(pitch)]])

        # Rz = np.asmatrix([[cos(yaw), -sin(yaw), 0],
        #                   [sin(yaw),  cos(yaw), 0],
        #                   [0,         0,        1]])

        # rot_cam2uav = Rz * Ry * Rx
        rot_cam2uav = rot_coord * Ry * Rx * Rz
        # rot_cam2uav = rot_coord * rot_cam2uav
        rot_cam2uav += 1e-7*np.identity(3)

        return rot_cam2uav

    def world2pixel(self):
        rot_cam2uav = self.get_rot_cam2uav()
        rot_uav2world = self.get_rot_uav2world()
        print("rot_cam2uav : \n", rot_cam2uav)
        print("rot_uav2world :\n ", rot_uav2world)

        rot_coord2 = np.asmatrix([[0, -1, 0],
                                  [0, 0, -1],
                                  [1, 0, 0]])

        pos_uav = rot_uav2world.I * (self.target_pos_world - self.tra_uav2world).T
        # pos_uav = self.target_pos_world - self.tra_uav2world
        # pos_cam = pos_uav - self.tra_cam2uav
        # pos_cam = np.matmul(np.linalg.inv(rot_cam2uav), (pos_uav - self.tra_cam2uav))
        pos_cam = rot_cam2uav.I * (pos_uav - self.tra_cam2uav.T)

        # pos_cam = rot_coord2 * pos_cam

        print("pos_cam : ", pos_cam)
        normalized_point = pos_cam / (pos_cam[2]+1e-7)
        pixel_point = self.camera_matrix * normalized_point
        # if pos_cam[2]<1e-7:
        #     pixel_point *= 0

        return np.array(pixel_point.T)[0]

if __name__ == "__main__":

    ptz_angle = [0, 90, 90]
    uav_angle = [0., 0., 0.]
    uav_location = [0, 0, 300]
    car_location = [0, 0, 0]
    car_angle = [0., 0., 10]
    zoom = 10.

    cam_control = CameraController()
    cam_control.set_params(ptz_angle, uav_angle, uav_location, car_location, car_angle, zoom)
    print("cam_control.camera_matrix : ", cam_control.camera_matrix)
    pixel_point = cam_control.world2pixel()
    print("pixel_point : ", pixel_point, pixel_point.shape)
