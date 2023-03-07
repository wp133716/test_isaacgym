import math
from math import pi, sin, cos, tan, asin, acos, atan2
import numpy as np
from isaacgym import gymapi
import torch
# from torch import pi, sin, cos, tan, asin, acos, atan2

def euler2quat(euler_angle):
    '''欧拉角->四元数'''

    roll, pitch, yaw = euler_angle    
    # roll = torch.tensor(roll)
    # pitch = torch.tensor(pitch)
    # yaw = torch.tensor(yaw)
    w = (cos(roll/2)*cos(pitch/2)*cos(yaw/2) + sin(roll/2)*sin(pitch/2)*sin(yaw/2))
    x = (sin(roll/2)*cos(pitch/2)*cos(yaw/2) - cos(roll/2)*sin(pitch/2)*sin(yaw/2))
    y = (cos(roll/2)*sin(pitch/2)*cos(yaw/2) + sin(roll/2)*cos(pitch/2)*sin(yaw/2))
    z = (cos(roll/2)*cos(pitch/2)*sin(yaw/2) - sin(roll/2)*sin(pitch/2)*cos(yaw/2))

    return x, y, z, w

def quat2euler(quaternion):
    '''四元数->欧拉角'''
    x, y, z, w = quaternion
    # x = torch.tensor(x)
    # y = torch.tensor(y)
    # z = torch.tensor(z)
    # w = torch.tensor(w)
    roll = atan2(2*(y*z+w*x), w**2-x**2-y**2+z**2)
    # print("############## quaternion : ", quaternion)
    # print("############## 2*(x*z-w*y) : ", 2*(x*z-w*y))
    pitch = -asin(np.clip(2*(x*z-w*y), -1, 1))
    yaw = atan2(2*(x*y+w*z), w**2+x**2-y**2-z**2)
    return roll, pitch, yaw

'''scipy库实现四元数与欧拉角的相互转换'''
from scipy.spatial.transform import Rotation as R

def quaternion2euler(quaternion):
    '''四元数 -> 欧拉角'''
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=False)
    # euler = r.as_euler('zyx', degrees=False)
    return euler

def euler2quaternion(euler):
    '''欧拉角->四元数'''
    r = R.from_euler('xyz', euler, degrees=False)
    # r = R.from_euler('zyx', euler, degrees=False)
    quaternion = r.as_quat()
    return quaternion

def euler2rotation(euler):
    '''欧拉角->旋转矩阵'''
    r = R.from_euler('xyz', euler, degrees=False)
    rotation_matrix = r.as_matrix()
    return rotation_matrix


def cclvf(current_pos, target_pos, speed, radius):
    x_ = current_pos[0] - target_pos[0]
    y_ = current_pos[1] - target_pos[1]

    r = math.sqrt(x_ * x_ + y_ * y_)

    # print("CCLVF distance:", r)
    # print()
    if r < 0.01:
        r = 0.01

    rd = radius #* (np.cos(i*4*np.pi/10000)+2)
    # print("CCLVF distance:", r," ,rd: ",rd)
    # print()
    if r < rd:
        c_ = r / rd
    else:
        c_ = rd / r

    #desired course angle
    r_rd_ = r * r - rd * rd
    
    factor = speed/(math.sqrt(pow(r,4)+(c_*c_-2)*rd*rd*r*r+pow(rd,4)))

    v = [-factor*(x_ * r_rd_ / r + c_ * rd * y_),-factor*(y_ * r_rd_ / r - c_ * rd * x_)]
    # v = [10, 10]
    # v = [-10, 10]
    # v = [10, -10]
    # v = [-10, -10]

    return v

def cclvf2(current_pos, target_pos, speed, radius):
    # current_pos = np.array(current_pos)
    x_ = current_pos[:,0] - target_pos[:,0]
    y_ = current_pos[:,1] - target_pos[:,1]
    z_ = current_pos[:,2] - target_pos[:,2]

    r = torch.norm(current_pos[:, :2] - target_pos[:, :2], dim=1)
    r = torch.max(r, torch.tensor(0.01))
    
    rd = radius #* (np.cos(i*4*np.pi/10000)+2)

    # mask = r<rd
    # c_ = (r / rd) * mask + (rd / r) * (r>=rd)
    c_ = torch.where(r<rd, r/rd, rd/r)

    #desired course angle
    r_rd_ = r * r - rd * rd
    
    factor = speed / (torch.sqrt((r**4) + (c_**2-2) * rd**2 * r**2 + rd**4))

    vx = -factor*(x_ * r_rd_ / r + c_ * rd * y_)
    vy = -factor*(y_ * r_rd_ / r - c_ * rd * x_)
    vz = -z_

    v = torch.stack((vx, vy, vz), axis=1)

    return v



class CameraController():
    def __init__(self, cam_props, num_env) -> None:
        self.num_env = num_env
        self.uav_angle = np.zeros((num_env, 3, 1))
        self.cam_angle = np.zeros((num_env, 3, 1))
        self.focal_dist = 18 * 0.001 # 焦距
        self.target_pos_world = np.zeros((num_env, 3, 1))
        self.tra_uav2world = np.zeros((num_env, 3, 1))
        self.tra_cam2uav = np.zeros((num_env, 3, 1))
        
        '''
        calculate the camera's internal parameter matrix
        '''
        # 相机传感器尺寸(mm)
        # self.__sensor_width = 33 #24.5 #50 #7.41 #
        # self.__sensor_height = 5.56
        
        # 相机分辨率
        self.__width = cam_props.width
        self.__height = cam_props.height
        
        # pixel per meter in the x,y axis
        # self.__width_meter =  self.__sensor_width * 0.001
        # self.__height_meter =  self.__sensor_height * 0.001
        
        # self.__alpha = self.__width / self.__width_meter
        # self.__beta = self.__height / self.__height_meter
        self.__u0 = self.__width/2
        self.__v0 = self.__height/2

        # self.__fx = self.__alpha * self.focal_dist * 0.001
        # self.__fy = self.__fx #self.__beta * self.focal_dist * 0.001

        self.__fx = cam_props.width / 2
        self.__fy = self.__fx

        self.camera_matrix = np.asmatrix([[self.__fx, 0.,        self.__u0],
                                          [0.       , self.__fy, self.__v0],
                                          [0.       , 0.,        1.       ]])

    
    def set_params(self, ptz_angle, uav_angle, uav_location, car_location, uav_matrix, view_matrix, projection_matrix, zoom:int):
        # uav_location[0] *= -1
        # car_location[0] *= -1
        # uav_location[1] *= -1
        # car_location[1] *= -1
        # uav_location[2] *= -1
        # car_location[2] *= -1
        self.uav_angle = np.asarray(uav_angle)
        self.cam_angle = np.asarray(ptz_angle)
        self.target_pos_world = np.asarray(car_location)[:,:,np.newaxis]
        self.tra_uav2world = np.asarray(uav_location)[:,:,np.newaxis]
        self.uav_matrix = np.asarray(uav_matrix)
        self.view_matrix = np.asarray(view_matrix)
        self.projection_matrix = np.asarray(projection_matrix)

        # self.focal_dist = zoom * 18 * 0.001

        # self.__fx = self.__alpha * self.focal_dist
        # self.__fy = self.__fx #self.__beta * self.focal_dist
        self.__fx = self.projection_matrix[0, 0] * self.__width / 2
        self.__fy = self.__fx
        self.camera_matrix = np.asarray([[self.__fx, 0.,       self.__u0],
                                          [0.,       self.__fy, self.__v0],
                                          [0.,       0.,        1.       ]])

    def get_rot_uav2world(self):
        roll = 0#self.uav_angle[0]# * pi / 180 # roll radians
        pitch = 0#self.uav_angle[1]# * pi / 180 # pitch radians
        yaw = 0#self.uav_angle[2]# * pi / 180 # yaw radians
        
        rot_uav2world = R.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()
        rot_uav2world = np.asarray(rot_uav2world)
        # rot_uav2world += 1e-7*np.identity(3)

        return rot_uav2world

    # def get_rot_cam2uav(self):
    #     roll = np.deg2rad(-10)#self.cam_angle[2]# * pi / 180 # roll radians
    #     pitch = np.deg2rad(-88) #self.cam_angle[1]# * pi / 180 # pitch radians
    #     yaw = np.deg2rad(60) #self.cam_angle[0]# * pi / 180 # yaw radians

    #     rot_coord = np.asmatrix([[0, 0, 1],
    #                              [1, 0, 0],
    #                              [0, 1, 0]])

    #     rot_matrix = R.from_euler('zxy', [roll, pitch, yaw], degrees=False).as_matrix()
    #     rot_cam2uav = rot_coord * rot_matrix
    #     rot_cam2uav += 1e-7*np.identity(3)

    #     return rot_cam2uav

    def world2pixel(self):
        # rot_cam2uav = self.get_rot_cam2uav()
        rot_cam2uav = self.view_matrix[:3, :3]
        rot_uav2world = self.get_rot_uav2world()
        print("rot_cam2uav : \n", rot_cam2uav)
        print("rot_uav2world :\n ", rot_uav2world)

        pos_uav = np.linalg.inv(rot_uav2world) @ (self.target_pos_world - self.tra_uav2world)
        # pos_uav = self.target_pos_world - self.tra_uav2world
        # pos_cam = pos_uav - self.tra_cam2uav
        # pos_cam = np.matmul(np.linalg.inv(rot_cam2uav), (pos_uav - self.tra_cam2uav))
        # pos_cam = np.linalg.inv(rot_cam2uav) @ (pos_uav - self.tra_cam2uav)
        pos_cam = np.linalg.inv(self.uav_matrix) @ (pos_uav - self.tra_cam2uav)

        # rot_coord1 = np.asmatrix([[-1, 0, 0],
        #                           [0, 1, 0],
        #                           [0, 0, -1]])
        # rot_coord2 = np.asarray([[1, 0, 0],
        #                           [0, -1, 0],
        #                           [0, 0, -1]])
        rot_coord3 = np.asarray([[0, -1, 0],
                                  [0, 0, -1],
                                  [1, 0, 0]])

        # pos_cam = rot_coord1 * pos_cam # rot_cam2uav = self.get_rot_cam2uav()
        # pos_cam = rot_coord2 * pos_cam # rot_cam2uav = self.view_matrix[:3, :3]
        pos_cam = rot_coord3 @ pos_cam
        pos_cam[:, 2:, 0:] = np.maximum(pos_cam[:,2:,0:], 1e-7)

        print("pos_uav : ", pos_uav)
        print("pos_cam : ", pos_cam)
        normalized_point = pos_cam / (pos_cam[:,2:])
        pixel_point = self.camera_matrix @ normalized_point
        print("normalized_point : ", normalized_point)
        print("self.camera_matrix : ", self.camera_matrix)

        # if pos_cam[2]<1e-7:
        #     pixel_point *= 0

        return pixel_point.squeeze(-1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.set_printoptions(edgeitems=30, infstr='inf', linewidth=4000, nanstr='nan', precision=6, suppress=True, threshold=10, formatter=None)

    plt.axis([0, 1600, 0, 900])
    for i in range(1):
        print(i)
        ptz_angle = np.deg2rad([35, 88, 56+i])
        uav_angle = np.deg2rad([20, 3, 52])
        uav_location = [0, 0, -300]
        car_location = [0, 0, 0]
        car_angle = np.deg2rad([0., 0., 0.])
        zoom = 1.

        cam_control = CameraController()
        cam_control.set_params(ptz_angle, uav_angle, uav_location, car_location, car_angle, zoom)
        print("cam_control.camera_matrix : ", cam_control.camera_matrix)
        pixel_point = cam_control.world2pixel()
        plt.scatter(pixel_point[0], pixel_point[1])
        plt.pause(0.1)
        print("pixel_point : ", pixel_point, pixel_point.shape)

    # current_pos = torch.tensor([[0, 0],
    #                             [0, 10]], dtype=torch.float)
    # target_pos = torch.tensor([[0, 0],
    #                            [0, 0]], dtype=torch.float)
    # # current_pos = torch.tensor([[0, 1]], dtype=torch.float)
    # # target_pos = torch.tensor([[10, 10]], dtype=torch.float)
    # # current_pos = torch.tensor([[0, 10]], dtype=torch.float)
    # # target_pos = torch.tensor([[10, 100]], dtype=torch.float)
    # speed = 10
    # radius = 100
    # order_vel = cclvf(current_pos[0], target_pos[0], speed, radius)
    # order_vel_batch = cclvf2(current_pos, target_pos, speed, radius)
    # print(order_vel)
    # print(order_vel_batch)
