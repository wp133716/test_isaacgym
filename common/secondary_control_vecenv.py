import numpy as np
from scipy.spatial.transform import Rotation as R
from math import asin, acos, atan2, sqrt, pi
import copy
np.set_printoptions(edgeitems=30, infstr='inf', linewidth=4000, nanstr='nan', precision=10, suppress=True, threshold=10, formatter=None)


class SecondaryControl:
    def __init__(self, width=1280, height=760, env_num=1):
        self.width = width
        self.height = height
        self.env_num = env_num

    def get_camera_matrix(self, width, height, width_meter, focal_dis):
        u0 = width / 2
        v0 = height / 2
        alpha = 1 / width_meter
        fxy = alpha * focal_dis
        camera_matrix = np.array([[fxy, 0  , u0],
                                  [0  , fxy, v0],
                                  [0  , 0  , 1]])
        return camera_matrix

    def get_sim_camera_matrix(self, width, height, width_meter, focal_dis):
        u0 = width / 2 + 0.5
        v0 = height / 2 + 0.5
        alpha = width / width_meter
        fxy = alpha * focal_dis * 0.001
        camera_matrix = np.array([[fxy, 0, u0],
                                [0, fxy, v0],
                                [0, 0, 1]])
        return camera_matrix


    def pixel2phy(self, pixel, camera_matrix):
        rot_coord = np.array([[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]])

        pixel_point = np.ones((self.env_num, 3))
        pixel_point[:, :2] = pixel

        print("pixel: ", pixel)
        print("pixel_point: ", pixel_point, pixel_point.shape)
        print("camera_matrix: ", camera_matrix, camera_matrix.shape)
        print("np.linalg.inv(camera_matrix): ", np.linalg.inv(camera_matrix), np.linalg.inv(camera_matrix).shape)
        aVector = np.linalg.inv(camera_matrix) @ pixel_point[:,:,np.newaxis]
        aVector = np.asarray(aVector)

        unit_vector = rot_coord @ aVector / np.linalg.norm(aVector, axis=1)[:, np.newaxis]
        return unit_vector


    def phy2pixel(self, unit_vector, camera_matrix):
        rot_coord = np.array([[0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0]])
        vector = rot_coord @ unit_vector
        pixel_point = camera_matrix @ vector
        pixel_point = pixel_point / pixel_point[2]
        roi = [pixel_point[0], pixel_point[1], 0, 0]
        return roi


    def get_rot_matrix(self, cam_angle_rad):
        rot_pitch = np.array([[np.cos(cam_angle_rad[1]), 0, np.sin(cam_angle_rad[1])],
                                [0, 1, 0],
                                [-np.sin(cam_angle_rad[1]), 0, np.cos(cam_angle_rad[1])]])
        rot_roll = np.array([[1, 0, 0],
                                [0, np.cos(cam_angle_rad[0]), -np.sin(cam_angle_rad[0])],
                                [0, np.sin(cam_angle_rad[0]), np.cos(cam_angle_rad[0])]])
        rot_yaw = np.array([[np.cos(cam_angle_rad[2]), -np.sin(cam_angle_rad[2]), 0],
                                [np.sin(cam_angle_rad[2]), np.cos(cam_angle_rad[2]), 0],
                                [0, 0, 1]])
        rot_matrix = rot_yaw @ rot_roll @ rot_pitch

        return rot_matrix


    def get_sim_rot_matrix(self, cam_angle_rad):
        rot_roll = np.array([[1, 0, 0],
                                [0, np.cos(cam_angle_rad[0]), -np.sin(cam_angle_rad[0])],
                                [0, np.sin(cam_angle_rad[0]), np.cos(cam_angle_rad[0])]])
        rot_pitch = np.array([[np.cos(cam_angle_rad[1]), 0, np.sin(cam_angle_rad[1])],
                                [0, 1, 0],
                                [-np.sin(cam_angle_rad[1]), 0, np.cos(cam_angle_rad[1])]])
        rot_yaw = np.array([[np.cos(cam_angle_rad[2]), -np.sin(cam_angle_rad[2]), 0],
                                [np.sin(cam_angle_rad[2]), np.cos(cam_angle_rad[2]), 0],
                                [0, 0, 1]])
        rot_matrix = rot_yaw @ rot_pitch @ rot_roll

        return rot_matrix


    '''
    @brief Caculation rotation of camera
    
    '''
    def servo_ext_pixel(self, camera_matrix, cam_angle, pixel_move):

        target_pixel = pixel_move + np.array([self.width/2, self.height/2])
        center_pixel = np.ones_like(pixel_move)
        center_pixel = center_pixel * np.array([self.width/2, self.height/2])

        cam_angle_rad = cam_angle# * pi/180

        unit_vector_move = self.pixel2phy(target_pixel, camera_matrix)
        unit_vector_target = self.pixel2phy(center_pixel, camera_matrix)
        print("unit_vector_move: ", unit_vector_move)
        print("unit_vector_target: ", unit_vector_target)

        # unit_vector_pos_move = get_sim_rot_matrix(cam_angle_rad) @ unit_vector_move
        unit_vector_pos_move = cam_angle_rad @ unit_vector_move
        print("unit_vector_pos_move: ", unit_vector_pos_move)

        servo_angle = np.zeros((self.env_num, 3, 1))
        # Set roll as zero.
        # servo_angle[0] = 0
        # Compute pitch for servo.
        servo_angle[:, 1] = np.arcsin(unit_vector_target[:,2]) - np.arcsin(unit_vector_pos_move[:,2])
        print("servo_angle: ", servo_angle)

        # Compute yaw for servo.
        # pos_uav_on_yaw = np.array([unit_vector_pos_move[0], unit_vector_pos_move[1], 0])
        pos_uav_on_yaw = copy.deepcopy(unit_vector_pos_move)
        pos_uav_on_yaw[:,2] = 0
        print("pos_uav_on_yaw: ", pos_uav_on_yaw)
        print("np.linalg.norm(pos_uav_on_yaw, axis=1): ", np.linalg.norm(pos_uav_on_yaw, axis=1))

        unit_vector_pos_move_on_yaw = pos_uav_on_yaw / np.linalg.norm(pos_uav_on_yaw, axis=1)[:, np.newaxis]
        print("unit_vector_pos_move_on_yaw: ", unit_vector_pos_move_on_yaw)
        print("np.arccos(unit_vector_pos_move_on_yaw[:,0]): ", np.arccos(unit_vector_pos_move_on_yaw[:,0]))
        print("servo_angle[:, 2]: ", servo_angle[:, 2])
        # servo_angle[:, 2] = acos(unit_vector_pos_move_on_yaw[0]) if (unit_vector_pos_move_on_yaw[1] > 0) else -acos(unit_vector_pos_move_on_yaw[0])
        servo_angle[:, 2] = np.where(unit_vector_pos_move_on_yaw[:,1] > 0 , np.arccos(unit_vector_pos_move_on_yaw[:,0]), -np.arccos(unit_vector_pos_move_on_yaw[:,0]))
        print("servo_angle: ", servo_angle)

        # Compute the angle for unit_vector_pos_move in coordinate system based on initial view.
        coordinate_angle = np.zeros((self.env_num, 3, 1))
        # Compute pitch for coordinate_angle.
        coordinate_angle[:,1] = -np.arcsin(unit_vector_move[:,2])
        # Compute yaw for coordinate_angle.
        pos_init_on_yaw = copy.deepcopy(unit_vector_move)
        pos_init_on_yaw[:,2] = 0
        # pos_init_on_yaw = np.array([unit_vector_move[0], unit_vector_move[1], 0])
        unit_vector_pos_init_on_yaw = pos_init_on_yaw / np.linalg.norm(pos_init_on_yaw, axis=1)[:, np.newaxis]
        # coordinate_angle[:,2] = acos(unit_vector_pos_init_on_yaw[0]) if (unit_vector_pos_init_on_yaw[1] > 0) else -acos(unit_vector_pos_init_on_yaw[0])
        coordinate_angle[:,2] = np.where(unit_vector_pos_init_on_yaw[:,1] > 0, np.arccos(unit_vector_pos_init_on_yaw[:,0]), -np.arccos(unit_vector_pos_init_on_yaw[:,0])) #np.arccos(unit_vector_pos_init_on_yaw[:,0])
        print("coordinate_angle: ", coordinate_angle)

        # unit_y_init = np.dot(self.get_sim_rot_matrix(cam_angle_rad), np.array([0,1,0]))
        # unit_z_init = np.dot(self.get_sim_rot_matrix(cam_angle_rad), np.array([0,0,1]))
        unit_y_init = cam_angle_rad @ np.array([0, 1, 0])
        unit_z_init = cam_angle_rad @ np.array([0, 0, 1])
        print("cam_angle_rad: ", cam_angle_rad, type(cam_angle_rad))
        print("unit_y_init: ", unit_y_init, unit_y_init.shape, type(unit_y_init))
        print("unit_z_init: ", unit_z_init, unit_z_init.shape, type(unit_z_init))
        rot_init_pitch_vector = R.from_rotvec(coordinate_angle[:,1] * unit_y_init).as_matrix()
        rot_init_yaw_vector = R.from_rotvec(coordinate_angle[:,2] * unit_z_init).as_matrix()
        print("rot_init_pitch_vector: \n", rot_init_pitch_vector, rot_init_pitch_vector.shape, type(rot_init_pitch_vector))
        print("rot_init_yaw_vector: \n", rot_init_yaw_vector, rot_init_yaw_vector.shape, type(rot_init_yaw_vector))
        # unit_move_view_vector = rot_init_yaw_vector @ rot_init_pitch_vector @ unit_y_init
        unit_move_view_vector = rot_init_yaw_vector @ unit_y_init[:,:,np.newaxis]
        # unit_rot_view_vector = np.dot(self.get_sim_rot_matrix(servo_angle), np.array([0, 1, 0]))
        print("servo_angle.squeeze(-1): \n", servo_angle.squeeze(-1), servo_angle.squeeze(-1).shape, type(servo_angle))
        print("R.from_euler('xyz', servo_angle.squeeze(-1), degrees=False): ", R.from_euler('xyz', servo_angle.squeeze(-1), degrees=False).as_matrix(), R.from_euler('xyz', servo_angle.squeeze(-1), degrees=False).as_matrix().shape)

        unit_rot_view_vector = R.from_euler('xyz', servo_angle.squeeze(-1), degrees=False).as_matrix() @ np.array([0, 1, 0])
        
        
        print("unit_move_view_vector: \n", unit_move_view_vector, unit_move_view_vector.shape, type(unit_move_view_vector))
        # print("get_sim_rot_matrix(servo_angle): ", self.get_sim_rot_matrix(servo_angle), type(self.get_sim_rot_matrix(servo_angle)))
        print("unit_rot_view_vector: ", unit_rot_view_vector, unit_rot_view_vector.shape, type(unit_rot_view_vector))
        print("unit_rot_view_vector @ unit_move_view_vector: ", unit_rot_view_vector @ unit_move_view_vector, type(unit_rot_view_vector @ unit_move_view_vector))
        # Compute roll for servo.
        # servo_angle[0] = acos(unit_rot_view_vector.dot(unit_move_view_vector))
        print("unit_rot_view_vector[:,np.newaxis] @ unit_move_view_vector: ", unit_rot_view_vector[:,np.newaxis] @ unit_move_view_vector)

        servo_angle[:,0] = np.arccos(np.clip(unit_rot_view_vector[:,np.newaxis] @ unit_move_view_vector, -1, 1)).squeeze(-1)
        # servo_angle[:,0] = -servo_angle[0] if (unit_move_view_vector[2]<0) else servo_angle[0]
        servo_angle[:,0] =  np.where(unit_move_view_vector[:,2]>0, servo_angle[:,0], -servo_angle[:,0]) #-servo_angle[0] if (unit_move_view_vector[2]<0) else servo_angle[0]

        # print("servo_angle: ", servo_angle)


        # Eigen::Vector3d errorVector = get_sim_rot_matrix(servo_angle) * unit_vector_target - unit_vector_pos_move;
        # if (errorVector.norm() < 1.e-6)
        # {
        #     std::cout << "*********Success!!!*********" << std::endl;
        # }
        # else
        # {
        #     std::cout << "*********Failure!!!*********" << std::endl;
        # }

        servo_angle = servo_angle * 180 / pi
        print("servo_angle: ", servo_angle)


        return servo_angle


if __name__ == "__main__":
    width = 1600 #1280
    height = 900 #760
    sc = SecondaryControl(width, height, 2)

    # camera_angle = np.array([-0, 90, 0])
    camera_angle = np.array([[-10, 90, 45],
                            [10, 90, -45]
                            ])
    camera_angle_as_matrix = R.from_euler('xyz', camera_angle, degrees=True).as_matrix()

    # cameraMatrix = np.array([[2586.12,       0,  width/2],
    #                         [   0   , 2586.12, height/2],
    #                         [   0   ,    0   ,   1]])

    camera_matrix = np.array([[[800. ,       0,  width/2],
                               [   0   , 800., height/2],
                               [   0   ,    0   ,   1]],
                              [[800. ,       0,  width/2],
                               [   0   , 800., height/2],
                               [   0   ,    0   ,   1]]
                               ])

    move_pixel = np.array([[25, 46],
                           [85, -96]
                         ])

    servo_angle = sc.servo_ext_pixel(camera_matrix, camera_angle_as_matrix, move_pixel)
    print("servo_angle : ", servo_angle)