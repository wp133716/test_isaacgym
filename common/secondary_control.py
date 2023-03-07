import numpy as np
from scipy.spatial.transform import Rotation as R
M_PI = 3.14159265358979323846
from math import asin, acos, atan2, sqrt, pi
np.set_printoptions(edgeitems=30, infstr='inf', linewidth=4000, nanstr='nan', precision=10, suppress=True, threshold=10, formatter=None)

class Rect:
    def __init__(self, x=None, y=None, width=None, height=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

class SecondaryControl:
    def __init__(self, width=1280, height=760):
        self.width = width
        self.height = height

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


    def pixel2phy(self, roi, camera_matrix):
        rot_coord = np.array([[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]])
        pixel_x = roi.x + roi.width / 2
        pixel_y = roi.y + roi.height / 2
        pixel_point = np.array([pixel_x, pixel_y, 1])
        print("pixel_point: ", pixel_point)
        print("camera_matrix: ", camera_matrix)
        aVector = np.linalg.inv(camera_matrix) @ pixel_point
        aVector = np.asarray(aVector)
        print("aVector: ", aVector, aVector.shape, type(aVector))
        print("aVector[0]: ", aVector[0], aVector[0].shape)
        print("aVector.squeeze(0): ", aVector.squeeze(), aVector.squeeze().shape)
        unit_vector = rot_coord @ aVector.squeeze() / np.linalg.norm(aVector.squeeze())
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
    def servo_ext_pixel(self, camera_matrix, cam_angle, x_pixel_move, y_pixel_move):

        move_roi = Rect()
        move_roi.width = 0
        move_roi.height = 0
        move_roi.x = width/2. + x_pixel_move
        move_roi.y = height/2. + y_pixel_move
    
        target_roi = Rect()
        target_roi.width = 0
        target_roi.height = 0
        target_roi.x = width/2
        target_roi.y = height/2
        
        cam_angle_rad = cam_angle# * M_PI/180

        unit_vector_move = self.pixel2phy(move_roi, camera_matrix)
        unit_vector_target = self.pixel2phy(target_roi, camera_matrix)
        print("unit_vector_move: ", unit_vector_move)
        print("unit_vector_target: ", unit_vector_target)

        # unit_vector_pos_move = get_sim_rot_matrix(cam_angle_rad) @ unit_vector_move
        unit_vector_pos_move = cam_angle_rad @ unit_vector_move
        print("unit_vector_pos_move: ", unit_vector_pos_move)

        servo_angle = np.zeros(3)
        # Set roll as zero.
        servo_angle[0] = 0
        # Compute pitch for servo.
        servo_angle[1] = asin(unit_vector_target[2]) - asin(unit_vector_pos_move[2])
        # Compute yaw for servo.
        pos_uav_on_yaw = np.array([unit_vector_pos_move[0], unit_vector_pos_move[1], 0])
        unit_vector_pos_move_on_yaw = pos_uav_on_yaw / np.linalg.norm(pos_uav_on_yaw)
        servo_angle[2] = acos(unit_vector_pos_move_on_yaw[0]) if (unit_vector_pos_move_on_yaw[1] > 0) else -acos(unit_vector_pos_move_on_yaw[0])
        print("servo_angle: ", servo_angle)

        # Compute the angle for unit_vector_pos_move in coordinate system based on initial view.
        coordinate_angle = np.zeros(3)
        # Compute pitch for coordinate_angle.
        coordinate_angle[1] = -asin(unit_vector_move[2])
        # Compute yaw for coordinate_angle.
        pos_init_on_yaw = np.array([unit_vector_move[0], unit_vector_move[1], 0])
        unit_vector_pos_init_on_yaw = pos_init_on_yaw / np.linalg.norm(pos_init_on_yaw)
        coordinate_angle[2] = acos(unit_vector_pos_init_on_yaw[0]) if (unit_vector_pos_init_on_yaw[1] > 0) else -acos(unit_vector_pos_init_on_yaw[0])
        print("coordinate_angle: ", coordinate_angle)

        # unit_y_init = np.dot(self.get_sim_rot_matrix(cam_angle_rad), np.array([0,1,0]))
        # unit_z_init = np.dot(self.get_sim_rot_matrix(cam_angle_rad), np.array([0,0,1]))
        unit_y_init = np.dot(cam_angle_rad, np.array([0, 1, 0]))
        unit_z_init = np.dot(cam_angle_rad, np.array([0, 0, 1]))
        rot_init_pitch_vector = R.from_rotvec(coordinate_angle[1] * unit_y_init).as_matrix()
        rot_init_yaw_vector = R.from_rotvec(coordinate_angle[2] * unit_z_init).as_matrix()
        # unit_move_view_vector = rot_init_yaw_vector @ rot_init_pitch_vector @ unit_y_init
        unit_move_view_vector = rot_init_yaw_vector @ unit_y_init
        unit_rot_view_vector = np.dot(self.get_sim_rot_matrix(servo_angle), np.array([0, 1, 0]))
        print("unit_y_init: ", unit_y_init, type(unit_y_init))
        print("unit_z_init: ", unit_z_init, type(unit_z_init))
        print("rot_init_pitch_vector: \n", rot_init_pitch_vector, type(rot_init_pitch_vector))
        print("rot_init_yaw_vector: \n", rot_init_yaw_vector, type(rot_init_yaw_vector))
        print("unit_move_view_vector: \n", unit_move_view_vector, type(unit_move_view_vector))
        print("get_sim_rot_matrix(servo_angle): ", self.get_sim_rot_matrix(servo_angle), type(self.get_sim_rot_matrix(servo_angle)))
        print("unit_rot_view_vector: ", unit_rot_view_vector, type(unit_rot_view_vector))
        print("unit_rot_view_vector @ unit_move_view_vector: ", unit_rot_view_vector @ unit_move_view_vector, type(unit_rot_view_vector @ unit_move_view_vector))
        # Compute roll for servo.
        # servo_angle[0] = acos(unit_rot_view_vector.dot(unit_move_view_vector))
        servo_angle[0] = acos(np.clip(unit_rot_view_vector @ unit_move_view_vector, -1, 1))
        servo_angle[0] = -servo_angle[0] if (unit_move_view_vector[2]<0) else servo_angle[0]

        print("servo_angle: ", servo_angle)


        # Eigen::Vector3d errorVector = get_sim_rot_matrix(servo_angle) * unit_vector_target - unit_vector_pos_move;
        # if (errorVector.norm() < 1.e-6)
        # {
        #     std::cout << "*********Success!!!*********" << std::endl;
        # }
        # else
        # {
        #     std::cout << "*********Failure!!!*********" << std::endl;
        # }

        servo_angle = servo_angle * 180 / M_PI
        print("servo_angle: ", servo_angle)


        return servo_angle


if __name__ == "__main__":
    width = 1600 #1280
    height = 900 #760
    sc = SecondaryControl(width, height)

    # camera_angle = np.array([-0, 90, 0])
    camera_angle_as_matrix = R.from_euler('xyz', [-10, 90, 45], degrees=True).as_matrix()

    # cameraMatrix = np.array([[2586.12,       0,  width/2],
    #                         [   0   , 2586.12, height/2],
    #                         [   0   ,    0   ,   1]])

    camera_matrix = np.array([[800. ,       0,  width/2],
                            [   0   , 800., height/2],
                            [   0   ,    0   ,   1]])

    servo_angle = sc.servo_ext_pixel(camera_matrix, camera_angle_as_matrix, 25, 46)
    # servo_angle = servo_ext_pixel(servo_ext_pixelParam, 25, -100)
    print("servo_angle : ", servo_angle)