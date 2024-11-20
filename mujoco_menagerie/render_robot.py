import mujoco
import mujoco.viewer
import os
import time
import numpy as np
import itertools

# Path to your Mujoco XML model file
# xml_file_path = "path_to_your_model.xml"  # Replace with your XML file path
# xml_file_path = '/home/iulian/chole_ws/src/dvrk_urdf/prograsp/obj_files/jaw_left/jaw_left.xml'
xml_file_path = '/home/iulian/chole_ws/src/drrobot/mujoco_menagerie/davinci_prograsp/scene.xml'
# xml_file_path = './universal_robots_ur5e/scene.xml'
# xml_file_path = './franka_emika_panda/panda.xml'

target_angles = {
    "roll": 0.5,
    "pitch": 0.3,
    "yaw": -0.5,
    "jaw": 0.5
}

# def control_joints(model, data, joint_angles):
def control_callback(model, data):
    # Apply position control to each target joint based on the target angles
    roll_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "roll")
    pitch_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pitch")
    right_jaw_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_jaw")
    left_jaw_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_jaw")

    kp = 50  # Position gain for all joints
    data.ctrl[roll_joint_id] = kp * 100 * (target_angles["roll"] - data.qpos[roll_joint_id])
    data.ctrl[pitch_joint_id] = kp * (target_angles["pitch"] - data.qpos[pitch_joint_id])
    data.ctrl[right_jaw_joint_id] = kp * (target_angles["right_jaw"] - data.qpos[right_jaw_joint_id])
    # data.ctrl[left_jaw_joint_id] = kp * (target_angles["left_jaw"] - data.qpos[left_jaw_joint_id])

def control_joints(model, data):
    ## use actuator name to find id
    roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "roll")
    pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch")
    right_jaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw")
    left_jaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "jaw")
    data.ctrl[pitch_id] = 0.5

def compute_camera_extrinsic_matrix(cam):
    """Returns the 4x4 extrinsic matrix considering lookat, distance, azimuth, and elevation."""
    
    # Convert azimuth and elevation to radians
    # print("azimuth type: ", type(cam.azimuth))
    # print("azimuth: ", cam.azimuth)
    azimuth_rad = np.deg2rad(round(float(cam.azimuth) + 1e-3))
    elevation_rad = np.deg2rad(round(float(cam.elevation) + 1e-3))
    r = float(cam.distance)
    # print("azimuth_rad: ", azimuth_rad)
    # print("elevation_rad: ", elevation_rad)
    # print("r: ", r)
    # Compute the camera position in spherical coordinates (of mujoco)
    x = -r * np.cos(azimuth_rad) * np.cos(elevation_rad)
    y = -r * np.sin(azimuth_rad) * np.cos(elevation_rad)
    z = -r * np.sin(elevation_rad)

    x += cam.lookat[0]
    y += cam.lookat[1]
    z += cam.lookat[2]

    C = np.array([x, y, z, 1])
    # print("C: ", C)
    # Compute the camera's forward vector
    forward = cam.lookat - C[:3]
    forward = forward / np.linalg.norm(forward)

    # Compute the camera's right vector
    right = np.cross(forward, np.array([0, 0, 1]))
    right = right / np.linalg.norm(right)
    
    # Compute the camera's up vector
    up = np.cross(right, forward)

    # Construct the rotation matrix
    rotation_matrix = np.array([
        [right[0], right[1], right[2], 0],
        [-up[0], -up[1], -up[2], 0],
        [forward[0], forward[1], forward[2], 0],
        [0, 0, 0, 1]
    ])
    # print("rotation_matrix: ", rotation_matrix)
    
    # Construct the translation matrix
    t = -rotation_matrix @ C
    # Compute the extrinsic matrix

    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix[:3, :3]
    extrinsic_matrix[:3, 3] = t[:3]
    
    return extrinsic_matrix

def random_base_placement(model, data):
    # Random rotation angles
    angle_x = np.random.uniform(-np.pi/4, np.pi/4)  # x-axis rotation
    angle_y = np.random.uniform(-np.pi/4, np.pi/4)  # y-axis rotation
    angle_z = np.random.uniform(-np.pi/4, np.pi/4)  # z-axis rotation
    print("angle_x: ", angle_x)
    print("angle_y: ", angle_y)
    print("angle_z: ", angle_z)
    # Compute quaternion for rotation
    def angle_to_quat(angle, axis):
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        quat = np.zeros(4)
        quat[0] = c
        quat[axis+1] = s
        return quat

    # Compute quaternion
    quat_x = angle_to_quat(angle_x, 0)
    quat_y = angle_to_quat(angle_y, 1)
    quat_z = angle_to_quat(angle_z, 2)

    # Combine rotations (quaternion multiplication)
    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    final_quat = quat_mul(quat_mul(quat_x, quat_y), quat_z)

    # Set base link orientation and position
    model.body('baselink').quat[:] = final_quat
    
    # Compute position adjustment to keep end-effector in frame
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_site")
    
    # Initial simulation step to update kinematics
    mujoco.mj_forward(model, data)
    
    # Get initial end-effector position
    initial_site_pos = data.site_xpos[site_id]
    
    # Adjust base position to center end-effector
    model.body('baselink').pos[:] = -initial_site_pos

    return model, data

def change_viewpoint(viewer):
    azimuth_offset = np.random.uniform(-180, 180)
    elevation_offset = np.random.uniform(-60, 20)
    radius_offset = np.random.uniform(0.5, 1.5) * 0.15


    print("azimuth: ", azimuth_offset)
    print("elevation: ", elevation_offset)
    print("radius: ", radius_offset)

    viewer.cam.lookat[:] = [0.0, 0.0, 0.0]       # Point the camera at (x, y, z)
    viewer.cam.distance = radius_offset                 # Set distance from the target
    viewer.cam.elevation = elevation_offset      # Set camera elevation angle
    viewer.cam.azimuth = azimuth_offset                 # Set camera azimuth angle
    return viewer


def main():

    try:
        # Load the model and create a Mujoco simulation instance
        model = mujoco.MjModel.from_xml_path(xml_file_path)
        data = mujoco.MjData(model)
        

        # Disable contact forces by turning off the contact flag
        ## turn rotation to quaternion and set it
        # rotate = np.array([1, 0.0, 0.0])
        # # rotate = rotate / np.linalg.norm(rotate)
        # # angle = np.linalg.norm(rotate)
        # angle = np.radians(60)
        # c = np.cos(angle / 2)
        # s = np.sin(angle / 2)
        # shaft_length = 0.5
        # model.body('baselink').pos[0] = 0.0
        # model.body('baselink').pos[1] = -np.sin(angle) * shaft_length
        # model.body('baselink').pos[2] = np.cos(angle) * shaft_length


        # model.body('baselink').quat[0] = c
        # model.body('baselink').quat[1] = s * rotate[0]
        # model.body('baselink').quat[2] = s * rotate[1]
        # model.body('baselink').quat[3] = s * rotate[2]
        model, data = random_base_placement(model, data)


        # model.body('baselink').quat[0] = 1.0
        # model.body('baselink').quat[1] = 0.0
        # model.body('baselink').quat[2] = 0.0
        # model.body('baselink').quat[3] = 0.0
        # Set initial joint angles (adjust the indices and values as needed)
        # Example: Set the roll, pitch, right_jaw, and left_jaw joints to specific angles
        print("object_pose", model.body('baselink').pos, model.body('baselink').quat)
        roll_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "roll")
        pitch_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pitch")
        right_jaw_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_jaw")
        left_jaw_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_jaw")
        passive_1_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "passive_1")
        passive_2_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "passive_2")

        # Set desired initial joint positions
        # data.qpos[right_jaw_joint_id] = -0.6
        # data.qpos[left_jaw_joint_id] = 0.37
        # data.qpos[passive_1_joint_id] = -0.55
        # data.qpos[passive_2_joint_id] = -0.55
        # mujoco.set_mjcb_control(control_joints)
        # mujoco.set_mjcb_control(control_callback)
        pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch")

        # Create a viewer and render the simulation
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # for _ in range(10000):
            # Set initial camera position and orientation


            count = 0

            while viewer.is_running():
                # target_angles["right_jaw"] = np.sin(time.time())
                # data.ctrl[right_jaw_joint_id] = 50 * (target_angles["right_jaw"] - data.qpos[right_jaw_joint_id])
                if count % 1000000 == 0:

                    # model, data = random_base_placement(model, data)
                    viewer = change_viewpoint(viewer)

                # data.ctrl[pitch_id] = 0.5
                mujoco.mj_step(model, data)
                viewer.sync()  # Synchronize the viewer with the simulation steps

                # Get end-effector site ID
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_site")
                
                # Get site position and orientation in world frame
                site_pos = data.site_xpos[site_id]
                site_mat = data.site_xmat[site_id].reshape(3,3)

                # Get camera rotation and position
                extrinsic = compute_camera_extrinsic_matrix(viewer.cam)
                # print("extrinsic: ", extrinsic)
                camera_rot = extrinsic[:3, :3] 
                camera_pos = extrinsic[:3, 3]
                # Transform site position to camera frame
                # 1. Translate to camera origin
                translated_pos = site_pos - camera_pos

                # 2. Rotate to camera coordinate system
                camera_relative_pos = camera_rot.T @ translated_pos

                # Transform rotation matrix to camera frame
                camera_relative_rot = camera_rot.T @ site_mat
                count += 1
                mujoco.mj_collision(model, data)
                print(data.qpos[pitch_joint_id])
                # print(data.ncon)
                # print(data.qpos)
                
                # input("Press Enter to continue...")
                # print("End-effector position (camera frame):", camera_relative_pos)
                # print("End-effector rotation matrix (camera frame):\n", camera_relative_rot)

                # if abs(data.qpos[right_jaw_joint_id] - target_angles["right_jaw"]) < 0.01:
                #     i +=1
                    # print(abs(data.qpos[right_jaw_joint_id] - target_angles["right_jaw"]), i)
                    # time.sleep(1)
                    # input("Press Enter to continue...")  # Pause the simulation until Enter is pressed

    except Exception as e:
        print(f"Failed to load or render the model: {e}")


if __name__ == "__main__":
    main()