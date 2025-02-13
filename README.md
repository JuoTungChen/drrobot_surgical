
# Differentiable Rendering for Surgical Tool Kinematics Estimation
### MICCAI 2025

## Overview
This project focuses on estimating the kinematics of surgical tools using a differentiable rendering pipeline. By combining modern techniques such as Gaussian Splatting, Linear Blend Skinning, and advanced segmentation methods, we enable accurate estimation of tool poses and joint angles from surgical videos. You can refer to [Dr. Robot](https://github.com/cvlab-columbia/drrobot/tree/main) for more information on training a differentiable robot model.

### Robot Model Preparation (converting to URDF)
To prepare for a URDF model for the surgical tools, here we outline the steps for preparing it using onshape:
1. If the CAD model is a single file in .iges format, open it in FreeCAD and download the components one by one (.iges).
2. Upload the components to onshape.
3. Use "Assembly" and insert all the components.
4. Mate the components one by one. For the revolute joints that is going to be "joint" in the URDF, make sure to use revolute mate and name them as "dof_*". For "links", use mate connector at the place you want to set the link origin and name them as "link_*". Also, make sure to add limits for the joints with "dof".
5. When mating the components, make sure the child is clicked first, and the parent comes after.
6. You can refer to this official documentation for more information: [onshape-to-robot](https://onshape-to-robot.readthedocs.io/en/latest/design.html)  
7. After the assembly is done, install ```onshape-to-robot``` and create a config.json in a folder you want to store the robot URDF and STL files.
8. Add your onshape access key, secret key, document id, and assmebly name to the config.json.
9. In terminal, use the following command to download and convert the document to URDF:
  ``` 
    onshape-to-robot [directory containing config.json]
  ```
10. Then you can use ```onshape-to-robot-bullet -f [directory containing robot.urdf]``` to visualize the URDF. Or you can create a launch file with robot_description, joint_state_publisher and use Rviz to visualize it.

### Converting to XML to enable controllable joints in MuJoCo
To make the URDF controllable in MuJoCo, you can follow the steps outlined below:
1. Install MuJoCO
2. Launch it using something like: ```./.mujoco/mujoco210/bin/simulate```.
3. Drag and drop the robot.urdf into the simulation.
4. Save it as .XML and quit the simulation.
5. Copy the saved mjmodel.xml in ```.mujoco/mujoco210/bin``` and save it to ```/mujoco_menagerie/[robot_name]``` in this repo.
6. Copy the .STL filss from the directory where you saved the urdf using onshape-to-robot and put it into ```/mujoco_menagerie/[robot_name]/assets```.
7. Modify the .XML file:
  * rename the model and add meshdir="assets" to the ```<compiler>``` tag
  * add ```<equality>``` if you want to couple left and right jaw motion
  * add ```<actuator>``` and use ```<position>``` for position control and tune the parameters like "forcerange", "ctrlrange", "kp", "gear", "dampratio"


## Pipeline
The kinematics estimation process involves the following steps:

1. Data Generation:
  Prepare training data by simulating or capturing images of the surgical tool in various configurations to train the differentiable tool model.

2. Learning a Differentiable Surgical Tool Model:
  Train a differentiable model of the surgical tool using a 3-stage training paradigm:

   <ol type="a">
     <li>Canonical 3D Gaussian training</li>
     <li>Deformation field training</li>
     <li>Joint training</li>
   </ol>

3. Tool Segmentation and Tracking:
    Use SAM-2 (Segment Anything Model v2) to accurately segment and track surgical tools in the video frames.

4. Initial Pose and Joint Angle Estimation:
    Make an initial guess of the tool’s pose and joint angles based on prior knowledge or default settings.

5. Optimization on the First Frame:
    Take the first segmented image from the video and optimize the tool's pose and joint angles by minimizing the difference between the rendered and segmented tool image.

6. Frame-by-Frame Optimization:
    Process the remaining frames sequentially. 
    For each frame:
    Refine the pose and joint angles by running the optimization process for n iterations.
    Use the results from the previous frame as the starting point for the current frame to ensure smooth transitions.

### Data Generation

```
python generate_robot_data.py --model_xml_dir mujoco_menagerie/davinci_prograsp
```

### Training 

```
python train.py --dataset_path data/prograsp --experiment_name prograsp_exp1
```
This script will automatically run through all three stages of training. The latest robot model will be saved under data/prograsp.

To visualize the model training process, you may run tensorboard --logdir output/


### Tool Segmentation
We use SAM-2 to segment the tool in the video.

### Tool Pose Estimation using Differentiable Rendering



