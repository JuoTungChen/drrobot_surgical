<div align="center">

# Differentiable Rendering for Surgical Tool Kinematics Estimation
### IROS 2025

## Overview
This project focuses on estimating the kinematics of surgical tools using a differentiable rendering pipeline. By combining modern techniques such as Gaussian Splatting, Linear Blend Skinning, and advanced segmentation methods, we enable accurate estimation of tool poses and joint angles from surgical videos. You can refer to [Dr. Robot](https://github.com/cvlab-columbia/drrobot/tree/main) for more information on training a differentiable robot model.

### Robot Model Preparation
To have a 

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



