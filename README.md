# glim_segmentation



# Ubuntu and Ros
Ubuntu  22.04
ROS2 Humble 
## livox_ros_driver_2
Follow [livox_ros2_driver Installation](https://github.com/Livox-SDK/livox_ros2_driver)

## GLIM
Follow [GLIM Installation](https://koide3.github.io/glim/installation.html)

## TensorRT-YOLO
- **CUDA**: Recommended version >= 11.0.1
- **TensorRT**: Recommended version >= 8.6.1

## Building
```
cd ~/glim_ws/src
git clone git@git.lwp.rug.nl:asig_project_team/2026_odo_sem_mapping_alex.git
colcon build --symlink-install
```

## Config 

We use the [GLIM config system](https://koide3.github.io/glim/quickstart.html#configuration-files)
You can find ours in config/

## Bags

Download from [Google Drive](https://drive.google.com/file/d/1zRj2bsb5mHkM32AORzi2rpCJcyh8JdqT/view?usp=sharing)

## Usage

The extension module can be loaded into GLIM by adding the name of the created shared library to extension_modules parameter in glim/config/config_ros.json
