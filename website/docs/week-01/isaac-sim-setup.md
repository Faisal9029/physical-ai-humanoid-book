---
sidebar_position: 4
---

# NVIDIA Isaac Sim Setup and Configuration

This guide will walk you through the installation and configuration of NVIDIA Isaac Sim, a high-fidelity simulation environment for robotics development with advanced physics, rendering, and AI capabilities.

## Overview

NVIDIA Isaac Sim is a comprehensive robotics simulation environment that provides:

- Photorealistic rendering with RTX acceleration
- Advanced physics simulation with PhysX
- Integrated AI and perception tools
- ROS 2 bridge for seamless integration
- Isaac ROS packages for perception and manipulation

## Prerequisites

Before installing Isaac Sim, ensure you have:

- NVIDIA GPU with RTX capabilities (RTX 3070 or higher recommended)
- NVIDIA drivers installed (535 or higher)
- CUDA toolkit compatible with your GPU
- ROS 2 Humble Hawksbill installed (completed in previous section)
- At least 20GB free disk space

## System Requirements Check

### Verify GPU and Drivers

```bash
# Check GPU availability
nvidia-smi

# Verify CUDA installation
nvcc --version

# Check compute capability
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

### Verify ROS 2 Installation

```bash
source /opt/ros/humble/setup.bash
ros2 --version
```

## Download Isaac Sim

### Option 1: Isaac Sim Omniverse App (Recommended)

1. Go to [NVIDIA Isaac Sim Downloads](https://developer.nvidia.com/isaac-sim)
2. Create or log into your NVIDIA Developer account
3. Download the Isaac Sim Omniverse App for Linux
4. Extract the archive to your preferred location (e.g., `~/isaac-sim`)

### Option 2: Isaac Sim via Docker (Alternative)

```bash
# Install Docker if not already installed
sudo apt install docker.io
sudo usermod -aG docker $USER
newgrp docker

# Pull Isaac Sim Docker image
docker pull nvcr.io/nvidia/isaac-sim:4.0.0
```

## Isaac Sim Installation

### Method 1: Omniverse App Installation

Extract Isaac Sim to your home directory:

```bash
# Create installation directory
mkdir -p ~/isaac-sim
cd ~/isaac-sim

# Extract the downloaded archive
tar -xzf isaac-sim-*.tar.gz

# Make the launcher executable
chmod +x isaac-sim.sh
```

### Method 2: Python Virtual Environment (Recommended for Development)

```bash
# Create a dedicated virtual environment
python3 -m venv ~/isaac-sim-env
source ~/isaac-sim-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Isaac Sim dependencies
pip install omni-isaac-gym-py
pip install pxr-usd>=21.8
pip install carb-sdk
```

## Isaac Sim Configuration

### Environment Variables Setup

Add the following to your `~/.bashrc`:

```bash
# Isaac Sim Environment Variables
export ISAACSIM_PATH="$HOME/isaac-sim"
export PYTHONPATH="$ISAACSIM_PATH/python:$PYTHONPATH"
export OMNI_URL="omniverse://localhost/NVIDIA/Assets/Isaac/4.0.0"

# GPU settings
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
```

Source the environment:

```bash
source ~/.bashrc
```

### Isaac Sim Python Integration

Create a Python configuration file `~/.isaac-sim-config.py`:

```python
# Isaac Sim Configuration
import omni
import carb

# Performance settings
carb.settings.get_settings().set("/app/window/dpi_scaling", 1.0)
carb.settings.get_settings().set("/app/advanced_physics/fixed_timestep", 1.0/60.0)

# Graphics settings
carb.settings.get_settings().set("/rtx-defaults/antialiasing/level", 4)
carb.settings.get_settings().set("/rtx-defaults/reflections/enable", True)

# Physics settings
carb.settings.get_settings().set("/physics/iterations", 8)
carb.settings.get_settings().set("/physics/solverType", 0)
```

## Isaac ROS Bridge Setup

### Install Isaac ROS Packages

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Create a workspace for Isaac ROS packages
mkdir -p ~/isaac_ws/src
cd ~/isaac_ws

# Clone Isaac ROS packages
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git -b humble
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git -b humble
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_perception.git -b humble
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag.git -b humble
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_point_cloud_interfaces.git -b humble
```

### Build Isaac ROS Packages

```bash
cd ~/isaac_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

### Install Additional Isaac ROS Dependencies

```bash
# Install CUDA dependencies for ROS
sudo apt install -y nvidia-cuda-toolkit
sudo apt install -y libnvinfer-dev libnvparsers-dev

# Install Isaac ROS utilities
pip3 install rospkg catkin_pkg
```

## Isaac Sim Launch and Testing

### Launch Isaac Sim (Method 1: Omniverse App)

```bash
cd ~/isaac-sim
./isaac-sim.sh
```

### Launch Isaac Sim (Method 2: Python Virtual Environment)

```bash
# Activate the Isaac Sim environment
source ~/isaac-sim-env/bin/activate

# Launch Isaac Sim
cd ~/isaac-sim
./python.sh
```

### Basic Isaac Sim Test

Create a simple test script `~/isaac-sim/test_basic.py`:

```python
"""Basic Isaac Sim test script"""
import omni
from pxr import UsdGeom, Gf
import carb

def test_basic_functionality():
    """Test basic Isaac Sim functionality"""
    print("Testing Isaac Sim basic functionality...")

    # Get the stage
    stage = omni.usd.get_context().get_stage()

    # Create a simple prim
    xform = UsdGeom.Xform.Define(stage, "/World/TestObject")

    # Set position
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0))

    print("✓ Basic functionality test passed!")

    # Create a simple cube
    cube = UsdGeom.Cube.Define(stage, "/World/TestObject/Cube")
    cube.GetSizeAttr().Set(0.5)

    print("✓ Cube creation test passed!")

    # Set simulation timestep
    settings = carb.settings.get_settings()
    settings.set("/app/player/playrate", 60.0)

    print("✓ Settings configuration test passed!")

# Run the test
test_basic_functionality()
```

## Isaac Sim ROS 2 Bridge Configuration

### Install ROS 2 Bridge Extension

1. Open Isaac Sim
2. Go to Window → Extensions
3. Search for "ROS 2 Bridge"
4. Install the ROS 2 Bridge extension
5. Restart Isaac Sim

### Configure ROS 2 Bridge

Create a ROS 2 bridge configuration file `~/isaac-sim/ros2_config.json`:

```json
{
  "bridge_extensions": [
    "omni.isaac.ros2_bridge"
  ],
  "bridge_nodes": [
    {
      "name": "isaac_sim_ros_bridge",
      "namespace": "isaac_sim",
      "qos": {
        "sensor_data": {
          "reliability": "best_effort",
          "durability": "volatile",
          "history": "keep_last",
          "depth": 10
        },
        "default": {
          "reliability": "reliable",
          "durability": "volatile",
          "history": "keep_last",
          "depth": 10
        }
      }
    }
  ]
}
```

## Performance Optimization

### Isaac Sim Graphics Settings

Add to your Isaac Sim configuration:

```bash
# Enable RTX features for better rendering
export RTX_GLOBAL_ENABLE=1

# Optimize memory usage
export RTX_MAX_MEMORY=0.8

# Enable multi-GPU if available
export RTX_MULTI_GPU=1
```

### Physics Optimization

For better simulation performance:

```bash
# In Isaac Sim, set physics parameters:
# /physics/iterations = 8 (balance between stability and performance)
# /physics/solverType = 0 (TGS solver for robotics)
# /app/player/updateRate = 60 (match your desired simulation rate)
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: "CUDA error: no kernel image is available"

**Solution**:
```bash
# Check CUDA compute capability
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Ensure Isaac Sim is compatible with your GPU
# Consider using Docker version if native version fails
```

#### Issue: "Isaac Sim fails to launch with OpenGL errors"

**Solution**:
```bash
# Check OpenGL support
glxinfo | grep "OpenGL version"
glxinfo | grep "direct rendering"

# For remote/cloud systems, ensure X11 forwarding or VNC is properly configured
```

#### Issue: "ROS 2 Bridge not connecting"

**Solution**:
```bash
# Verify ROS 2 environment
source /opt/ros/humble/setup.bash
printenv | grep ROS

# Check if ROS 2 bridge extension is enabled in Isaac Sim
# Verify domain IDs match between ROS 2 and Isaac Sim
```

### Verification Commands

```bash
# Check Isaac Sim installation
ls -la ~/isaac-sim/

# Check virtual environment (if using method 2)
source ~/isaac-sim-env/bin/activate
python -c "import omni; print('Isaac Sim Python modules loaded successfully')"

# Check Isaac ROS packages
source ~/isaac_ws/install/setup.bash
ros2 pkg list | grep isaac
```

## Validation Checklist

- [ ] Isaac Sim launches without errors
- [ ] Basic test script runs successfully
- [ ] ROS 2 Bridge extension installed and working
- [ ] Isaac ROS packages built and accessible
- [ ] GPU acceleration is active (check nvidia-smi during simulation)
- [ ] Isaac Sim can connect to ROS 2 network

## Next Steps

Once Isaac Sim is successfully installed and configured:

1. Continue to [Gazebo Setup](./gazebo-setup.md) as an alternative simulation environment
2. Proceed to [Environment Validation](./environment-validation.md) to test your complete setup
3. Explore the Isaac Sim tutorials and examples

Your Isaac Sim environment is now ready for the Physical AI and Humanoid Robotics course!