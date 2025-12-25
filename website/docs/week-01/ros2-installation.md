---
sidebar_position: 3
---

# ROS 2 Installation and Configuration

This guide will walk you through the complete installation and configuration of ROS 2 Humble Hawksbill, the foundation of your Physical AI development environment.

## Overview

ROS 2 (Robot Operating System 2) provides the middleware and tools necessary for developing robotic applications. In this course, we'll use ROS 2 Humble Hawksbill, which offers:

- Real-time performance capabilities
- Improved security features
- Better support for embedded systems
- Enhanced communication protocols

## Prerequisites

Before installing ROS 2, ensure you have:

- Ubuntu 22.04 LTS installed
- Internet connection for package downloads
- Administrative (sudo) access to your system
- Basic familiarity with terminal commands

## System Preparation

### Update System Packages

```bash
sudo apt update && sudo apt upgrade -y
```

### Install Prerequisites

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository universe
```

### Set Locale for UTF-8 Support

```bash
locale  # check for UTF-8
sudo apt install -y locales
sudo locale-gen en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

## Install ROS 2 Humble Hawksbill

### Add ROS 2 Repository

First, add the ROS 2 GPG key:

```bash
sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
```

Add the repository to your sources list:

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### Install ROS 2 Packages

Update package lists and install ROS 2:

```bash
sudo apt update
sudo apt install -y ros-humble-desktop-full
```

### Install Additional Development Tools

```bash
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### Initialize rosdep

```bash
sudo rosdep init
rosdep update
```

## Environment Setup

### Setup ROS 2 Environment Variables

Add ROS 2 setup to your bashrc:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Verify Installation

Test that ROS 2 is properly installed:

```bash
ros2 --version
```

You should see output similar to: `ros2 foxy.0.0-xxxxxx`

## Create a ROS 2 Workspace

### Create Workspace Directory

```bash
mkdir -p ~/physical_ai_ws/src
cd ~/physical_ai_ws
```

### Build the Workspace

```bash
colcon build --symlink-install
```

### Source the Workspace

```bash
echo "source ~/physical_ai_ws/install/setup.bash" >> ~/.bashrc
source ~/physical_ai_ws/install/setup.bash
```

## Essential ROS 2 Packages for Physical AI

### Install Simulation Packages

```bash
sudo apt install -y ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
sudo apt install -y ros-humble-joint-state-publisher ros-humble-robot-state-publisher
```

### Install Perception Packages

```bash
sudo apt install -y ros-humble-vision-opencv ros-humble-cv-bridge
sudo apt install -y ros-humble-image-transport ros-humble-camera-info-manager
```

### Install Navigation Packages

```bash
sudo apt install -y ros-humble-navigation2 ros-humble-nav2-bringup
sudo apt install -y ros-humble-slam-toolbox
```

### Install Control Packages

```bash
sudo apt install -y ros-humble-ros2-control ros-humble-ros2-controllers
sudo apt install -y ros-humble-joint-trajectory-controller
sudo apt install -y ros-humble-diff-drive-controller
```

## Python Package Integration

### Install ROS 2 Python Tools

```bash
pip3 install ros-numpy
pip3 install transforms3d
pip3 install pyquaternion
```

### Install Additional Python Libraries

```bash
pip3 install numpy scipy matplotlib
pip3 install opencv-python
pip3 install open3d
```

## Configuration Files

### Create ROS 2 Configuration

Create a configuration file for your ROS 2 environment:

```bash
mkdir -p ~/.ros2
```

Create `~/.ros2/config.yaml`:

```yaml
# ROS 2 Configuration for Physical AI Development
ros_domain_id: 42  # Unique domain ID for this course

# Logging configuration
logging:
  level: INFO
  format: "[{name}] [{levelname}] {message}"
  style: '{'

# Performance tuning
performance:
  # Use shared memory transport when possible
  use_intraprocess: true

  # QoS settings for robotics applications
  qos:
    sensor_data:
      reliability: best_effort
      durability: volatile
    services:
      reliability: reliable
      durability: volatile
```

## Testing ROS 2 Installation

### Test Basic ROS 2 Functionality

Open two terminal windows and run:

Terminal 1:
```bash
source /opt/ros/humble/setup.bash
source ~/physical_ai_ws/install/setup.bash
ros2 run demo_nodes_cpp talker
```

Terminal 2:
```bash
source /opt/ros/humble/setup.bash
source ~/physical_ai_ws/install/setup.bash
ros2 run demo_nodes_cpp listener
```

You should see messages being published and received between the nodes.

### Test Workspace Creation

```bash
cd ~/physical_ai_ws
mkdir -p src/my_first_robot
cd src/my_first_robot
git clone https://github.com/ros2/tutorials.git
cd ~/physical_ai_ws
colcon build --packages-select demo_nodes_cpp
```

## Common Configuration Issues and Solutions

### Setting ROS Domain ID

If you're on a network with other ROS users:

```bash
export ROS_DOMAIN_ID=42
echo "export ROS_DOMAIN_ID=42" >> ~/.bashrc
```

### Setting RMW Implementation

For better performance with large messages:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
echo "export RMW_IMPLEMENTATION=rmw_cyclonedx_cpp" >> ~/.bashrc
```

### Increasing Shared Memory Limits

For large data transfers:

```bash
echo 'kernel.shmmax=134217728' | sudo tee -a /etc/sysctl.conf
echo 'kernel.shmall=2097152' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## ROS 2 Package Management

### Creating a New Package

```bash
cd ~/physical_ai_ws/src
ros2 pkg create --build-type ament_python my_robot_package --dependencies rclpy std_msgs geometry_msgs sensor_msgs
```

### Building Packages

```bash
cd ~/physical_ai_ws
colcon build --packages-select my_robot_package
source install/setup.bash
```

## Troubleshooting

### Package Installation Issues

If you encounter issues with package installation:

```bash
# Clean package cache
sudo apt clean
sudo apt autoremove
sudo apt update

# Verify repository configuration
apt policy ros-humble-desktop-full
```

### Environment Issues

If ROS 2 commands are not found:

```bash
# Manually source the environment
source /opt/ros/humble/setup.bash
source ~/physical_ai_ws/install/setup.bash

# Check if environment is properly configured
echo $ROS_DISTRO  # Should output "humble"
echo $AMENT_PREFIX_PATH  # Should show ROS 2 installation paths
```

### Workspace Build Issues

If colcon build fails:

```bash
# Clean build artifacts
rm -rf build/ install/ log/

# Rebuild with verbose output
colcon build --event-handlers console_direct+
```

## Performance Optimization

### Setting CPU Governor for Real-time Performance

```bash
sudo apt install linux-tools-generic
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl enable cpufrequtils
sudo systemctl start cpufrequtils
```

### Optimizing Network Settings

```bash
# Add to /etc/sysctl.conf for better network performance
echo 'net.core.rmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## Verification Checklist

- [ ] ROS 2 version command works: `ros2 --version`
- [ ] Workspace builds without errors: `colcon build`
- [ ] Talker/listener test works
- [ ] Essential packages installed (navigation, control, perception)
- [ ] Environment variables properly set
- [ ] Python packages installed and accessible

## Next Steps

Once ROS 2 is successfully installed and configured:

1. Continue to [Isaac Sim Setup](./isaac-sim-setup.md) for advanced simulation
2. Review [Gazebo Setup](./gazebo-setup.md) as an alternative simulation environment
3. Proceed to [Environment Validation](./environment-validation.md) to test your complete setup

Your ROS 2 foundation is now ready for the Physical AI and Humanoid Robotics course!