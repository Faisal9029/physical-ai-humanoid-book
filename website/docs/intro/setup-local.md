---
sidebar_position: 3
---

# Local Development Environment Setup

This guide will walk you through setting up a complete Physical AI development environment on your local Ubuntu 22.04 system. This setup provides optimal performance for simulation and perception tasks.

## Prerequisites Check

Before starting, ensure you have:

- Ubuntu 22.04 LTS installed
- NVIDIA GPU with latest drivers (470+ recommended)
- At least 32GB RAM and 100GB free disk space
- Internet connection for package downloads

## System Preparation

### Update System Packages

```bash
sudo apt update && sudo apt upgrade -y
```

### Install Essential Development Tools

```bash
sudo apt install -y build-essential cmake git python3-pip python3-dev
sudo apt install -y curl wget vim htop
```

## Install ROS 2 Humble Hawksbill

### Set Locale

```bash
locale  # check for UTF-8
sudo apt update && sudo apt install locales
sudo locale-gen en_US.UTF-8
```

### Add ROS 2 Repository

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository universe
```

### Install ROS 2 GPG Key

```bash
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
```

### Add ROS 2 Repository to Sources List

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### Install ROS 2 Packages

```bash
sudo apt update
sudo apt install -y ros-humble-desktop-full
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### Initialize rosdep

```bash
sudo rosdep init
rosdep update
```

### Setup ROS 2 Environment

Add ROS 2 setup to your bashrc:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Install NVIDIA Isaac Sim

### Prerequisites for Isaac Sim

```bash
sudo apt install -y python3.10-venv python3-pip
```

### Download Isaac Sim

1. Go to [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) and download the latest version
2. Extract the downloaded archive to your preferred location (e.g., `~/isaac-sim`)

### Setup Isaac Sim Environment

```bash
# Create a Python virtual environment for Isaac Sim
python3 -m venv ~/isaac-sim-env
source ~/isaac-sim-env/bin/activate
pip install --upgrade pip
```

### Install Isaac Sim Dependencies

```bash
# Activate Isaac Sim environment
source ~/isaac-sim-env/bin/activate

# Install Isaac ROS dependencies
pip install omni-isaac-gym-py
pip install pxr-usd>=21.8
```

## Install Gazebo (Alternative Simulation)

### Install Gazebo Garden

```bash
sudo apt install software-properties-common
sudo add-apt-repository ppa:ignitionrobotics/garden
sudo apt-get update
sudo apt-get install gz-garden
```

## Install Additional Tools

### Install Ollama for Local LLMs

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Install vLLM (Alternative LLM Framework)

```bash
pip3 install vllm
```

### Install Additional Python Libraries

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers accelerate
pip3 install openai-whisper
pip3 install opencv-python
pip3 install numpy scipy matplotlib
```

## Create Workspace Structure

### ROS 2 Workspace

```bash
mkdir -p ~/physical_ai_ws/src
cd ~/physical_ai_ws
colcon build --symlink-install
echo "source ~/physical_ai_ws/install/setup.bash" >> ~/.bashrc
source ~/physical_ai_ws/install/setup.bash
```

## Hardware Recommendations

### Robot Platforms

For hands-on experience, we recommend one of these platforms:

- **Unitree Go2**: Quadruped robot with excellent ROS 2 support
- **Hiwonder TonyPi Pro**: Bipedal robot for humanoid form factor
- **Unitree G1**: Advanced humanoid robot (for advanced users)

### Sensor Packages

- **Intel RealSense D455**: RGB-D camera for perception
- **Hesai PandarQT**: LiDAR for navigation
- **Logitech C920**: HD webcam for basic vision tasks

## Environment Validation

### Test ROS 2 Installation

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Test basic ROS 2 functionality
ros2 run demo_nodes_cpp talker
```

### Test Isaac Sim (in separate terminal)

```bash
# Activate Isaac Sim environment
source ~/isaac-sim-env/bin/activate

# Launch Isaac Sim
cd ~/isaac-sim
./isaac-sim.sh
```

### Test Gazebo (in separate terminal)

```bash
gz sim
```

## Next Steps

Once your local environment is set up, proceed to:

1. [Hardware Recommendations and Cost Analysis](./hardware-recommendations.md) - For specific hardware choices
2. [ROS 2 Installation Deep Dive](./ros2-installation.md) - For detailed ROS 2 setup
3. [Isaac Sim Configuration](./isaac-sim-setup.md) - For detailed Isaac Sim setup

Your local development environment is now ready for the Physical AI and Humanoid Robotics course!