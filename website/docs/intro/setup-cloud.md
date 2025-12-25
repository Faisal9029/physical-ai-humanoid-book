---
sidebar_position: 4
---

# Cloud Development Environment Setup

This guide will walk you through setting up a Physical AI development environment using cloud-based GPU instances. This approach is ideal if you don't have access to compatible local hardware.

## Cloud Provider Options

### AWS (Recommended)

For optimal performance with Isaac Sim and perception tasks, we recommend:

- **Instance Type**: `g5.xlarge` or `g6e.xlarge` with NVIDIA T4 or L4 GPU
- **Alternative**: `p3.2xlarge` with NVIDIA V100 GPU for higher performance
- **Storage**: 100GB+ EBS volume (gp3 recommended)
- **OS**: Ubuntu 22.04 LTS AMI

### Alternative Cloud Providers

- **Azure**: NCv3 series with NVIDIA V100 GPUs
- **Google Cloud**: A2 series with NVIDIA A100 GPUs
- **Lambda Labs**: GPU cloud instances with pre-configured environments

## AWS Setup Guide

### Launch EC2 Instance

1. Navigate to AWS EC2 Console
2. Launch Instance
3. Choose AMI: Ubuntu 22.04 LTS
4. Instance Type: g5.xlarge (or larger based on needs)
5. Storage: 100GB+ (gp3 recommended)
6. Security Group: Enable SSH (port 22), HTTP (port 80), HTTPS (port 443)
7. Review and Launch

### Connect to Your Instance

```bash
ssh -i your-key-pair.pem ubuntu@your-instance-ip
```

## System Preparation

### Update System

```bash
sudo apt update && sudo apt upgrade -y
```

### Install Essential Tools

```bash
sudo apt install -y build-essential cmake git python3-pip python3-dev
sudo apt install -y curl wget vim htop screen tmux
```

## Install NVIDIA GPU Drivers

### Check GPU Availability

```bash
lspci | grep -i nvidia
```

### Install NVIDIA Drivers

```bash
sudo apt install -y nvidia-driver-535-server
sudo reboot
```

### Verify GPU Installation

```bash
nvidia-smi
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

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Install Gazebo (Cloud-Compatible Alternative)

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

## Remote Development Setup

### Install VS Code Server

```bash
# Install code-server for browser-based development
curl -fsSL https://code-server.dev/install.sh | sh
code-server --bind-addr 0.0.0.0:8080
```

### Set Up SSH Tunneling

For accessing GUI applications and simulations:

```bash
# Local machine: Create SSH tunnel
ssh -L 8080:localhost:8080 -L 8000:localhost:8000 -i your-key-pair.pem ubuntu@your-instance-ip
```

## Performance Optimization for Cloud

### GPU Memory Management

```bash
# Check GPU memory usage
nvidia-smi

# Set GPU to persistence mode
sudo nvidia-smi -pm 1
```

### Network Optimization

```bash
# Optimize network settings for data transfer
echo 'net.core.rmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## Cost Management

### Instance Scheduling

- Use Spot Instances for development work (up to 70% cost savings)
- Stop instances when not in use
- Consider Reserved Instances for long-term projects

### Monitoring

```bash
# Install monitoring tools
sudo apt install -y awscli
aws configure
```

## Environment Validation

### Test ROS 2 Installation

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Test basic ROS 2 functionality
ros2 run demo_nodes_cpp talker
```

### Test Gazebo

```bash
gz sim
```

### Test GPU Acceleration

```bash
# Verify CUDA
nvidia-smi
nvcc --version
```

## Latency Considerations

> **⚠️ Latency Trap Warning**: Cloud-based robotics development may introduce network latency that affects real-time simulation and robot control. For latency-sensitive applications:
> - Use GPU instances in regions closest to your location
> - Consider hybrid approach (cloud simulation, local control)
> - Test performance-critical applications on local hardware when possible

## Next Steps

Once your cloud environment is set up, proceed to:

1. [Hardware Recommendations and Cost Analysis](./hardware-recommendations.md) - For cost comparison between local and cloud
2. [ROS 2 Installation Deep Dive](./ros2-installation.md) - For detailed ROS 2 setup
3. [Gazebo Configuration](./gazebo-setup.md) - For detailed Gazebo setup

Your cloud-based development environment is now ready for the Physical AI and Humanoid Robotics course!