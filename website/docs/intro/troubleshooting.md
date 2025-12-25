---
sidebar_position: 5
---

# Troubleshooting

This guide covers common issues you may encounter during setup and development of your Physical AI and Humanoid Robotics environment.

## ROS 2 Installation Issues

### Package Installation Failures

**Problem**: ROS 2 packages fail to install with repository errors.

**Solution**:
```bash
# Clean package cache
sudo apt clean
sudo apt update

# Verify ROS 2 repository key
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /tmp/ros.key
sudo apt-key add /tmp/ros.key

# Or re-add the repository
sudo add-apt-repository --remove ppa:ros/noetic
sudo add-apt-repository ppa:ros/humble
sudo apt update
```

### ROS 2 Environment Not Loading

**Problem**: ROS 2 commands not found after installation.

**Solution**:
```bash
# Manually source ROS 2
source /opt/ros/humble/setup.bash

# Add to bashrc if missing
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## GPU and Isaac Sim Issues

### NVIDIA Driver Issues

**Problem**: `nvidia-smi` command not found or GPU not detected.

**Solution**:
```bash
# Check if NVIDIA drivers are installed
lspci | grep -i nvidia

# Install or reinstall drivers
sudo apt install nvidia-driver-535
sudo reboot
```

### Isaac Sim Launch Failures

**Problem**: Isaac Sim fails to launch with GPU errors.

**Solution**:
```bash
# Check GPU status
nvidia-smi

# Verify CUDA installation
nvcc --version

# Check Isaac Sim dependencies
python3 -c "import torch; print(torch.cuda.is_available())"
```

## Gazebo Issues

### Gazebo Not Starting

**Problem**: `gz sim` command fails to launch.

**Solution**:
```bash
# Check graphics drivers
glxinfo | grep "OpenGL renderer"

# Set environment variables
export MESA_GL_VERSION_OVERRIDE=3.3
export GZ_SIM_SYSTEM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gazebo-11/plugins
```

### Rendering Problems in Gazebo

**Problem**: Poor performance or visual artifacts in Gazebo.

**Solution**:
```bash
# For cloud instances, try software rendering
export MESA_GL_VERSION_OVERRIDE=3.3
export LIBGL_ALWAYS_SOFTWARE=1
```

## Python and Package Issues

### Python Package Installation Failures

**Problem**: Python packages fail to install with compilation errors.

**Solution**:
```bash
# Upgrade pip
pip3 install --upgrade pip

# Install build dependencies
sudo apt install build-essential python3-dev

# Install packages individually if needed
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```

### Virtual Environment Issues

**Problem**: Python packages not available in expected environment.

**Solution**:
```bash
# Activate the correct environment
source ~/isaac-sim-env/bin/activate

# Verify environment
which python3
pip3 list
```

## Network and Remote Access Issues

### SSH Connection Problems

**Problem**: Unable to connect to cloud instance via SSH.

**Solution**:
- Verify security group allows SSH (port 22)
- Check that your key pair has correct permissions: `chmod 400 your-key.pem`
- Verify the instance is running and has a public IP

### Port Forwarding Issues

**Problem**: Cannot access services running on cloud instance.

**Solution**:
```bash
# Verify security group allows the port
# Check if service is running on correct interface
netstat -tlnp | grep :port_number

# For local testing, bind to all interfaces
# Instead of 127.0.0.1, use 0.0.0.0
```

## Common Error Messages

### "No module named 'ros2'"

**Cause**: ROS 2 environment not sourced.

**Solution**:
```bash
source /opt/ros/humble/setup.bash
```

### "CUDA error: no kernel image is available for execution"

**Cause**: PyTorch compiled for different CUDA architecture.

**Solution**:
```bash
# Reinstall PyTorch for your GPU architecture
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "Failed to initialize NVML"

**Cause**: NVIDIA drivers not properly installed or GPU not accessible.

**Solution**:
```bash
# Check if NVIDIA drivers are loaded
lsmod | grep nvidia

# Reinstall drivers if needed
sudo apt install --reinstall nvidia-driver-535
sudo reboot
```

## Performance Issues

### Slow Simulation Performance

**Problem**: Simulation runs slowly or with low frame rates.

**Solutions**:
- Reduce simulation complexity (fewer objects, simpler models)
- Lower rendering quality settings
- Check GPU utilization: `nvidia-smi`
- Close unnecessary applications

### High Memory Usage

**Problem**: System runs out of memory during simulation.

**Solutions**:
```bash
# Check memory usage
free -h
htop

# Increase swap space if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Package and Dependency Issues

### Colcon Build Failures

**Problem**: `colcon build` fails with dependency errors.

**Solutions**:
```bash
# Check for missing dependencies
rosdep install --from-paths src --ignore-src -r -y

# Clean build and rebuild
rm -rf build/ install/ log/
colcon build --symlink-install
```

### APT Package Conflicts

**Problem**: APT reports package conflicts during installation.

**Solutions**:
```bash
# Update package lists
sudo apt update

# Try to fix broken packages
sudo apt --fix-broken install

# Clean package cache
sudo apt clean
sudo apt autoremove
```

## Validation Scripts

### Environment Check Script

Create this script to validate your environment:

```bash
#!/bin/bash
# check_environment.sh

echo "=== ROS 2 Check ==="
source /opt/ros/humble/setup.bash
if command -v ros2 &> /dev/null; then
    echo "✓ ROS 2 installed and accessible"
    ros2 --version
else
    echo "✗ ROS 2 not accessible"
fi

echo -e "\n=== GPU Check ==="
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA drivers accessible"
    nvidia-smi
else
    echo "✗ NVIDIA drivers not accessible"
fi

echo -e "\n=== Python Check ==="
python3 -c "import torch; print(f'✓ PyTorch CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "✗ PyTorch not accessible"

echo -e "\n=== Gazebo Check ==="
if command -v gz &> /dev/null; then
    echo "✓ Gazebo installed and accessible"
    gz --version
else
    echo "✗ Gazebo not accessible"
fi
```

## Getting Help

### Community Resources

- **ROS Answers**: https://answers.ros.org/
- **Isaac Sim Forum**: https://forums.developer.nvidia.com/
- **Gazebo Community**: https://community.gazebosim.org/
- **GitHub Issues**: Check and report issues in the course repository

### Support Channels

- Course Discord/Slack community
- Office hours (if available)
- Email support for critical issues

## Next Steps

If you continue to experience issues:

1. Run the validation script to identify specific problems
2. Check the [FAQ](#) for common solutions
3. Search community resources with your specific error message
4. Create a detailed issue report with:
   - Your system configuration
   - Exact error messages
   - Steps to reproduce the issue
   - What you've already tried