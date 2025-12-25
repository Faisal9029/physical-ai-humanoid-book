---
sidebar_position: 6
---

# Environment Validation

This guide provides comprehensive validation procedures to ensure your Physical AI development environment is properly configured and ready for the course. By the end of this validation, you should have a fully functional development setup.

## Overview

This validation ensures that all components of your Physical AI development environment are working correctly. We'll test:

- ROS 2 installation and functionality
- Isaac Sim (primary simulation environment)
- Gazebo (alternative simulation environment)
- AI integration tools (Ollama/local LLMs)
- Overall system performance and compatibility

## Prerequisites

Before running validation tests, ensure you have:

- Completed all previous setup sections (ROS 2, Isaac Sim, Gazebo)
- Properly sourced all environment variables
- At least 4GB free RAM available
- Stable internet connection for any missing package downloads

## Validation Script

### Automated Validation Script

Create a comprehensive validation script to test your environment:

```bash
# Create validation script
cat > ~/validate_environment.sh << 'EOF'
#!/bin/bash

echo "==========================================="
echo "Physical AI Development Environment Validation"
echo "==========================================="
echo

# Function to run a test and report result
run_test() {
    local test_name="$1"
    local command="$2"
    local expected="$3"

    echo -n "Testing: $test_name ... "

    if eval "$command" > /dev/null 2>&1; then
        if [ -z "$expected" ] || [ "$expected" = "success" ]; then
            echo "✓ PASS"
            return 0
        else
            echo "✗ FAIL (command succeeded but expected failure)"
            return 1
        fi
    else
        if [ "$expected" = "failure" ]; then
            echo "✓ PASS (expected failure)"
            return 0
        else
            echo "✗ FAIL"
            return 1
        fi
    fi
}

# Source environments
source /opt/ros/humble/setup.bash 2>/dev/null
source ~/physical_ai_ws/install/setup.bash 2>/dev/null
source ~/isaac_ws/install/setup.bash 2>/dev/null

total_tests=0
passed_tests=0

# Test 1: Check ROS 2 installation
((total_tests++))
if run_test "ROS 2 Installation" "ros2 --version"; then
    ((passed_tests++))
fi

# Test 2: Check basic ROS 2 functionality
((total_tests++))
if run_test "ROS 2 Core Services" "ros2 node list" "success"; then
    ((passed_tests++))
fi

# Test 3: Check GPU availability
((total_tests++))
if run_test "NVIDIA GPU Available" "nvidia-smi" "success"; then
    ((passed_tests++))
fi

# Test 4: Check CUDA availability
((total_tests++))
if run_test "CUDA Installation" "nvcc --version" "success"; then
    ((passed_tests++))
fi

# Test 5: Check Python environment
((total_tests++))
if run_test "Python 3.10+" "python3 -c 'import sys; assert sys.version_info >= (3, 10)'" "success"; then
    ((passed_tests++))
fi

# Test 6: Check PyTorch with CUDA
((total_tests++))
if run_test "PyTorch CUDA Support" "python3 -c 'import torch; assert torch.cuda.is_available()'" "success"; then
    ((passed_tests++))
fi

# Test 7: Check Ollama availability
((total_tests++))
if run_test "Ollama Service" "ollama --version" "success"; then
    ((passed_tests++))
fi

# Test 8: Check Isaac Sim Python modules (if using virtual env)
if [ -d ~/isaac-sim-env ]; then
    ((total_tests++))
    if run_test "Isaac Sim Python Env" "source ~/isaac-sim-env/bin/activate && python -c 'import omni'" "success"; then
        ((passed_tests++))
    fi
fi

# Test 9: Check Gazebo installation
((total_tests++))
if run_test "Gazebo Installation" "gz --version" "success"; then
    ((passed_tests++))
fi

# Test 10: Check Git availability
((total_tests++))
if run_test "Git Installation" "git --version" "success"; then
    ((passed_tests++))
fi

# Test 11: Check workspace build
((total_tests++))
if run_test "ROS 2 Workspace" "cd ~/physical_ai_ws && source install/setup.bash && ros2 pkg list | grep demo_nodes_cpp" "success"; then
    ((passed_tests++))
fi

# Test 12: Check Isaac ROS packages (if installed)
if [ -d ~/isaac_ws ]; then
    ((total_tests++))
    if run_test "Isaac ROS Packages" "cd ~/isaac_ws && source install/setup.bash && ros2 pkg list | grep isaac_ros" "success"; then
        ((passed_tests++))
    fi
fi

# Test 13: Check available disk space (minimum 10GB)
((total_tests++))
available_space=$(df -h $HOME | awk 'NR==2 {print $4}' | sed 's/[^0-9]*//g')
if [ "$available_space" -gt 10 ]; then
    echo "Testing: Available Disk Space (10GB+) ... ✓ PASS"
    ((passed_tests++))
else
    echo "Testing: Available Disk Space (10GB+) ... ✗ FAIL"
fi
((total_tests++))

# Test 14: Check system memory (minimum 16GB)
total_memory=$(free -g | awk 'NR==2 {print $2}')
if [ "$total_memory" -ge 16 ]; then
    echo "Testing: System Memory (16GB+) ... ✓ PASS"
    ((passed_tests++))
else
    echo "Testing: System Memory (16GB+) ... ⚠ SKIP (only $total_memory GB available)"
fi
((total_tests++))

# Summary
echo
echo "==========================================="
echo "Validation Summary"
echo "==========================================="
echo "Total Tests: $total_tests"
echo "Passed: $passed_tests"
echo "Failed: $((total_tests - passed_tests))"
echo

if [ $passed_tests -eq $total_tests ]; then
    echo "🎉 All validation tests passed!"
    echo "Your Physical AI development environment is ready for the course."
    exit 0
else
    failed_count=$((total_tests - passed_tests))
    echo "❌ $failed_count validation test(s) failed."
    echo "Please review the failed tests and resolve the issues before continuing."
    echo "Check the troubleshooting guide for common solutions."
    exit 1
fi
EOF

chmod +x ~/validate_environment.sh
```

## Manual Validation Steps

### 1. ROS 2 Validation

#### Test ROS 2 Installation

```bash
# Check ROS 2 version
ros2 --version

# Check ROS 2 environment
printenv | grep ROS
```

#### Test ROS 2 Core Functionality

Open two terminal windows:

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

### 2. GPU and Isaac Sim Validation

#### Test GPU Availability

```bash
# Check GPU information
nvidia-smi

# Test CUDA
nvcc --version

# Test PyTorch with CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"
```

#### Test Isaac Sim (if installed)

```bash
# Test Isaac Sim Python modules
source ~/isaac-sim-env/bin/activate  # if using virtual env
python3 -c "import omni; print('Isaac Sim modules loaded successfully')"
```

### 3. Gazebo Validation

#### Test Gazebo Installation

```bash
# Check Gazebo version
gz --version

# Try launching Gazebo (headless first)
gz sim --version
```

### 4. AI Tools Validation

#### Test Ollama

```bash
# Check Ollama status
systemctl status ollama

# Test Ollama functionality
ollama --version

# List available models (should return empty list if no models installed yet)
ollama list
```

#### Test Python AI Libraries

```bash
# Test PyTorch and CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Test Transformers
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Test OpenCV
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

## Performance Validation

### System Performance Test

Run a basic performance test to ensure your system can handle the course requirements:

```bash
# Test CPU performance
echo "Testing CPU performance..."
timeout 10 stress-ng --cpu 4 --timeout 10s

# Test memory availability
free -h

# Check available disk space
df -h $HOME
```

### GPU Performance Test

```bash
# Check GPU utilization during basic operations
nvidia-smi dmon -s u -d 1 &

# Run a simple PyTorch operation
python3 -c "
import torch
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print('GPU matrix multiplication successful')
else:
    print('CUDA not available')
"
```

## Troubleshooting Failed Tests

### ROS 2 Issues

If ROS 2 tests fail:

```bash
# Re-source the environment
source /opt/ros/humble/setup.bash
source ~/physical_ai_ws/install/setup.bash

# Check for missing packages
sudo apt update
sudo apt install ros-humble-desktop-full

# Rebuild workspace
cd ~/physical_ai_ws
rm -rf build/ install/ log/
colcon build --symlink-install
```

### GPU/Isaac Sim Issues

If GPU or Isaac Sim tests fail:

```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA installation
nvcc --version

# Check Isaac Sim installation
ls -la ~/isaac-sim/
```

### Gazebo Issues

If Gazebo tests fail:

```bash
# Check Gazebo installation
dpkg -l | grep gz

# Check environment variables
echo $GAZEBO_MODEL_PATH
echo $GAZEBO_PLUGIN_PATH
```

## Validation Checklist

### Essential Components

- [ ] ROS 2 Humble Hawksbill installed and functional
- [ ] NVIDIA GPU drivers properly installed
- [ ] CUDA toolkit available and functional
- [ ] Isaac Sim installed and accessible
- [ ] Gazebo installed and functional
- [ ] Ollama service running
- [ ] Python 3.10+ with required libraries
- [ ] Development workspace built successfully

### Performance Requirements

- [ ] At least 16GB system RAM (recommended)
- [ ] At least 50GB free disk space
- [ ] NVIDIA GPU with 8GB+ VRAM (recommended)
- [ ] Stable internet connection
- [ ] CPU with 4+ cores

### ROS 2 Functionality

- [ ] ROS 2 talker/listener test passes
- [ ] Workspace builds without errors
- [ ] Isaac ROS packages available (if installed)
- [ ] Gazebo ROS 2 bridge packages available

## Time-Based Validation

According to the course requirements, environment setup should take under 4 hours. Let's time your validation:

```bash
# Record validation start time
echo "Environment validation started at: $(date)"

# Run the validation script
time ~/validate_environment.sh

# Record validation end time
echo "Environment validation completed at: $(date)"
```

## Expected Outcomes

### Success Criteria

For the environment validation to be considered successful:

- All essential component tests pass (ROS 2, GPU, Isaac Sim/Gazebo)
- At least 80% of validation tests pass
- System meets minimum performance requirements
- No critical errors during validation

### Acceptable Minor Issues

These issues don't prevent course progression but should be noted:

- Slower than expected simulation performance (if using cloud)
- Missing optional packages that aren't required for core functionality
- Minor environment variable configuration issues

## Next Steps

### If Validation Passes

If your environment validation passes successfully:

1. Continue to [Week 2: Robot Fundamentals](../week-02/index.md)
2. Your development environment is ready for the course
3. Consider installing additional models for Ollama:
   ```bash
   ollama pull llama3
   ollama pull mistral
   ```

### If Validation Fails

If validation fails:

1. Review the failed tests and their error messages
2. Check the [troubleshooting guide](../intro/troubleshooting.md)
3. Re-run the validation after fixing issues
4. Seek help in the course community if needed

## Performance Benchmarks

### Expected Performance Metrics

After validation, you should observe:

- ROS 2 talker/listener: 10+ messages per second
- PyTorch CUDA: Available and functional
- GPU memory: 8GB+ available for Isaac Sim
- System memory: 16GB+ available for simulation

### Performance Optimization

If performance is below expectations:

1. Close unnecessary applications
2. Increase swap space if needed
3. Optimize GPU settings
4. Consider upgrading hardware if using cloud instances

## Final Verification

Once all tests pass, run one final comprehensive check:

```bash
# Source all environments
source /opt/ros/humble/setup.bash
source ~/physical_ai_ws/install/setup.bash
if [ -d ~/isaac_ws ]; then
    source ~/isaac_ws/install/setup.bash
fi

# Final check of all components
echo "ROS 2 Version: $(ros2 --version)"
echo "GPU Available: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "Gazebo Version: $(gz --version)"
echo "Ollama Version: $(ollama --version)"
echo "Python: $(python3 --version)"
echo "CUDA Available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

echo
echo "🎉 Your Physical AI development environment is validated and ready!"
echo "Proceed to Week 2 to begin learning about robot fundamentals."
```

Your environment validation is now complete! You're ready to begin the Physical AI and Humanoid Robotics course.