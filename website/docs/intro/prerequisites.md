---
sidebar_position: 2
---

# Prerequisites

Before starting this Physical AI and Humanoid Robotics course, you'll need to prepare your development environment. This section outlines the requirements and recommendations for both local and cloud-based development.

## System Requirements

### Local Development (Recommended)

For the best performance with simulation and perception tasks, we recommend:

- **Operating System**: Ubuntu 22.04 LTS (fully supported)
- **CPU**: Multi-core processor (Intel i7 or AMD Ryzen 7 recommended)
- **RAM**: 32GB minimum, 64GB recommended
- **GPU**: NVIDIA RTX 4080/4090 or RTX A4000/A5000 series for optimal Isaac Sim performance
- **Storage**: 100GB+ free space for ROS 2, Isaac Sim, and project files
- **Internet**: Stable connection for initial setup and package downloads

### Cloud Development

For those without high-end local hardware:

- **AWS**: g5.xlarge or g6e.xlarge instances with attached GPU
- **Alternative**: Cloud providers with NVIDIA T4/V100 GPU instances
- **Minimum specs**: 16GB RAM, 8 vCPUs, GPU with 16GB+ VRAM

## Software Requirements

### Essential Software

- **Git**: Version control system
- **Docker**: Containerization (if using containerized development)
- **Python 3.10+**: For ROS 2 Humble Hawksbill
- **Node.js 18+**: For Docusaurus documentation
- **NVIDIA Drivers**: Latest drivers for GPU acceleration (if using local development)

### Development Tools

- **IDE**: VS Code with ROS extensions recommended
- **Terminal**: Bash or Zsh for command-line operations
- **Browser**: Chrome or Firefox for simulation visualization

## Knowledge Prerequisites

### Required Background

- **Programming**: Basic Python programming skills
- **Linux**: Comfortable with command-line operations
- **Mathematics**: Basic understanding of linear algebra and calculus
- **Robotics**: Fundamental concepts (optional but helpful)

### Helpful Background

- **ROS Basics**: Understanding of ROS concepts (will be covered in Week 2)
- **Machine Learning**: Basic understanding of neural networks
- **Computer Vision**: Basic image processing concepts

## Setup Verification

Before proceeding to the environment setup, verify that you have:

1. **Git installed**:
   ```bash
   git --version
   ```

2. **Python 3.10+ installed**:
   ```bash
   python3 --version
   ```

3. **Sufficient disk space**: At least 100GB free space

4. **Internet access**: Stable connection for downloading large packages

## Next Steps

Once you've verified your prerequisites, choose your development approach:

- [Local Setup](./setup-local.md) - For those with compatible hardware
- [Cloud Setup](./setup-cloud.md) - For those using cloud-based development

Both approaches will lead to the same development environment and capabilities.