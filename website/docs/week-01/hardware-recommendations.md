---
sidebar_position: 2
---

# Hardware Recommendations and Cost Analysis

This section provides detailed hardware recommendations for your Physical AI development environment, including cost analysis and performance considerations for both local and cloud-based setups.

## Local Development Hardware

### CPU Requirements

For optimal performance in robotics simulation and AI processing:

- **Minimum**: Intel i7-10700K or AMD Ryzen 7 3700X
- **Recommended**: Intel i9-12900K or AMD Ryzen 9 5900X
- **High-end**: Intel i9-13900K or AMD Ryzen 9 7950X

**Cost Analysis**:
- Minimum: $300-400
- Recommended: $500-600
- High-end: $700-800

### GPU Requirements (Critical for Isaac Sim)

For NVIDIA Isaac Sim and perception tasks:

- **Minimum**: NVIDIA RTX 3070 (8GB VRAM)
- **Recommended**: NVIDIA RTX 4080 (16GB VRAM) or RTX A4000 (16GB VRAM)
- **High-end**: NVIDIA RTX 4090 (24GB VRAM) or RTX A5000 (24GB VRAM)

**Cost Analysis**:
- RTX 3070: $400-500 (used)
- RTX 4080: $900-1,200
- RTX 4090: $1,600-2,000
- RTX A4000: $1,000-1,200
- RTX A5000: $2,500-3,000

### RAM Requirements

- **Minimum**: 32GB DDR4-3200
- **Recommended**: 64GB DDR4-3200 or DDR5-4800
- **High-end**: 128GB DDR4-3200 or DDR5-4800

**Cost Analysis**:
- 32GB: $100-150
- 64GB: $200-300
- 128GB: $400-600

### Storage Requirements

- **SSD**: 1TB NVMe M.2 (minimum) for OS and development
- **Additional**: 2TB HDD or additional SSD for datasets and projects

**Cost Analysis**:
- 1TB NVMe: $100-150
- 2TB Additional: $150-200

### Total Local Setup Costs

| Configuration | CPU | GPU | RAM | Storage | Total |
|---------------|-----|-----|-----|---------|-------|
| Minimum | $350 | $450 | $125 | $125 | ~$1,050 |
| Recommended | $550 | $1,050 | $250 | $150 | ~$2,000 |
| High-end | $750 | $1,800 | $500 | $250 | ~$3,300 |

## Cloud Development Costs

### AWS Options

#### g5.xlarge (NVIDIA T4 GPU)
- 4 vCPUs, 16GB RAM, 16GB GPU VRAM
- **On-demand**: $0.75/hour
- **Spot**: $0.15/hour (70-80% savings)
- **Monthly (full time)**: $540-$600 (on-demand), $108-$120 (spot)

#### g6e.xlarge (NVIDIA L4 GPU)
- 4 vCPUs, 16GB RAM, 24GB GPU VRAM
- **On-demand**: $1.00/hour
- **Spot**: $0.20/hour
- **Monthly (full time)**: $720-$800 (on-demand), $144-$160 (spot)

#### p3.2xlarge (NVIDIA V100 GPU)
- 8 vCPUs, 61GB RAM, 16GB GPU VRAM
- **On-demand**: $3.06/hour
- **Spot**: $0.61/hour
- **Monthly (full time)**: $2,200-$2,400 (on-demand), $440-$480 (spot)

### Cost Optimization Strategies

1. **Use Spot Instances**: Up to 80% savings with interruption handling
2. **Scheduled Scaling**: Stop instances during non-working hours
3. **Reserved Instances**: 30-60% savings for long-term usage (1-3 years)
4. **Right-sizing**: Choose instance types that match your workload

## Robot Platform Recommendations

### Unitree Go2 (Recommended for Learning)

- **Type**: Quadruped robot
- **Price**: ~$2,200-2,700
- **Pros**: Excellent ROS 2 support, robust platform, good documentation
- **Cons**: Quadruped instead of humanoid, limited to 4 legs vs human-like form

### Hiwonder TonyPi Pro (Bipedal Alternative)

- **Type**: Bipedal humanoid robot
- **Price**: ~$1,800-2,200
- **Pros**: Bipedal form factor, humanoid appearance, reasonable price
- **Cons**: Limited computational power, less mature ROS 2 support

### Unitree G1 (Advanced Humanoid)

- **Type**: True humanoid robot
- **Price**: ~$16,000+
- **Pros**: True humanoid form, advanced capabilities, cutting-edge platform
- **Cons**: Very expensive, limited accessibility, high barrier to entry

## Sensor Package Recommendations

### Intel RealSense D455 (Recommended)

- **Type**: RGB-D camera
- **Price**: ~$300-350
- **Features**: Depth sensing, color imaging, IMU
- **ROS 2 Support**: Excellent

### Hesai PandarQT (LiDAR Option)

- **Type**: 4-beam LiDAR
- **Price**: ~$1,200-1,500
- **Features**: 100m range, 40 FPS, low power
- **ROS 2 Support**: Good

### StereoLabs ZED 3 (Alternative Depth)

- **Type**: Stereo camera
- **Price**: ~$400-500
- **Features**: 3D mapping, spatial tracking
- **ROS 2 Support**: Good

## Cloud vs Local Decision Framework

### Choose Local When:

- You have budget for hardware ($2,000-3,300)
- You need consistent, low-latency performance
- You'll use the system for 1+ years
- You have space for the hardware
- You need to connect to physical robots frequently

### Choose Cloud When:

- You want to start quickly without hardware investment
- You need flexible, scalable resources
- You're only using the system short-term (<6 months)
- You have poor local internet for large downloads
- You need to collaborate across different locations

## Performance Benchmarks

### Simulation Performance (Isaac Sim)

- **Local RTX 4080**: 200-300 FPS for basic scenes
- **Local RTX 4090**: 300-500 FPS for complex scenes
- **Cloud T4**: 100-150 FPS for basic scenes
- **Cloud L4**: 150-250 FPS for basic scenes

### AI Inference Performance

- **Local RTX 4080**: 20-30 tokens/sec for Llama 2 (7B)
- **Local RTX 4090**: 30-50 tokens/sec for Llama 2 (7B)
- **Cloud T4**: 10-15 tokens/sec for Llama 2 (7B)
- **Cloud L4**: 15-25 tokens/sec for Llama 2 (7B)

## Budget-Friendly Alternatives

### Used Hardware

- Look for used RTX 3080/3090 for good value
- Refurbished workstations with professional GPUs
- Consider upgrading existing systems incrementally

### Educational Discounts

- Check for NVIDIA, AWS, or other vendor educational discounts
- University partnerships may provide access to cloud credits
- Student pricing for development tools

## ROI Considerations

### For Learning/Research

- Local hardware: Better for long-term learning and experimentation
- Cloud: Better for short-term projects or when budget is tight

### For Production Development

- Local hardware: Better for consistent development and testing
- Cloud: Better for scalable testing and deployment validation

## Next Steps

Based on your budget and requirements, choose your hardware setup:

1. If choosing local development, proceed to purchase recommended components
2. If choosing cloud development, review AWS setup guides
3. Continue to [ROS 2 Installation](./ros2-installation.md) to begin software setup
4. Consider your robot platform choice for later weeks

Remember that the robot platform choice doesn't need to be made immediately - you can start with simulation and add hardware later as needed.