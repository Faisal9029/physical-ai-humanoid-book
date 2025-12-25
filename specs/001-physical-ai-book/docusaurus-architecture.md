# Docusaurus Architecture Sketch: Physical AI and Humanoid Robotics Book

## Overall Architecture

```
website/
├── docs/
│   ├── intro/                           # Introduction and setup
│   │   ├── index.md                     # Welcome and overview
│   │   ├── prerequisites.md             # Prerequisites and requirements
│   │   ├── setup-local.md               # Local environment setup
│   │   ├── setup-cloud.md               # Cloud environment setup
│   │   └── troubleshooting.md           # Common issues and solutions
│   ├── week-01/                         # Week 1: Environment Setup
│   │   ├── index.md                     # Week overview
│   │   ├── hardware-recommendations.md  # Hardware options and cost analysis
│   │   ├── ros2-installation.md         # ROS 2 installation and configuration
│   │   ├── isaac-sim-setup.md           # NVIDIA Isaac Sim setup
│   │   ├── gazebo-setup.md              # Gazebo setup (alternative)
│   │   └── environment-validation.md    # Environment validation tests
│   ├── week-02/                         # Week 2: Robot Fundamentals
│   │   ├── index.md                     # Week overview
│   │   ├── robot-models.md              # Robot models and URDF
│   │   ├── joint-control.md             # Joint control concepts
│   │   ├── kinematics.md                # Forward and inverse kinematics
│   │   └── dynamics.md                  # Robot dynamics basics
│   ├── week-03/                         # Week 3: ROS 2 Control Systems
│   │   ├── index.md                     # Week overview
│   │   ├── controllers.md               # ROS 2 controllers
│   │   ├── joint-state-publisher.md     # Joint state publishing
│   │   ├── robot-state-publisher.md     # Robot state publishing
│   │   └── control-interfaces.md        # Control interfaces
│   ├── week-04/                         # Week 4: Simulation Fundamentals
│   │   ├── index.md                     # Week overview
│   │   ├── gazebo-basics.md             # Gazebo basics
│   │   ├── isaac-sim-basics.md          # Isaac Sim basics
│   │   ├── physics-engines.md           # Physics engines comparison
│   │   └── environment-modeling.md      # Environment modeling
│   ├── week-05/                         # Week 5: Perception Systems
│   │   ├── index.md                     # Week overview
│   │   ├── sensor-integration.md        # Sensor integration
│   │   ├── camera-setup.md              # Camera setup and calibration
│   │   ├── lidar-integration.md         # LiDAR integration
│   │   └── sensor-fusion.md             # Sensor fusion basics
│   ├── week-06/                         # Week 6: Isaac Perception
│   │   ├── index.md                     # Week overview
│   │   ├── isaac-ros-pipelines.md       # Isaac ROS perception pipelines
│   │   ├── object-detection.md          # Object detection in Isaac
│   │   ├── pose-estimation.md           # Pose estimation
│   │   └── scene-understanding.md       # Scene understanding
│   ├── week-07/                         # Week 7: Navigation Systems
│   │   ├── index.md                     # Week overview
│   │   ├── mapping.md                   # SLAM and mapping
│   │   ├── path-planning.md             # Path planning algorithms
│   │   ├── localization.md              # Robot localization
│   │   └── navigation-stack.md          # ROS navigation stack
│   ├── week-08/                         # Week 8: LLM Integration
│   │   ├── index.md                     # Week overview
│   │   ├── local-llm-setup.md           # Local LLM setup (Ollama/vLLM)
│   │   ├── whisper-integration.md       # Whisper for speech processing
│   │   ├── vision-language-models.md    # Vision-language models
│   │   └── action-planning.md           # Action planning with LLMs
│   ├── week-09/                         # Week 9: Vision-Language-Action
│   │   ├── index.md                     # Week overview
│   │   ├── vla-concepts.md              # VLA pipeline concepts
│   │   ├── multimodal-inputs.md         # Multimodal inputs processing
│   │   ├── action-translation.md        # Translating to robot actions
│   │   └── feedback-loops.md            # Perception-action feedback
│   ├── week-10/                         # Week 10: Advanced Control
│   │   ├── index.md                     # Week overview
│   │   ├── trajectory-generation.md     # Trajectory generation
│   │   ├── balance-control.md           # Balance control for bipeds
│   │   ├── gait-planning.md             # Gait planning
│   │   └── whole-body-control.md        # Whole body control
│   ├── week-11/                         # Week 11: Sim-to-Real Transfer
│   │   ├── index.md                     # Week overview
│   │   ├── domain-randomization.md      # Domain randomization
│   │   ├── sim-to-real-challenges.md    # Sim-to-real transfer challenges
│   │   ├── robot-calibration.md         # Robot calibration
│   │   └── safety-considerations.md     # Safety considerations
│   ├── week-12/                         # Week 12: Human-Robot Interaction
│   │   ├── index.md                     # Week overview
│   │   ├── voice-interaction.md         # Voice interaction
│   │   ├── gesture-recognition.md       # Gesture recognition
│   │   ├── intention-interpretation.md  # Intention interpretation
│   │   └── social-robotics.md           # Social robotics basics
│   ├── week-13/                         # Week 13: Capstone Project
│   │   ├── index.md                     # Capstone overview
│   │   ├── project-specification.md     # Project requirements
│   │   ├── implementation-guide.md      # Implementation guide
│   │   ├── testing-verification.md      # Testing and verification
│   │   └── deployment.md                # Deployment considerations
│   ├── appendices/
│   │   ├── hardware-specs.md            # Hardware specifications
│   │   ├── api-reference.md             # API references
│   │   ├── troubleshooting.md           # Troubleshooting guide
│   │   ├── further-reading.md           # Further reading
│   │   └── glossary.md                  # Glossary of terms
│   └── tutorials/
│       ├── quick-start.md               # Quick start tutorial
│       ├── ros2-basics.md               # ROS 2 basics tutorial
│       └── isaac-sim-tutorial.md        # Isaac Sim tutorial
├── src/
│   ├── components/                      # Custom React components
│   │   ├── CodeRunner/                  # Interactive code runner
│   │   ├── HardwareSpec/                # Hardware specification tables
│   │   └── CostCalculator/              # Cost calculation tool
│   └── pages/                          # Custom pages
│       ├── index.js                    # Home page
│       └── simulator.js                # Simulator integration page
├── static/                            # Static assets
│   ├── img/                           # Images and diagrams
│   ├── videos/                        # Video tutorials
│   └── models/                        # 3D models and URDF files
├── docusaurus.config.js               # Docusaurus configuration
├── sidebars.js                        # Sidebar navigation
└── package.json                       # Dependencies
```

## Component Architecture

### Navigation Structure
- **Top-level navigation**: Introduction, Weekly Modules (1-13), Appendices, Tutorials
- **Sidebar structure**: Each week has 4-5 sub-topics following the learning progression
- **Progress tracking**: Weekly progress indicators and milestone achievements

### Key Features
1. **Hardware specification tables** with cost analysis and compatibility matrices
2. **Interactive code blocks** with copy/run functionality
3. **Embedded simulation viewers** for visualization
4. **Cost calculators** for hardware and cloud setups
5. **Cross-referencing system** between related concepts
6. **API reference integration** for ROS 2, Isaac Sim, and Gazebo

### Content Types
- **Conceptual explanations** with diagrams and examples
- **Step-by-step tutorials** with expected outputs
- **Code examples** with syntax highlighting and execution context
- **Hardware recommendations** with spec comparisons
- **Troubleshooting guides** with common solutions
- **Research concurrent integration** with citations and references

### Technical Implementation
- **Markdown + MDX** for rich content integration
- **Docusaurus v3** with modern React components
- **GitHub Pages deployment** with proper asset optimization
- **Mobile-responsive design** for accessibility
- **Search functionality** with custom indexing
- **Version control integration** for content updates