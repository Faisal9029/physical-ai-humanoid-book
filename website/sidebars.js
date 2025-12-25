// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro/index',
    {
      type: 'category',
      label: 'Introduction',
      items: ['intro/prerequisites', 'intro/setup-local', 'intro/setup-cloud', 'intro/troubleshooting'],
    },
    {
      type: 'category',
      label: 'Week 1: Environment Setup',
      items: [
        'week-01/index',
        'week-01/hardware-recommendations',
        'week-01/ros2-installation',
        'week-01/isaac-sim-setup',
        'week-01/gazebo-setup',
        'week-01/environment-validation'
      ],
    },
    {
      type: 'category',
      label: 'Week 2: Robot Fundamentals',
      items: [
        'week-02/index',
        'week-02/robot-models',
        'week-02/joint-control',
        'week-02/kinematics',
        'week-02/dynamics'
      ],
    },
    {
      type: 'category',
      label: 'Week 3: ROS 2 Control Systems',
      items: [
        'week-03/index',
        'week-03/controllers',
        'week-03/joint-state-publisher',
        'week-03/robot-state-publisher',
        'week-03/control-interfaces'
      ],
    },
    {
      type: 'category',
      label: 'Week 4: Simulation Fundamentals',
      items: [
        'week-04/index',
        'week-04/gazebo-basics',
        'week-04/isaac-sim-basics',
        'week-04/physics-engines',
        'week-04/environment-modeling'
      ],
    },
    {
      type: 'category',
      label: 'Week 5: Perception Systems',
      items: [
        'week-05/index',
        'week-05/sensor-integration',
        'week-05/camera-setup',
        'week-05/lidar-integration',
        'week-05/sensor-fusion'
      ],
    },
    {
      type: 'category',
      label: 'Week 6: Isaac Perception',
      items: [
        'week-06/index',
        'week-06/isaac-ros-pipelines',
        'week-06/object-detection',
        'week-06/pose-estimation',
        'week-06/scene-understanding'
      ],
    },
    {
      type: 'category',
      label: 'Week 7: Navigation Systems',
      items: [
        'week-07/index',
        'week-07/mapping',
        'week-07/path-planning',
        'week-07/localization',
        'week-07/navigation-stack'
      ],
    },
    {
      type: 'category',
      label: 'Week 8: LLM Integration',
      items: [
        'week-08/index',
        'week-08/local-llm-setup',
        'week-08/whisper-integration',
        'week-08/vision-language-models',
        'week-08/action-planning'
      ],
    },
    {
      type: 'category',
      label: 'Week 9: Vision-Language-Action Pipelines',
      items: [
        'week-09/index',
        'week-09/vla-concepts',
        'week-09/multimodal-inputs',
        'week-09/action-translation',
        'week-09/feedback-loops'
      ],
    },
    {
      type: 'category',
      label: 'Week 10: Advanced Control',
      items: [
        'week-10/index',
        'week-10/trajectory-generation',
        'week-10/balance-control',
        'week-10/gait-planning',
        'week-10/whole-body-control'
      ],
    },
    {
      type: 'category',
      label: 'Week 11: Sim-to-Real Transfer',
      items: [
        'week-11/index',
        'week-11/domain-randomization',
        'week-11/sim-to-real-challenges',
        'week-11/robot-calibration',
        'week-11/safety-considerations'
      ],
    },
    {
      type: 'category',
      label: 'Week 12: Human-Robot Interaction',
      items: [
        'week-12/index',
        'week-12/voice-interaction',
        'week-12/gesture-recognition',
        'week-12/intention-interpretation',
        'week-12/social-robotics'
      ],
    },
    {
      type: 'category',
      label: 'Week 13: Capstone Project',
      items: [
        'week-13/index',
        'week-13/project-specification',
        'week-13/implementation-guide',
        'week-13/testing-verification',
        'week-13/deployment'
      ],
    },
    {
      type: 'category',
      label: 'Appendices',
      items: [
        'appendices/hardware-specs',
        'appendices/api-reference',
        'appendices/troubleshooting',
        'appendices/further-reading',
        'appendices/glossary'
      ],
    },
    {
      type: 'category',
      label: 'Tutorials',
      items: [
        'tutorials/quick-start',
        'tutorials/ros2-basics',
        'tutorials/isaac-sim-tutorial'
      ],
    },
  ],
};

export default sidebars;