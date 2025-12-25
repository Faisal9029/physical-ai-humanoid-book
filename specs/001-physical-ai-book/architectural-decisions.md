# Architectural Decision Records: Physical AI and Humanoid Robotics Book

## Decision 1: Local vs. Cloud Dev Environment

**Status**: Decided
**Date**: 2025-12-21

**Context**:
The development environment choice significantly impacts the learning experience, cost structure, and performance characteristics for readers. Two primary approaches were considered: on-premise high-performance workstation or cloud-based development with edge deployment.

**Options Considered**:
- **Option A**: On-premise RTX workstation (high CapEx, low latency)
  - Pros: Low latency, high performance, no internet dependency, full control
  - Cons: High upfront cost, hardware maintenance, limited scalability
- **Option B**: Cloud-native (AWS g5/g6e + Jetson edge kit; lower upfront cost, latency tradeoff)
  - Pros: Lower upfront cost, scalability, managed infrastructure, pay-per-use
  - Cons: Network latency, ongoing costs, limited control, internet dependency

**Decision**:
Document both paths with cost tables, setup steps, and "Latency Trap" warning. The primary recommendation is to start with cloud-based development for accessibility, with on-premise as an advanced option for performance-critical applications.

**Rationale**:
This approach maximizes accessibility for the target audience (AI/ML developers, robotics students) while providing clear pathways for different budget and performance requirements. The "Latency Trap" warning addresses the critical issue where cloud-based simulation and real-time robotics control can suffer from unacceptable latency.

**Implications**:
- Content must include dual setup guides
- Cost analysis tables required
- Performance benchmarks for both approaches
- Clear guidance on when to choose each path

## Decision 2: Robot Proxy Choice

**Status**: Decided
**Date**: 2025-12-21

**Context**:
Selecting an appropriate robot platform is crucial for the practical exercises and examples throughout the book. The choice affects ROS 2 compatibility, affordability, and educational value.

**Options Considered**:
- **Option A**: Unitree Go2 (quadruped, robust ROS 2 support, affordable)
  - Pros: Excellent ROS 2 support, affordable, robust platform, good for learning
  - Cons: Quadruped instead of humanoid, limited to 4 legs vs human-like form
- **Option B**: Hiwonder TonyPi Pro (bipedal form, limited AI compute)
  - Pros: Bipedal form factor, humanoid appearance, reasonable price
  - Cons: Limited computational power, less mature ROS 2 support
- **Option C**: Unitree G1 (true humanoid, expensive)
  - Pros: True humanoid form, advanced capabilities, cutting-edge platform
  - Cons: Very expensive, limited accessibility, high barrier to entry

**Decision**:
Primary narrative uses Go2 as proxy due to its excellent ROS 2 support and affordability; G1 noted for advanced labs; TonyPi flagged as kinematics-only option.

**Rationale**:
The Go2 provides the best balance of affordability, ROS 2 compatibility, and educational value. Its quadruped nature doesn't significantly impact the learning of core robotics concepts while providing a robust platform for experimentation. Advanced users can graduate to G1 for humanoid-specific challenges.

**Implications**:
- All examples and exercises based on Go2
- G1 mentioned for advanced scenarios
- TonyPi referenced for kinematics-only exercises
- Content remains applicable to other platforms

## Decision 3: LLM Integration Strategy

**Status**: Decided
**Date**: 2025-12-21

**Context**:
The approach to integrating large language models affects API dependency, cost, privacy, and reproducibility of the examples in the book.

**Options Considered**:
- **Option A**: Cloud-based OpenAI Whisper + GPT
  - Pros: Proven technology, managed service, consistent performance
  - Cons: API dependency, ongoing costs, privacy concerns, vendor lock-in
- **Option B**: Local open-weight models (Llama 3, Mistral) via Ollama/vLLM on workstation
  - Pros: No API dependency, privacy control, reproducibility, no vendor lock-in
  - Cons: Higher hardware requirements, setup complexity, performance variation

**Decision**:
Default to open-weight local models to avoid API dependency; OpenAI shown as optional alternative.

**Rationale**:
This decision aligns with the project's open-source ethos and ensures reproducibility for all readers. Local models guarantee that the book's examples will continue to work regardless of API changes or subscription costs, while still providing the option for readers who prefer cloud-based solutions.

**Implications**:
- Primary examples use local models
- Ollama/vLLM setup instructions required
- Cloud alternatives documented as optional
- Hardware requirements for local models specified

## Decision 4: Simulation Stack

**Status**: Decided
**Date**: 2025-12-21

**Context**:
The simulation environment choice affects the quality of visual rendering, physics accuracy, and hardware requirements for readers.

**Options Considered**:
- **Option A**: Gazebo only (lightweight, ROS-native)
  - Pros: Lightweight, ROS 2 native, broad compatibility, lower hardware requirements
  - Cons: Limited visual fidelity, less realistic rendering
- **Option B**: Gazebo + Unity (high-fidelity rendering)
  - Pros: High visual fidelity, realistic rendering, good for perception tasks
  - Cons: Complex setup, Unity license requirements, higher hardware needs
- **Option C**: NVIDIA Isaac Sim (photorealistic, RTX required)
  - Pros: Photorealistic rendering, advanced physics, perception training ready
  - Cons: High hardware requirements (RTX), NVIDIA licensing, complexity

**Decision**:
Isaac Sim as primary; Gazebo for minimal setups; Unity mentioned but not required.

**Rationale**:
Isaac Sim provides the most advanced simulation capabilities needed for the Vision-Language-Action pipelines that are central to the book's focus. However, Gazebo is maintained as a minimal alternative to ensure accessibility for readers with less powerful hardware.

**Implications**:
- Primary examples target Isaac Sim
- Gazebo fallback configurations provided
- Hardware requirements clearly specified
- Perception training examples optimized for Isaac Sim

## Decision 5: Content Structure and Modularity

**Status**: Decided
**Date**: 2025-12-21

**Context**:
The organization of content affects learning progression, modularity, and the ability to adapt the material for different audiences and time constraints.

**Options Considered**:
- **Option A**: Sequential, tightly-coupled modules
  - Pros: Clear progression, builds on previous knowledge
  - Cons: Less flexible, difficult to adapt, single point of failure
- **Option B**: Modular, loosely-coupled approach with cross-references
  - Pros: Flexible, adaptable, can be customized for different audiences
  - Cons: More complex navigation, potential redundancy

**Decision**:
Modular approach with 13-week structure that allows for both sequential and selective learning paths.

**Rationale**:
The modular approach supports different learning styles and time constraints while maintaining clear progression. It allows educators to customize the material for their specific needs while ensuring comprehensive coverage of all essential topics.

**Implications**:
- Each week's content is self-contained
- Clear dependencies documented between modules
- Cross-references for related concepts
- Multiple learning paths supported

## Decision 6: Research-Concurrent Documentation Approach

**Status**: Decided
**Date**: 2025-12-21

**Context**:
The approach to gathering and incorporating current information from official sources affects the accuracy, currency, and maintenance of the documentation.

**Options Considered**:
- **Option A**: Separate research phase before writing
  - Pros: Organized, comprehensive research upfront
  - Cons: Information may become outdated, less efficient
- **Option B**: Research-concurrent approach during writing
  - Pros: Current information, efficient workflow, immediate validation
  - Cons: Potential interruptions, requires discipline

**Decision**:
Research-concurrent approach: pull official docs (ROS 2, Isaac ROS, Docusaurus) during writing, not in a separate phase.

**Rationale**:
This approach ensures the content remains current with the latest official documentation and tools. It also allows for immediate validation of examples and procedures as they are written, reducing the likelihood of outdated or incorrect information.

**Implications**:
- Writers must have access to current documentation
- Examples validated against current tool versions
- Regular updates needed as tools evolve
- APA citation style for all references