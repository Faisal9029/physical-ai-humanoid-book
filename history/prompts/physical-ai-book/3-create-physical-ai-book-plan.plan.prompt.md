---
id: 3
title: Create Physical AI Book Plan
stage: plan
date: 2025-12-21
surface: agent
model: claude-sonnet-4-5-20250929
feature: physical-ai-book
branch: 001-physical-ai-book
user: user
command: /sp.plan
labels: [planning,architecture,robotics,ai,documentation]
links:
  spec: specs/001-physical-ai-book/spec.md
  plan: specs/001-physical-ai-book/docusaurus-architecture.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/001-physical-ai-book/docusaurus-architecture.md
 - specs/001-physical-ai-book/course-structure.md
 - specs/001-physical-ai-book/architectural-decisions.md
 - specs/001-physical-ai-book/validation-plan.md
tests:
---

## Prompt

```
/sp.plan

Create:
- A modular Docusaurus v3 architecture sketch for a book on Physical AI & Humanoid Robotics
- A section structure that maps directly to the 13-week course modules
- A research-concurrent approach that integrates hardware specs, tool documentation (ROS 2, Isaac Sim, Gazebo), and open-source project examples as writing proceeds
- A quality validation plan tied to the success criteria in the spec (e.g., reproducibility, hardware clarity, VLA pipeline correctness)

Decisions needing documentation:
1. **Local vs. Cloud Dev Environment**
   - Option A: On-premise RTX workstation (high CapEx, low latency)
   - Option B: Cloud-native (AWS g5/g6e + Jetson edge kit; lower upfront cost, latency tradeoff)
   - Decision: Document both paths with cost tables, setup steps, and "Latency Trap" warning

2. **Robot Proxy Choice**
   - Option A: Unitree Go2 (quadruped, robust ROS 2 support, affordable)
   - Option B: Hiwonder TonyPi Pro (bipedal form, limited AI compute)
   - Option C: Unitree G1 (true humanoid, expensive)
   - Decision: Primary narrative uses Go2 as proxy; G1 noted for advanced labs; TonyPi flagged as kinematics-only

3. **LLM Integration Strategy**
   - Option A: Cloud-based OpenAI Whisper + GPT
   - Option B: Local open-weight models (Llama 3, Mistral) via Ollama/vLLM on workstation
   - Decision: Default to open-weight local models to avoid API dependency; OpenAI shown as optional alternative

4. **Simulation Stack**
   - Option A: Gazebo only (lightweight, ROS-native)
   - Option B: Gazebo + Unity (high-fidelity rendering)
   - Option C: NVIDIA Isaac Sim (photorealistic, RTX required)
   - Decision: Isaac Sim as primary; Gazebo for minimal setups; Unity mentioned but not required

Testing strategy:
- **Reproducibility Checks**: Every code block and CLI command must be tested in a clean Ubuntu 22.04 + Docusaurus environment
- **Hardware Validation**: All cost tables and compatibility claims cross-checked against official vendor docs (NVIDIA, Unitree, Intel RealSense)
- **Module Integrity**: Each week's content must independently build as a Docusaurus sidebar section
- **Success Criteria Mapping**: Explicit validation that readers can:
  a) Set up a dev environment (local or cloud)
  b) Run a simulated ROS 2 node in Gazebo/Isaac
  c) Deploy a Whisper-to-ROS action pipeline
  d) Understand Sim-to-Real constraints

Technical details:
- Use research-concurrent approach: pull official docs (ROS 2, Isaac ROS, Docusaurus) during writing, not in a separate phase
- Follow APA citation style as defined in `/sp.constitution` for all hardware specs, papers, and tool references
- Organize content by phases:
  **Research** → Gather toolchain docs, hardware benchmarks, open-source VLA examples
  **Foundation** → Set up Docusaurus structure, sidebar navigation, MDX components for CLI/code
  **Analysis** → Map course weeks to chapters; define hardware/software prerequisites per module
  **Synthesis** → Write narrative content with embedded, executable workflows and cost/performance guidance
```

## Response snapshot

Created comprehensive planning artifacts for the Physical AI and Humanoid Robotics book:

1. **Docusaurus Architecture** (`docusaurus-architecture.md`): Detailed directory structure and component architecture with navigation, content types, and technical implementation details

2. **Course Structure** (`course-structure.md`): 13-week module breakdown with learning objectives, topics, deliverables, and success criteria for each week

3. **Architectural Decisions** (`architectural-decisions.md`): Documentation of 6 key decisions including environment choice, robot proxy, LLM strategy, simulation stack, content structure, and research approach

4. **Validation Plan** (`validation-plan.md`): Comprehensive quality validation plan with reproducibility checks, hardware validation, module integrity testing, and success criteria mapping

The plan addresses all specified requirements including the 4 key architectural decisions with clear rationales, 13-week course structure with detailed content, and validation procedures tied to the original success criteria.

## Outcome

- ✅ Impact: Created complete architectural plan for robotics book
- 🧪 Tests: Comprehensive validation plan with reproducibility checks
- 📁 Files: Created 4 planning artifacts (architecture, course structure, decisions, validation)
- 🔁 Next prompts: Ready for task generation (/sp.tasks)
- 🧠 Reflection: Plan addresses all requirements with detailed validation strategy

## Evaluation notes (flywheel)

- Failure modes observed: N/A
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A