# LinkedIn — project listing

Copy-paste ready text for LinkedIn **Projects** / **Featured**.

**Repository:** https://github.com/romil0001/SO101_Visuomotor_Manipulation  
**Suggested media:** `assets/frames/overhead_real.jpg`, `assets/frames/wrist_real.jpg`, `assets/graphs/success_overall.png`

---

## Project fields

| Field | Value |
|-------|--------|
| **Project name** | SO-101 Visuomotor Manipulation Platform |
| **Project URL** | https://github.com/romil0001/SO101_Visuomotor_Manipulation |
| **Skills** | Robotics · Imitation Learning · Computer Vision · Python · PyTorch · Robot Manipulation · Machine Learning · Research |

---

## Description (full)

Built an end-to-end research platform for bimanual robot manipulation on a low-cost SO-101 dual-arm cell. The system combines leader–follower teleoperation, multi-view RGB sensing, and metric wrist depth (Intel RealSense D405) over a distributed Mac + Windows sensing setup.

Implemented a local visuomotor pipeline for demonstration collection, offline imitation learning (Action Chunking Transformer / ACT and SmolVLA), and closed-loop evaluation. Introduced a hybrid depth-assisted grasp controller that uses wrist depth at contact while keeping approach and place under the base policy.

Evaluated policies under a fixed protocol with repeated trials across scene conditions and binary success labels, reporting overall and per-condition success rates rather than training loss. Open-source research software with methods, architecture, and reproducible experiment workflow.

---

## Description (short)

Designed and built a bimanual SO-101 visuomotor manipulation platform: teleoperation, RGB + wrist-depth sensing, ACT/SmolVLA training, hybrid depth-assisted grasping, and protocol-based real-robot evaluation with per-condition success rates. Open source: https://github.com/romil0001/SO101_Visuomotor_Manipulation

---

## Announcement post (optional)

I’m sharing my open-source research project: **SO-101 Visuomotor Manipulation Platform**.

I built a full stack for real-robot imitation learning — from leader–follower teleoperation and multi-camera sensing (including metric wrist depth) to ACT / SmolVLA training and protocol-based evaluation.

Focus areas:
• Distributed sensing (Mac control + Windows depth/RGB publisher)
• Offline visuomotor learning without Hub-only workflows
• Hybrid depth-assisted grasping at contact
• Fair eval via condition grids and binary task success

Repo: https://github.com/romil0001/SO101_Visuomotor_Manipulation

Happy to connect with others working on robot learning / manipulation.

#Robotics #ImitationLearning #Visuomotor #MachineLearning #OpenSource
