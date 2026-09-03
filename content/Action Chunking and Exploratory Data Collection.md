---
status: active
draft: "true"
---
Underlying problem: Imitation learning suffer from exponentially compounding error as horizon increases
- A common mitigation strategy involves interactive collection of expert data at suboptimal state to observe “how expert recovers from error”
- On the other hand, this paper suggest techniques that doesn’t require this interactivity like action chunking and noise injection is sufficient for preventing this compounding error

Control theory 

Action chunking definition (ACT)
- Motivation: To enable low cost hardware to perform complex tasks, the planning phase becomes more difficult. Specifically, behavioral cloning (a form of imitation learning) from 
- Key idea: use learning from “closed loop visual feedback” and actively compensating for errors
