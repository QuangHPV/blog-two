---
publish: true
status: triage
---
Motivation: VLA likes RT-2 or $\pi_{0}$ use vision-language pretrained knowledge to directly map observation to action. They seem to pose many problems
- Data hungry: requires imitation learning or RL, which is very data hungry
- Poor generalization: 

World Action Models propose to combines World Models into VLA by additionally predicting (or generating) the future state of the world
- The model learns a physical understanding of the world and can reason upon it directly as an additional source of information (less data hungry)
- It can also generalize better because its state prediction capabilities learn the distribution of the world

DreamZero

How do you train on unlabeled human egocentric dataset?
How does generalist robot policies actually unify action?
Is the video backbone different from the 
What should the video backbone output? (the objective)
What shall WE do then, without physical robots? World Model? Evaluation
Is the choice of diffusion vs autoregressive decoupled from the choice of with or w/o World Model?
