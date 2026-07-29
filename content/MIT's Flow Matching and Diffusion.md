---
dg-publish: true
dg-show-toc: true
publish: true
title: MIT's Flow Matching and Diffusion
---
# Flow and Diffusion Models
Generative model convert initial distribution into data distribution

Flow
- Trajectory is a time dependent function. Vector field is  and ODE

$$X=X_{0},\frac{dX_{t}}{dt}=u_{t}(X_{t})$$

- Flow: function that input an initial condition and time and output the state from the initial condition such that it satisfy the evolution condition for each initial condition
- Uniqueness and existence of ODE solutions
- Numerical ODE: simulation/sampling with Euler method

Flow models
- Parameterize the vector field as a neural network, set the initial condition to randomized sample from normal distribution, the trajectory to endpoint satisfy the ODE with the vector field, fit the neural network such that the endpoint is the data distribution

Diffusion models

$$X=X_{0},\frac{dX_{t}}{dt}=u_{t}(X_{t})+\sigma_{t}dW_{t}$$

- SDE: random variables, diffusion coefficient, brownian motions, existence and uniqueness
- Stochastic evolution and simulation/sampling
- Diffusion models

Why not just train neural network to output sample instead of vector field? → Training stability

# Flow matching
Probability path is a time dependent probability density function
- Conditional probability path, Gaussian probability path is an interpolation between data and noise

$$p_{t}(x|z)\sim \mathcal{N}(\alpha_{t}z,\beta_{t}^2\mathbf{I})$$

- Marginal probability paths: integrate over all data point, time dependent

Vector field
- Conditional vector field: initialize + evolution → conditional probability path. 
- Conditional vector field for Gaussian probability path is a linear interpolation between data and noise

$$u_{t}^{target}(x|z)=\left( \dot{\alpha_{t}}-\frac{\dot{\beta_{t}}}{\beta_{t}}\alpha_{t} \right)z+\frac{\dot{\beta_{t}}}{\beta_{t}}x$$

- Marginal vector field: integrating over posterior, marginalization trick following gives the data distribution #wtf 

$$u_{t}^{target}(x)=\int u_{t}^{target}(x|z)p(z|x)dz$$

Flow matching learn the marginal vector field

$$L_{FM}=\mathbb{E}_{t,\mathbf{x},\mathbf{z}}\left[\|u_{\theta}(\mathbf{x},t)-u_{target}(\mathbf{x},t)\|^2\right]$$

- Regressing the marginal vector field (FM) is intractable, regressing the conditional vector field (CFM) is tractable and have the exact same effect

$$L_{CFM}=\mathbb{E}_{t,\mathbf{x},\mathbf{z}}\left[\|u_{\theta}(\mathbf{x},t)-u_{target}(\mathbf{x},t|\mathbf{z})\|^2\right]=L_{FM}-C$$

- Meaning the gradient update is the same and the minimizer of CFM is the same minimizer of FM
- Flow matching doesn’t simulate an ODE, so it’s simpler and more scalable
- Sampling noise $x$ with linear schedule, computing vector field with interpolation

$$x=tz+(1-t)\varepsilon$$

# Score Matching and Score Function
# Classifier-free Guidance
