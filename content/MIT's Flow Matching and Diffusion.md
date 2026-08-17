---
publish: true
title: MIT's Flow Matching and Diffusion
status: triage
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
Score function is the **gradient of log-likelihood** of a probability
$$\nabla\log p(x)$$
- It points towards the direction where log-likelihood increases the most
- Define conditional score and marginal score based on probability path
- Integrating conditional score over posterior gives the marginal score
$$\nabla\log p_{t}(x)=\int \nabla \log p_{t}(x|z) \frac{p_{t}(x|z)p(z)}{p_{t}(x)}dz$$
Example: Score of a conditional probability path
$$\begin{align}
\nabla\log p_{t}(x|z)=-\frac{1}{\beta_{t}^2}x+\frac{\alpha_{t}}{\beta_{t}^2}z\tag{1}
\end{align}$$
Reparameterizing the noise schedule as follow, we can express the vector field in terms of the score function 

$$\begin{align*}
a_{t}=\left(\beta_{t}^2\frac{\dot{\alpha}_{t}}{\alpha_{t}}-\dot{\beta}_{t}\beta_{t} \right),b_{t}=\frac{\dot{\alpha}_{t}}{\alpha_{t}} \\
u_{t}^{target}(x)=a_{t}\nabla\log p_{t}(x)+\beta_{t}x
\end{align*}
$$

→ Score matching learn score functions, which is *basically equivalent* to flow matching learning vector field.
- Denoising score matching loss is basically the conditional flow matching loss, regressing the conditional score function
- Exactly the same as flow matching, we sample a sample in time by sampling data point and noise, but this time using equation $(1)$ instead

SDE sampling
- Stochastic sampling follows the probability path if the Fokker-Planck equation holds
$$X_{0}\sim p_{init},\quad dX_{t}=\left( u_{t}^{target}(X_{t})+\frac{\sigma^2_{t}}{2}\nabla \log p_{t}(x) \right)dt+\sigma_{t}dW_{t}$$
- In addition to the drift term, we now add random Brownian motion every time, which gives us more flexibility. There needs to be a correction to the drift term due to the introduction of the diffusion term.
- Given the model trained without these noise step, we can now have different ways of sampling, given any diffusion coefficient, by correcting the drift term by an amount according to the score function
→ In practice, ODE sampling often leads to the **best results**.

# Classifier-free Guidance
Vanilla guidance
- We introduce a condition $y$ and model the joint distribution $p_{t}(x,y)$, or equivalently the vector field $u_{t}^{target}(x|y)$
- We optimize the flow matching loss. Note that in practice the condition $y$ is not involved in the conditional vector field because the end point is already defined
$$\mathcal{L}_{CFM}=\mathbb{E}_{z,x,y}\left[\left\|u_{t}^{\phi}(x|y)-u_{t}^{target}(x|z)\right\|^2\right]$$
- Sampling involves injecting the condition along with the noise through every step of the ODE simulator
$$X_{0}\sim p_{init},\quad X_{t}\leftarrow X_{t}+hu_{t}^{\phi}(X_{t}|y)$$
→ This doesn’t work well in practice

Classifier guidance
- According to Bayes rule 
$$\begin{align*}
p(x|y)&=\frac{{p(y|x)p(x)}}{p(y)}\\
\nabla_{x} \log p_{t}(x|y)&=\nabla_{x}\log p_{t}(y|x)+\nabla_{x}p_{t}(x)\\
u_{t}^{target}(x|y)&=u_{t}^{target}(x)+a_{t}\nabla_{x}\log p_{t}(y|x)
\end{align*}$$
- As such the guided vector field can be expressed as a sum of the unguided vector field and the score function of a classifier
- During sampling, we apply a weight $w\geq 1$ to this classifier term to reinforce the adherence of the model to the prompt
$$\tilde{u}_{t}^{w}(x|y)=u_{t}^{target}(x)+\textcolor{red}{w}a_{t}\nabla_{x}\log p_{t}(y|x)$$
→ It’s retraining-free but requires a separate classifier model

Classifier-free guidance
- The classifier term can be expressed as
$$\begin{align*}
\tilde{u}_{t}^{w}(x|y)&=u_{t}^{target}(x)+wa_{t}\nabla_{x}\log p_{t}(y|x)\\
&=u_{t}^{target}(x)+wa_{t}(\nabla_{x}\log p_{t}(x|y)-\nabla_{x}\log p_{t}(x))\\
&=u_{t}^{target}(x)+w(a_{t}\nabla_{x}\log p_{t}(x|y)+b_{t}x-a_{t}\nabla_{x}\log p_{t}(x)-b_{t}(x))\\
&=u_{t}^{target}(x)+w(u_{t}^{target}(x|y)-u_{t}^{target}(x))\\
&=wu_{t}^{target}(x|y)+(1-w)u_{t}^{target}(x)
\end{align*}$$
- We change the formulation of this derivation by introducing an empty token $\varnothing$, to express the vector field when not following any condition.
- Doing so enables us to train a single model $u_t^{\phi}(x|y)$, and during training we have to sometimes drop the label to learn the empty token alongside actual conditions. 
- CFG is a heuristic, we’re not sampling from the data distribution anymore
$$\begin{align*}
dX_{t}&=\frac{1}{2}\sigma^2\nabla_{x}\log p_{t}(X_{t})dt+\sigma dW_{t}\\
&=\frac{1}{2}\sigma^2\left( -\frac{2\theta}{\sigma^2}X_{t} \right)dt+\sigma dW_{t}\\
&=-\theta X_{t}dt+\sigma dW_{t}
\end{align*}$$
# Latent Space & Network Architecture
Standard AE involves an encoder and a decoder. Reconstruction loss is minimized
$$\mathcal{L}_{recontruction}=\mathbb{E}_{x\sim p_{data}}[\|\mu_{\theta}(\mu_{\phi}(x))-x\|^2]$$
- Latent distribution is hard to interpret and 

VAE
- Models the means and variance of encoder and decoder
- Reconstruction loss is MLE (or MSE). Prior loss to keep the latent distribution “nice”
$$\mathcal{L}_{prior}=\mathbb{E}_{z}(q_{\phi}(z|x))$$
- Reparameterization trick

Network architecture
- Time: sinusoidal function
- Prompt: CLIP, pretrained
- Image/latent: patchify into sequence
- DiT: self attention, cross attention, positional embedding
- Example: StableDiffusion3, Meta MovieGen

Time conventions
- Flow time: 1 to 0
- Diffusion time: 0 to infinity
- Discrete time: DDPM/DDIM

