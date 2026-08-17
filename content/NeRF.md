---
publish: true
status: reference
---

Difference between view synthesis and surface reconstruction: the output of surface reconstruction are common 3D representation such as mesh, voxels, etc. The output of view synthesis are just simply 2D images, often from new viewing angles than the input images. The key difference is that view synthesis **doesn’t require capturing the geometry** of the 3D scene/model.

![[Pasted image 20260817142239.png]]

NeRF is a method that uses an implicit representation for view synthesis. Unlike explicit representation, implicit representation requires extra steps to generate views. Different types of implicit representation (e.g. SDF) may require different techniques (e.g. sphere tracing). NeRF uses **volume rendering**.

First, we represent our 3D scene using a **radiance field**, which is a function that input a 3D coordinates $(x,y,z)$ and a 2D viewing angle $(\theta,\phi)$ and output a color and density. The 3D coordinates is the world space coordinate, and the view angles represent the view direction **standing at exactly that coordinates**. Our model is learnable MLP of this radiance field.
![[Screenshot 2026-08-17 at 14.15.16.png]]
Having specified this 5D-4D radiance field, the entire NeRF pipeline is as follows
- We are given a set of existing views, consisting of a sets of 2D images of the same scene, *each calibrated with a camera intrinsic and extrinsic*
- For each view, we *cast (sample) ray from the camera through the pixels in its virtual image plane* into the 3D scenes. We march along these rays to *sample discrete points* in the 3D scene
- For each ray and its set of sampled points, we query our MLP radiance field to find the density and color of each point in the ray view angle. 
- We then perform **ray marching (discretized radiance integration)** to accumulate the radiance emitted from each point to find the color of the pixel on the virtual image plane along that ray.
$$\begin{align}
&T_{i}=\prod_{j=i+1}^n(1-\alpha_{j})=\exp\left( -\sum_{j=i+1}^n\sigma_{j}\delta_{j} \right) \\
&\boxed{I=\sum_{i}T_{i}\alpha_{i}\left( \frac{c_{i}}{\sigma_{i}} \right)} \quad\text{where}\quad\delta_{i}=t_{i+1}-t_{i}
\end{align}$$
- This give us the predicted image according to MLP of the given view. Comparing this with the ground truth image give us a loss (Mean Squared Error) which we then use to train our MLP through backpropagation
- After training on all existing view, we synthesize new views by giving the same pipeline a new camera intrinsic/extrinsic: casting rays through the virtual image plane → sampling points on the rays → query the MLP for density and color → integrate radiance to get the pixel colors

Tricks
- Positional encoding
- Hierarchical sampling

3DGS
