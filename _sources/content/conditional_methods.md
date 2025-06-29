# 6. Conditional Diffusion Methods
Conditional diffusion models extend standard diffusion models by incorporating external conditioning information such as text, class labels, images, audio, or poses. This additional input guides the generation process, enabling controlled and task-specific outputs.

## Classifier Guidance
In classifier guidance, the diffusion model is steered using an auxiliary classifier trained to predict the class label $y$ from the noisy input $x_t$. Given the joint distribution:

$$
p(x_t,y)=p(x_t)p(y|x_t)
$$

The standard noise prediction:

$$
\nabla_{x_t} \log p(x_t) = - \frac{\epsilon_t}{\sqrt{(1-\bar{\alpha}_t)}}, \quad \text{assume } p(x_t)=q(x_t|x_0) 
$$

Using standard noise prediction, the score function of this distribution becomes:

$$
\begin{align*}
\nabla_{x_t} \log p(x_t,y) &= \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y|x_t)\\
&= - \frac{1}{\sqrt{(1-\bar{\alpha}_t)}} \epsilon_\theta(x_t,t)+\nabla_{x_t} \log p(y|x_t)\\
&= - \frac{1}{\sqrt{(1-\bar{\alpha}_t)}} (\epsilon_\theta(x_t,t)- \sqrt{(1-\bar{\alpha}_t)}\nabla_{x_t} \log p(y|x_t))
\end{align*}
$$

The adjusted noise term with classifier guidance is:

$$
\bar{\epsilon}_\theta(x_t,t) = \epsilon_\theta(x_t,t)- \sqrt{(1-\bar{\alpha}_t)}\nabla_{x_t} \log p(y|x_t)
$$

A classifier network $p_\phi(y|x_t)$ estimates $p(y|x_t)$ and its gradient $\nabla_{x_t} \log p(y|x_t)$ with respect to $x_t$. The updated noise prediction is as follows:

$$
\bar\epsilon_\theta(x_t,t)=\epsilon_\theta(x_t,t)- \sqrt{1-\bar{\alpha}_t}\nabla_{x_t} \log p_\phi(y|x_t))
$$

The strength of the classifier guidance can be controlled by adding a weight parameter $\omega\ge1$:

$$
\bar\epsilon_\theta(x_t,t)=\epsilon_\theta(x_t,t)-\omega\sqrt{1-\bar{\alpha}_t}\nabla_{x_t} \log p_\phi(y|x_t))
$$

$$
\begin{align*}
\bar\epsilon_\theta(x_t,t)&=\epsilon_\theta(x_t,t)-\omega\sqrt{1-\bar{\alpha}_t}\nabla_{x_t} \log p_\phi(y|x_t))\\
&=-\sqrt{1-\bar{\alpha}_t} (
\nabla_{x_t} \log p_\phi(x_t)+\omega\nabla_{x_t} \log p_\phi(y|x_t))\\
&=-\sqrt{1-\bar{\alpha}_t} (
\nabla_{x_t} \log p_\phi(x_t)+\nabla_{x_t} \log p_\phi(y|x_t)^\omega )\\
&=-\sqrt{1-\bar{\alpha}_t}
\nabla_{x_t} \log ( p_\phi(x_t)p_\phi(y|x_t)^\omega )
\end{align*}
$$

So, $p_\phi(x_t)p_\phi(y|x_t)^\omega$ is  used in place of $p_\phi(x_t)p_\phi(y|x_t)$. The term $p_\phi(y|x_t)^\omega$ amplifies large values, which make the network focus more on the modes of the classifier. This results in higher fidelity to input labels but less diversity.
Classifier guidance provide improved generation quality using labels, but at the expense of training an additional classifier. 

## Classifier-Free Guidance (CFG)
Classifier-free guidance eliminates the need for a separate classifier by training a single noise prediction model that handles both conditional and unconditional generation. The noise prediction network is modified to take the condition (label) $y$ as an additional input.

$$
\hat{\epsilon}_\theta(x_t,t) \to \hat{\epsilon}_\theta(x_t,y,t)
$$

A network is trained jointly for conditional and unconditional input by introducing a null symbol $\phi$.

$$
\hat{\epsilon}_\theta(x_t,\phi,t) \to \text{Noise prediction for null condition input}
$$

The impact of the conditioning can be enhanced by extrapolating the conditional noise: $\hat{\epsilon}_\theta(x_t,y,t)$ from the null-condition noise $\hat{\epsilon}_\theta(x_t,\phi,t)$:

$$
\tilde{\epsilon}_\theta(x_t,y,t) = (1+\omega)\hat{\epsilon}_\theta(x_t,y,t) - \omega \hat{\epsilon}_\theta(x_t,\phi,t), \quad \text{where } \omega\ge 0.
$$

## Connection to classifier guidance
If $\lambda:=1+\omega$, then

$$
\begin{align*}
\tilde{\epsilon}_\theta(x_t,y,t) &= \lambda\hat{\epsilon}_\theta(x_t,y,t) + (1-\lambda) \hat{\epsilon}_\theta(x_t,\phi,t)\\
&=\hat{\epsilon}_\theta(x_t,\phi,t)+ \lambda(\hat{\epsilon}_\theta(x_t,y,t) -  \hat{\epsilon}_\theta(x_t,\phi,t)) \quad \text{\{extrapolation\}}\\
&=- \sqrt{1-\bar{\alpha}_t}(\nabla_{x_t} \log p(x_t)+\lambda(\nabla_{x_t} \log p(x_t|y)-\nabla_{x_t} \log p(x_t)))\\
&=- \sqrt{1-\bar{\alpha}_t}
\nabla_{x_t} \log \left(p(x_t) \left( \frac{p(x_t|y)}{ p(x_t)}\right)^\lambda\right)\\
\end{align*}
$$

$$
\begin{align*}
p(x_t) \left( \frac{p(x_t|y)}{ p(x_t)}\right)^\lambda &\propto p(x_t) \left( \frac{p(x_t|y)p(y)}{ p(x_t)}\right)^\lambda \quad \text{\{Multiply constant } p(y)^\lambda \text{\}}\\
&= p(x_t)(p(y|x_t))^\lambda \quad \text{\{ Bayes' theorem \} }
\end{align*}
$$

It is equivalent to the condition enhancement in
Classifier Guidance. Pros and cons:
- Easy to implement. Versatile as not only labels but any additional information can be used. 
- The noise predictor needs to be evaluated twice in the generation process.
\end{itemize}

## Negative prompt
CFG can also enable negative prompting:

$$
\tilde{\epsilon}_\theta(x_t,y,t) = (1+\omega)\hat{\epsilon}_\theta(x_t,y_+,t) - \omega \hat{\epsilon}_\theta(x_t,y_-,t)
$$

where $y_+$ and $y_-$ are positive and negative prompts. 

## Latent Diffusion Models
One of the typical issues with diffusion models is the size/resolution of the input data. Higher dimensional data require higher compute, making the training and inference slow. To reduce computational costs, latent diffusion models operate in a lower-dimensional latent space rather than directly on high-dimensional image data. The noise prediction U-Net is applied to this latent representation, with cross-attention used to incorporate conditioning information.

![Latent Diffusion Models.](images/latent_diffusion_unet.png)


$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right).V
$$

Query ($Q$): output of each U-net layer.\\
Key $K$ and Value $V$: output of the input condition encoder.

## ControlNet
ControlNet enables conditional generation by fine-tuning only the conditional encoding pathway:
- The pre-trained noise prediction network is frozen.
- For encoding of the conditional image, a copy the pre-trained encoder parameters is used, which are allowed to be updated during finetuning.
- The encoded conditional image information is combined with the noisy image using zero convolution.


![Controlnet and zero convolution.](images/controlnet_zero_conv.png)


Zero Convolution $Z$ is a 1 × 1 convolution layer with learnable weight (scaling) parameters $a$ and bias (offset) parameters $b$, both of which are initialized with zero.

$$
\begin{align*}
Z(x;a,b)&=ax+b\\
y_c &= F(x;\theta)+Z(F(x+Z(c;a_1,b_1);\theta_c; a_2,b_2
\end{align*}
$$

where c is the condition image. In the beginning $\theta_c=\theta$ and $a_i, b_i=0$. Zero convolution helps in gradually incorporating the conditional information with the original noisy image. 

## Lora
The Controlnet based network benefits from a special training scheme, where parameters are updated using a low-rank condition on the weights. A bottleneck architecture is used where the intermediate outputs are reduced in dimensionality by representing weight matrices as a product of a low rank matrices. 

![Low-rank adaptation for network training.](images/lora.png)

## DDIM Deterministic Sampling 
if $\sigma_t = 0$, for all t in DDIM,
- Compute $x_{0|t} = \frac{1}{\sqrt{\bar{\alpha}_t}}( x_t - \sqrt{1-\bar{\alpha}_t} \hat\epsilon_\theta(x_t,t))$
- Compute $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_{0|t} + \sqrt{1-\bar{\alpha}_{t-1}} \hat\epsilon_\theta(x_t,t)$

$$
\begin{align*}
x_{t-1} &= \sqrt{\frac{\bar{\alpha}_{t-1}}{\bar{\alpha}_t}} (x_t - \sqrt{1-\bar{\alpha}_t} \hat\epsilon_\theta(x_t,t)) + \sqrt{1-\bar{\alpha}_{t-1}} \hat\epsilon_\theta(x_t,t)\\
&= \sqrt{\bar{\alpha}_{t-1}} \left[ \sqrt{\frac{1}{\bar{\alpha}_{t}}}  x_t + \left( \sqrt{\frac{1}{\bar{\alpha}_{t-1}}-1}- \sqrt{\frac{1}{\bar{\alpha}_{t}}-1}  \right)\hat\epsilon_\theta(x_t,t) \right]
\end{align*}
$$

## DDIM Inversion
For deterministic sampling, the mapping from $x_T$ to $x_0$ is fixed. Then, how can we also compute the inverse mapping from $x_0$ to $x_T$?
For the forward process with small time intervals,
approximate $(x_{t+1}-x_t)$ by simply replacing $(t-1)$ with $(t+1)$ in the $(x_{t-1}-x_t)$ formulation. 

$$
\begin{align*}
x_{t-1}-x_t &= \sqrt{\bar{\alpha}_{t-1}} \left[ \left( \sqrt{\frac{1}{\bar{\alpha}_{t}}} -\sqrt{\frac{1}{\bar{\alpha}_{t-1}}} \right) x_t + \left( \sqrt{\frac{1}{\bar{\alpha}_{t-1}}-1}- \sqrt{\frac{1}{\bar{\alpha}_{t}}-1}  \right)\hat\epsilon_\theta(x_t,t) \right]\\
x_{t+1}-x_t &= \sqrt{\bar{\alpha}_{t+1}} \left[ \left( \sqrt{\frac{1}{\bar{\alpha}_{t}}} -\sqrt{\frac{1}{\bar{\alpha}_{t+1}}} \right) x_t + \left( \sqrt{\frac{1}{\bar{\alpha}_{t+1}}-1}- \sqrt{\frac{1}{\bar{\alpha}_{t}}-1}  \right)\hat\epsilon_\theta(x_t,t) \right]\\
\end{align*}
$$

Inversion fails when the number of time steps is too small (when the time intervals are too large).


## Image Editing Using DDIM Inversion
DDIM inversion helps when the generation of an image is based on existing images while preserving important information. Image editing applications are a good candidate for such techniques. Typically,
- DDIM inversion is performed using the original prompt in CFG.
- which is then followed by a reverse processing using a new prompt in CFG.

![Left: DDIM Inversion and Regeneration. Right: Image Editing using DDIM Inversion and Regeneration](images/ddim_inversion_image_edit.png)

- Good image editing can be achieved by using a high CFG weight for both inversion and generation.
- However, inversion tends to fail when CFG weight $\omega$ is high.

## Null-Text Inversion
To address some issues related to inversion with high CFG weight, in null-text inversion, first inversion is performed with $\omega = 1$. Let the latent variables $\{x_t^*\}_{t=1,....T}$ be pivots, followed by an inversion with $\omega >>1$ while enforcing the latent variable $x_t$ to be close to the corresponding latent variable $x_t^*$ while tuning some parameters. 

With $x_{t-1}(x_t,c,\phi)$ indicating the computation of $x_{t-1}$ from $x_t$, input prompt $c$, and null prompt $\phi$, minimize

$$
||x_{t-1}^* - x_{t-1}(x_t, c, \phi||^2_2 
$$

![Null-Text Inversion.](images/null_inversion.png)

## Summary

| Method                   | Extra Classifier   | Conditioning Type  | Pros                               | Cons                                            |
| ------------------------ | ------------------ | ------------------ | ---------------------------------- | ----------------------------------------------- |
| Classifier Guidance      | Yes                | Label only         | High fidelity, strong control      | Requires separate classifier                    |
| Classifier-Free Guidance | No                 | Any (text, image)  | Simple, flexible, versatile        | Double forward passes needed                    |
| ControlNet               | No (tuned encoder) | Image (pose, edge) | Explicit spatial control, reusable | Increased model size/training                   |
| Latent Diffusion         | No                 | Any                | Efficient memory/computation       | Loss of some detail in latent space             |
| DDIM Inversion           | No                 | Any                | Image editing & regeneration       | Inversion difficult for small steps or high CFG |



## Reference
- Ho, Jonathan, and Tim Salimans. "Classifier-free diffusion guidance." arXiv preprint arXiv:2207.125 (2022).
- Zhang, Lvmin, Anyi Rao, and Maneesh Agrawala. "Adding conditional control to text-to-image diffusion models." Proceedings of the IEEE/CVF international conference on computer vision. 2023.
- Dhariwal, Prafulla, and Alexander Nichol. "Diffusion models beat gans on image synthesis." Advances in neural information processing systems 34 (2021): 8780-8794.
- Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.