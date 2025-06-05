# Denoising Diffusion Probabilistic Models (DDPM)
Let us consider a special case of Markovian hierarchical Variational Autoencoders (VAEs), where the latent space has the same dimensionality as the data space. In this setup, the variational posteriors 
$q_\phi(x_{t+1}|x_t)$ are not learned but predefined, i.e.,

$$
q_\phi(x_{t+1}|x_t) \to q(x_{t+1}|x_t)
$$

modeled as a linear Gaussian. The Gaussian parameters of the latent encoders vary over time such that the distribution of the latent at the final time step $T$ is a standard Gaussian.

![An illustration of a Variational Diffusion Model. $x_0$ represents the original data (e.g., natural images), $x_T$ corresponds to pure Gaussian noise, and $x_t$ denotes intermediate noisy states of $x_0$. Each $q(x_t|x_{t-1})$ is modeled as a Gaussian that uses the output of the previous state as its mean.](images/diffusion_model.png)

**Forward process:**

$$
\begin{align*} 
q(x_{1:T}|x_0) &= \prod_{t=1}^T q(x_{t}|x_{t-1})\\
q(x_t|x_{t-1}) &= \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t \mathbf{I})
\end{align*} 
$$

where the noise schedule $\{\beta_t\in (0,1)\}_{t=1}^T$ is monotonically increasing: $\beta_1\leq\beta_2\leq...\leq\beta_T$. The forward process can be interpreted as gradually adding Gaussian noise over time. There are two common variants of adding noise:
- Variance Preserving: $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t \mathbf{I})$,  where the goal is to keep the variance very small for each step. 
- Variance Exploding: $q(x_t|x_{t-1}) = \mathcal{N}(x_t; x_{t-1}, (\sigma_i^2-\sigma_{i-1}^2)\mathbf{I})$

Further more, the $\beta_t$ can be constant, linearly or quadratically increasing, a cosine function, or learned.

**Note:** For reverse step $p_\theta(x_{t-1}|x_t)$ becomes Gaussian only when $\beta_t$ is small $(\beta_t<<1)$.

**Reverse process:** 

$$
p_\theta(x_{0:T}) = p(x_{T}) \prod_{t=1}^T p_\theta(x_{t-1}|x_{t}),\quad\text{where } p(x_T)=\mathcal{N}(x_T, 0, \mathbf{I})
$$

---

## ELBO for Diffusion model

For the diffusion model the negative Evidence Lower Bound (ELBO) is minimized:

$$
\begin{align*} 
- &\log\; p(x) = -\log \; \int p_\theta(x_{0:T}) dx_{1:T} = -\log \; \int \frac{p_\theta(x_{0:T})q(x_{1:T}|x_0)}{q(x_{1:T}|x_0)} dx_{1:T} = -\log \; \mathbb{E}_{q(x_{1:T}|x_0)} \left[ \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \right]\\
& \geq \mathbb{E}_{q(x_{1:T}|x_0)} \left[ -\log \; \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right] = ... = \\
&= -\underset{\textbf{reconstruction term}}{\underbrace{\mathbb{E}_{q(x_1|x_0)} \left[ \log \; p_\theta(x_0|x_1)\right]}}
+\underset{\textbf{prior matching term}}{\underbrace{\mathbb{E}_{q(x_{T-1}|x_0)} \left[ D_{KL} (q(x_T|x_{T-1})|| p(x_T))\right]}}
+ \underset{\textbf{consistency term}}{\underbrace{\sum_{t=1}^{T-1} \mathbb{E}_{q(x_{t-1}, x_{t+1} |x_0)} \left[ D_{KL} (q(x_t|x_{t-1})|| p_\theta(x_t|x_{t+1}))\right]}}
\end{align*}
$$

### Consistency term
The consistency term enforces agreement between the forward and reverse transitions at each timestep: 

$$
\sum_{t=1}^{T-1} \mathbb{E}_{q(x_{t-1}, x_{t+1} |x_0)} \left[ D_{KL} (q(x_t|x_{t-1})|| p_\theta(x_t|x_{t+1}))\right]
$$

![Consistency term.](images/consistency_term.png)

However, this term involves expectations over two random variables, which is computationally expensive. To reduce complexity, the ELBO can be reparameterized using the Markov assumption ($x_t$ only depends on $x_{t-1}$). Hence,

$$
\begin{align*} 
q(x_t|x_{t-1}) &= q(x_t|x_{t-1}, x_0)\\
q(x_t|x_{t-1}) &= \frac{q(x_{t-1}|x_{t}, x_0)q(x_t|x_0)}{q(x_{t-1}|x_0)}
\end{align*} 
$$

Using this, the ELBO can be rewritten as:

$$
\begin{align*} 
\underset{\textbf{reconstruction term}}{\underbrace{-\mathbb{E}_{q(x_1|x_0)} \left[ \log \; p_\theta(x_0|x_1)\right]}} 
+\underset{\textbf{prior matching term}}{\underbrace{ D_{KL} (q(x_T|x_0)|| p(x_T))}}+ \underset{\textbf{denoising matching term}}{\underbrace{\sum_{t=2}^{T} \mathbb{E}_{q(x_{t}| x_0)} \left[ D_{KL} (q(x_{t-1}|x_{t}, x_0)|| p_\theta(x_{t-1}|x_{t}))\right]}}
\end{align*}
$$


### Reconstruction Term
Reconstruction term resembles the VAE reconstruction loss and can be optimized using Monte Carlo estimation. 

$$
\mathbb{E}_{q(x_1|x_0)} \left[ \log \; p_\theta(x_0|x_1)\right]
$$

### Prior Matching Term 
The prior matching term ensures the final noisy state aligns with the standard Gaussian prior. Since both $q(x_T|x_0)$ and $p(x_T)$ are predefined, there are no learnable parameters.

$$
D_{KL} (q(x_T|x_0)|| p(x_T))
$$

### Denoising Matching Term
Denoising matching term minimize the difference between the forward and reverse denoising process by using $q(x_{t-1}|x_t,x_0)$ as a ground truth for the reverse process.

$$
\sum_{t=2}^{T} \mathbb{E}_{q(x_{t}| x_0)} \left[ D_{KL} (q(x_{t-1}|x_{t}, x_0)|| p_\theta(x_{t-1}|x_{t}))\right]
$$

---

### Forward Convergence

**For forward steps, Under certain assumptions $q(x_t|x_0)= N(x;0,I)$. How?**

From our assumptions about encoder transitions:

$$
\begin{align*}
q(x_t|x_{t-1}) &= \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t \mathbf{I})  &&\quad\quad  \text{where } \{\beta_t\in (0,1)\}_{t=1}^T \text{and } \beta_1<\beta_2<...<\beta_T\\
& or && \\
q(x_t|x_{t-1}) &= \mathcal{N}(x_t; \sqrt{\alpha_t}x_{t-1}, (1-\alpha_t) \mathbf{I})
&& \quad \quad \text{where } \alpha_t = 1- \beta_t \quad\{\alpha_t\in (0,1)\}_{t=1}^T \text{and } \alpha_1\geq\alpha_2\geq...\geq\alpha_T
\end{align*}
$$

---

Given $x_1 \sim \mathcal{N}(\mu_1, \sigma_1^2 \mathbf{I})$ and $x_1 \sim \mathcal{N}(\mu_2, \sigma_2^2 \mathbf{I})$, then 
$$
x_1+x_2 \sim \mathcal{N}(\mu_1  +\mu_2, (\sigma_1^2+\sigma_2^2) \mathbf{I})
$$

---

Also, if $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0,1)$, and $x_1 = \sigma_1\epsilon_1$, $x_2 = \sigma_2\epsilon_2$, then 
$$
x_1 + x_2 \sim \mathcal{N}(0,(\sigma_1^2 + \epsilon_2^2)\mathbf{I})$ and $x_1 + x_2 = \sqrt{\sigma_1^2 + \epsilon_2^2} \epsilon
$$

---

Using the above expression for forward steps,

$$
\begin{align*} 
q(x_1|x_0) &= \mathcal{N}(x_1; \sqrt{\alpha_1}x_{0}, (1-\alpha_1) \mathbf{I})\\
q(x_2|x_{1}) &= \mathcal{N}(x_2; \sqrt{\alpha_2}x_{1}, (1-\alpha_2) \mathbf{I})
\end{align*}
$$

Using reparameterization trick and $\epsilon_0, \epsilon_1 \sim N(0,1)$,

$$
\begin{align*} 
x_1 &= \sqrt{\alpha_1}x_{0}+ \sqrt{(1-\alpha_1)} \epsilon_0 \\
x_2 &= \sqrt{\alpha_2}x_{1}+ \sqrt{(1-\alpha_2)} \epsilon_1 
= \sqrt{\alpha_2}(\sqrt{\alpha_1}x_{0}+ \sqrt{(1-\alpha_1)} \epsilon_0)+ \sqrt{(1-\alpha_2)} \epsilon_1 
= \sqrt{\alpha_2\alpha_1}x_{0}
+ \sqrt{\alpha_2(1-\alpha_1)} \epsilon_0+ \sqrt{(1-\alpha_2)} \epsilon_1\\
&= \sqrt{\alpha_2\alpha_1}x_{0}
+ \sqrt{(1-\alpha_2\alpha_1)} \bar{\epsilon}_0\\
x_t &= \sqrt{\alpha_t}x_{t-1}+ \sqrt{(1-\alpha_t)} \epsilon_{t-1}
=\sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}x_{t-2}+ \sqrt{(1-\alpha_{t-1})} \epsilon_{t-2})+ \sqrt{(1-\alpha_t)} \epsilon_{t-1} \\
&= \sqrt{\alpha_t\alpha_{t-1}}x_{t-2}
+ \sqrt{(1-\alpha_t\alpha_{t-1})} {\epsilon}_{t-2} 
= ...\\
&= \sqrt{\prod_{i=1}^t \alpha_i} x_0 + \sqrt{(1-\prod_{i=1}^t \alpha_i)} {\epsilon}_{0}
\end{align*}
$$

Therefore, one can directly model $x_t$ from $x_0$ using following Gaussian distribution with using a Markovian chain:

$$
q(x_t|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}}x_0, (1-\bar{\alpha}_t)\mathbf{I}), \quad \text{where} \; \bar{\alpha}_t = \prod_{i=1}^t \alpha_i
$$

---

**Question:** What is $\lim_{T\to \infty} \bar{\alpha}_T = \lim_{T\to \infty} \prod_{t=1}^T \alpha_t = \lim_{T\to \infty} \prod_{t=1}^T (1- \beta_t)$?

As $T\to \infty$, $\bar{\alpha}_T \to 0$. So $q(x_T|x_{0}) \to N(x_T; 0, I)$, which shows the convergence of the forward process to a standard Gaussian.

---

## Denoising Matching Term 

$$
\sum_{t=2}^{T} \mathbb{E}_{q(x_{t}| x_0)} \left[ D_{KL} (q(x_{t-1}|x_{t}, x_0)|| p_\theta(x_{t-1}|x_{t}))\right]
$$

For denoising match term, $p_\theta(x_{t-1}|x_t)$ should be close to $q(x_{t-1}|x_t, x_0)$ for each t. 
What is $q(x_{t-1}|x_{t}, x_0)$?

$$
\begin{align*} 
&q(x_{t-1}|x_{t}, x_0) = q(x_{t}|x_{t-1}, x_0) \frac{q(x_{t-1}| x_0)}{q(x_{t}| x_0)} 
= q(x_{t}|x_{t-1}) \frac{q(x_{t-1}| x_0)}{q(x_{t}| x_0)} \quad \text{\{Using Markovian assumption}\}\\
& q(x_{t}| x_{t-1})= \mathcal{N}(\sqrt{\alpha_t}x_{t-1}, (1-\alpha_t)\mathbf{I}) \\
& q(x_{t-1}| x_0)= \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}x_0, (1-\bar{\alpha}_{t-1})\mathbf{I})\\
& q(x_{t}| x_0)= \mathcal{N}(\sqrt{\bar{\alpha_t}}x_0, (1-\bar{\alpha}_t)\mathbf{I})
\end{align*}
$$

Using above expressions, 

$$
\begin{align*} 
q(x_{t-1}|x_{t}, x_0) &= q(x_{t}|x_{t-1}) \frac{q(x_{t-1}| x_0)}{q(x_{t}| x_0)}\\
&\propto exp\left( -\frac{1}{2} \left( 
\frac{(x_t-\sqrt{\alpha_t}x_{t-1})}{1-\alpha_t} 
+\frac{(x_t-\sqrt{\bar{\alpha}_{t-1}}x_{0})}{1-\bar{\alpha}_{t-1}} 
-\frac{(x_t-\sqrt{\bar{\alpha}_{t}}x_{0})}{1-\bar{\alpha}_{t}} 
\right)\right)\\
&= ...\\
&= \mathcal{N}(\tilde{\mu}(x_t, x_0), \tilde{\sigma}_t^2 \mathbf{I}) \quad \text{\{another normal distribution\}}
\end{align*}
$$

where 

$$
\tilde{\mu}(x_t, x_0) = \frac{\sqrt{\alpha}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{1-\bar{\alpha}_{t}} x_0 
$$

$$
\tilde{\sigma}_t^2 = \frac{(1-\bar{\alpha}_{t-1})(1-\alpha_t)}{1-\bar{\alpha}_{t}}
$$

The mean $\tilde{\mu}$ is a function of $x_t$  and $x_0$. The covariance is predefined from user defined $\beta_t$ or $\alpha_t$. So, $p_\theta(x_{t-1}|x_t)$ can be modeled as a Gaussian with predefined $\sigma$ to match $q(x_{t-1}|x_t,x_0)$, but note that $p_\theta(x_{t-1}|x_t)$ is not conditioned on $x_0$.

---

If $p(x) = N(x; \mu_p, \sigma^2I)$ and $q(x) = N(x; \mu_q, \sigma^2I)$, then 

$$
D_{KL}(p||q) = \frac{1}{2\sigma^2} ||\mu_q-\mu_p||^2 
$$

---

**How to model the variational distribution $p_\theta(x_{t-1}|x_t)$?** 

For $q(x_{t-1}|x_{t}, x_0)= N(\tilde{\mu}(x_t, x_0), \tilde{\sigma}_t^2 I)$, the variance $\tilde{\sigma}_t^2$ is not a function of $x_t$ and $x_0$. Hence, we can define the variational distribution $p_\theta(x_{t-1}|x_t)$ as 

$$
p_\theta(x_{t-1}|x_{t})= N({\mu_\theta}(x_t, t), \tilde{\sigma}_t^2 I) \quad \text{where } \mu_\theta(x_t, t) \text{: mean predictor}.
$$ 

$$
\begin{align*}
\mathbb{E}_{q(x_{t}| x_0)} & [ D_{KL} (q(x_{t-1}|x_{t}, x_0)|| p_\theta(x_{t-1}|x_{t}))] = \frac{1}{2\tilde{\sigma}_t^2} \mathbb{E}_{q(x_{t}| x_0)} \left[|| \mu_\theta(x_t,t)- \tilde{\mu}(x_t,x_0)||^2 \right] 
\end{align*}
$$

---

### $x_0$ Predictor

Taking a closer look at $\tilde{\mu}(x_t,\epsilon_t)$, we can also define $\mu_\theta$ as a function of $x_t$ and $\tilde{x}_\theta(x_t,t)$, where $\tilde{x}_\theta(x_t,t)$ is parametrized using a neural network that seeks to predict $x_0$ firm the noisy $x_t$ and time index $t$. 

$$
\begin{align*}
\tilde{\mu}(x_t,x_0) & = \frac{\sqrt{\alpha}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{1-\bar{\alpha}_{t}} x_0\\
\tilde{\mu}_\theta(x_t,t) & = \frac{\sqrt{\alpha}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{1-\bar{\alpha}_{t}} \tilde{x}_\theta(x_t,t)\\
\end{align*}
$$

$$
\begin{align*}
\mathbb{E}_{q(x_{t}| x_0)}  [ D_{KL} (q(x_{t-1}|x_{t}, x_0)|| p_\theta(x_{t-1}|x_{t}))] & =\frac{1}{2\tilde{\sigma}_t^2} \mathbb{E}_{q(x_{t}| x_0)} \left[|| \mu_\theta(x_t,t)- \tilde{\mu}(x_t,x_0)||^2 \right] \\
&=\frac{1}{2\tilde{\sigma}_t^2}
\frac{\bar{\alpha}_{t-1}(1-\alpha_t)^2}{(1-\bar{\alpha}_t)^2}
\mathbb{E}_{q(x_{t}| x_0)} \left[|| \hat{x}_\theta(x_t,t)- x_0||^2 \right]\\ 
&= w_t \mathbb{E}_{q(x_{t}| x_0)} \left[|| \hat{x}_\theta(x_t,t)- x_0||^2 \right]
\end{align*}
$$

- $x_t$ is sampled from $x_0$.
- From $x_t$, predict the expected value of $x_0$ that would result in sampling $x_t$ from it through the forward jump.
- Note that our goal is to sample $x_0$ from a standard normal sample $x_T$ and through latent variables $x_{T-1}, x_{T-2}, ..., x_1$. But for every $x_t$, we directly predict the expected value of $x_0$ from $x_t$.

---

### $\epsilon_t$ Predictor

**From the forward jump:**  $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon_t$. 
If $x_t$ and $x_0$ are given, define $\epsilon_t$ as

$$ 
\epsilon_t = \frac{1}{\sqrt{1-\bar{\alpha}_t}}x_t- \frac{\sqrt{\bar{\alpha}_t}}{\sqrt{1-\bar{\alpha}_t}}x_0 
$$

One can also rewrite $\tilde{\mu}(x_t,x_0)$ as a function of $x_t$ and $\epsilon_t$. 

$$
\begin{align*}
\tilde{\mu}(x_t,\epsilon_t) & = \frac{\sqrt{\alpha}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{1-\bar{\alpha}_{t}} x_0= \frac{1}{\sqrt{{\alpha}_{t}}} \left( x_t - \frac{1- {\alpha}_{t}}{\sqrt{1-\bar{\alpha}_{t}}}\epsilon_t\right)
\end{align*}
$$

$$
\begin{align*}
\mathbb{E}_{q(x_{t}| x_0)} [ D_{KL} (q(x_{t-1}|x_{t}, x_0)|| p_\theta(x_{t-1}|x_{t}))] &=\frac{1}{2\tilde{\sigma}_t^2} \mathbb{E}_{q(x_{t}| x_0)} \left[|| \mu_\theta(x_t,t)- \tilde{\mu}(x_t,x_0)||^2 \right] \\
&=\frac{1}{2\tilde{\sigma}_t^2}
\frac{(1-\bar{\alpha}_{t})^2}{\alpha_t(1-\bar{\alpha}_t)}
\mathbb{E}_{q(x_{t}| x_0)} \left[|| \hat{\epsilon}_\theta(x_t,t)- \epsilon_t||^2 \right]\\ 
&= w'_t \mathbb{E}_{q(x_{t}| x_0)} \left[|| \hat{\epsilon}_\theta(x_t,t)- \epsilon_t||^2 \right]
\end{align*}
$$

From $x_t$, predict the expected value of $\epsilon_t$ that would result in a sampling $x_t$ from $x_0$ through the forward jump.\\
Although all three interpretations: mean prediction, $x_0$ prediction, and noise prediction, are equivalent, in practice, $\epsilon_t$ predictor is used since the $\epsilon_t$ are well normalized and scaled standard normal samples, which makes it easier to train the neural network. 


For DDPM ELBO,
- The reconstruction term ($-\mathbb{E}_{q(x_1|x_0)} \left[ \log \; p_\theta(x_0|x_1)\right]$ is same as with VAEs and it is also negligible compared to other loss terms. 
- The prior matching term ($ D_{KL} (q(x_T|x_0)|| p(x_T))$) converges to zero when $T\to \infty$. 
- The denoising matching term ($\sum_{t=2}^{T} \mathbb{E}_{q(x_{t}| x_0)} \left[ D_{KL} (q(x_{t-1}|x_{t}, x_0)|| p_\theta(x_{t-1}|x_{t}))\right]$) is the only prominent term, which can also be written as 

$$
\mathbb{E}_{t>1,q(x_{t}| x_0)} \left[ D_{KL} (q(x_{t-1}|x_{t}, x_0)|| p_\theta(x_{t-1}|x_{t}))\right]
$$ 

---

## Training DDPM

$$
\mathbb{E}_{t>1,q(x_{t}| x_0)} \left[ D_{KL} (q(x_{t-1}|x_{t}, x_0)|| p_\theta(x_{t-1}|x_{t}))\right]
$$

**Repeat:**
1. Take a random $x_0$.
2. Sample $t\sim \mathcal{U}(\{1,...,T\})$.
3. Sample $\epsilon_t \sim \mathcal{N}(0,\mathbf{I})$. 
4. Compute $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon_t  $
5. Take a gradient descent step on $\nabla_\theta|| \hat{\epsilon}_\theta(x_t,t)-\epsilon_t||^2$. 

## Reverse Process (DDPM): Generation
1.  Sample $x_t \sim \mathcal{N}(0,\mathbf{I})$.
2. For $t=T,...,1,$ repeat: 
    - Compute $\tilde{\mu}= \frac{1}{\sqrt{\bar{\alpha}_{t}}} \left( x_t - \frac{1- {\alpha}_{t}}{\sqrt{1-\bar{\alpha}_{t}}}\tilde{\epsilon}_\theta(x_t,t)\right)$.
    - Sample $z_t \sim \mathcal{N}(0,\mathbf{I})$
    - Compute $x_{t-1} = \bar{\mu}+ \bar{\sigma} z_t$.


At each time step $t$, given $x_t$, $\epsilon_t$ is predicted. The prediction of $x_0$ can be computed from $x_t$ and $\epsilon_t$.
Given $x_t$ and the noise prediction $\bar{\epsilon}_\theta(x_t,t)$: 

$$
 x_{0|t} = \frac{1}{\sqrt{\bar{\alpha}_t}}( 1 - \sqrt{1-\bar{\alpha}_t} \hat\epsilon_\theta(x_t,t)) 
$$

---

## Reference
- Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in neural information processing systems 33 (2020): 6840-6851.
- Luo, Calvin. "Understanding diffusion models: A unified perspective." arXiv preprint arXiv:2208.11970 (2022).
