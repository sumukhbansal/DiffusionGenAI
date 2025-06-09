# Denoising Diffusion Implicit Models (DDIM)
The Denoising Diffusion Probabilistic Model (DDPM) is based on a Markovian forward process, which results in a sequential and slow generation process:

$$
q(x_{1:T}|x_0)= \prod_{t=1}^Tq(x_t|x_{t-1})
$$

Since each step depends only on the previous one, generating a sample requires progressing through all T steps, typically 1000 or more.

The obivious question is can one speed Up sampling?
By adopting a non-Markovian forward process, as in DDIM (Denoising Diffusion Implicit Models), a faster sampling can be achieved. This enables a deterministic or accelerated sampling strategy. 

---

## DDIM Forward Process (Non-Markovian)
DDIM proposes a different forward process:

$$
q_\sigma(x_{1:T}|x_0)= q_\sigma(x_T|x_0)\prod_{t=2}^Tq(x_{t-1}|x_{t},x_0)
$$

In the forward process,
- $x_t$ is sampled from $x_0$ first.
- Each $x_{t-1}$ is sampled from $x_t$, $x_0$ (non-Markovian).

## Defining $q_\sigma(x_{t-1}|x_t,x_0)$.
Let $q_\sigma(x_{t-1}|x_t,x_0)$ be a Gaussian with a linear mean and variance $\sigma_t^2$:

$$
q_\sigma(x_{t-1}|x_t,x_0) = \mathcal{N}(\omega_0x_0+\omega_tx_t+b, \sigma_t^2 \mathbf{I})
$$

The aim is to match marginal:

$$
q_\sigma(x_t|x_0)=\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t) \mathbf{I})
$$

---

## How to find $\omega_0, \omega_1$ and $b$?
**Induction:** $q_\sigma(x_t|x_0)=\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t) \mathbf{I})$. What should be $\omega_0, \omega_1$ and $b$ in order to ensure that 

$$
q_\sigma(x_{t-1}|x_0)=\mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}x_0, (1-\bar{\alpha}_{t-1})\mathbf{I})
$$

**Hint-1:**

$$
\begin{align*}
q_\sigma(x_{t-1}|x_t,x_0) &= \mathcal{N}(\omega_0x_0+\omega_tx_t+b, \sigma_t^2 \mathbf{I})\\
q_\sigma(x_t|x_0)&=\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t) \mathbf{I})\\
\text{Marginalization} \implies  q_\sigma(x_{t-1}|x_0) &= \int q_\sigma(x_t|x_0) q_\sigma(x_{t-1}|x_t,x_0) dx_t
\end{align*}
$$

**Hint-2:** When $p(x) = \mathcal{N}(\mu, \sigma_x^2  \mathbf{I})$ and $p(y|x) = \mathcal{N}(ax+b, \sigma_y^2  \mathbf{I})$, then

$$
p(x) = \int p(x) p(y|x) dx = \mathcal{N}(a\mu + b, (\sigma_y^2+ a^2\sigma_x^2)\mathbf{I})
$$

$$
\begin{align*}
q_\sigma(x_t|x_0)=&\mathcal{N}(\omega_0x_0+\omega_t(\sqrt{\bar{\alpha}_t}x_0)+ b, (\sigma_t^2 + \omega_t^2(1-\bar{\alpha}_t)) \mathbf{I})\\
=&\mathcal{N} (\sqrt{\bar{\alpha}_{t-1}}x_0, (1-\bar{\alpha}_{t-1}) \mathbf{I})
\end{align*}
$$

$$
\begin{align*}
\omega_t &= \sqrt{\frac{1-\bar\alpha_{t-1}-\sigma_t^2}{1-\bar\alpha_{t}}}\\
\omega_0 &= \sqrt{1-\bar\alpha_{t-1}}- \sqrt{\bar\alpha_{t}}\sqrt{\frac{1-\bar\alpha_{t-1}-\sigma_t^2}{1-\bar\alpha_{t}}}\\
b&=0
\end{align*}
$$

$$
\begin{align}
q_\sigma(x_{t-1}|x_t,x_0) &= \mathcal{N}(\omega_0x_0+\omega_tx_t+b, \sigma_t^2 \mathbf{I})\\
&= \mathcal{N}(\sqrt{\bar\alpha_{t-1}}x_0 - \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\frac{(x_t-\sqrt{\bar\alpha_{t}}x_0)}{\sqrt{1-\bar\alpha_{t}}}, \sigma_t^2 \mathbf{I})
\end{align}
$$

With arbitrary $\sigma_t^2$, $q_\sigma(x_t|x_0)$ remains same as DDPM.
To summarize, 
- **In DDPM:** $q_\sigma(x_t|x_{t-1})$ is defined and $q_\sigma(x_t|x_0)$ \&  $q_\sigma(x_{t-1}|x_{t},x_0)$ are derived.
- **In DDIM:**  $q_\sigma(x_{t-1}|x_{t},x_0)$ is defined and $q_\sigma(x_t|x_0)$ \&  $q_\sigma(x_t|x_)$ are derived.

**What if we set $\sigma_t^2=0$? Then, forward and reverse processes become deterministic.**

Loss function: The noise predictor $\hat{\epsilon}_\theta(x_t,t)$ trained for DDPM can be directly used in the DDIM reverse process.

---

**DDPM Reverse process:**

$$
q(x_{t-1}|x_t,x_0) = \mathcal{N}\left( \frac{\sqrt{\alpha}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_{t}} x_0, \left(\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}} \beta_t\right) \mathbf{I} \right)
$$

For each time step t=T,...,1 repeat:
- Compute $x_{0|t} = \frac{1}{\sqrt{\bar{\alpha}_t}}( x_t - \sqrt{1-\bar{\alpha}_t} \hat\epsilon_\theta(x_t,t))$
- Compute $\tilde{\mu}= \frac{\sqrt{\alpha}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_{t}} x_{0|t}$
- Sample $z_t \sim \mathcal{N}(0,\mathbf{I})$
- Compute $x_{t-1} = \bar{\mu}+ \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}} \beta_t} z_t$


---

**DDIM Reverse process:**

$$
\begin{align*}
q_\sigma(x_{t-1}|x_t,x_0)= \mathcal{N}(\sqrt{\bar\alpha_{t-1}}x_0 - \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\frac{(x_t-\sqrt{\bar\alpha_{t}}x_0)}{\sqrt{1-\bar\alpha_{t}}}, \sigma_t^2 \mathbf{I})
\end{align*}
$$


For each time step t=T,...,1 repeat:
- Compute $x_{0|t} = \frac{1}{\sqrt{\bar{\alpha}_t}}( x_t - \sqrt{1-\bar{\alpha}_t} \hat\epsilon_\theta(x_t,t))$
- Compute $\tilde{\mu}= \sqrt{\bar\alpha_{t-1}}x_0 - \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\frac{(x_t-\sqrt{\bar\alpha_{t}}x_{0|t})}{\sqrt{1-\bar\alpha_{t}}}$
- Sample $z_t \sim \mathcal{N}(0,\mathbf{I})$
-Compute $x_{t-1} = \bar{\mu}+ \sigma_t z_t$

DDPM can be seen as a special case of DDIM when

$$ 
\sigma_t^2 = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_{t}}\beta_t
$$

In this case DDIM operates as a Markovian process. 
To control Stochasticity, $\sigma_t$ can be parametrized as $\sigma_t = \eta  \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}} \beta_t}$
Where $\eta=0$ gives a deterministic process and $\eta=1$ gives DDPM.

---

## Accelerating Sampling Process
The DDPM/DDIM reverse process with the full sequence of time steps $t \in [1,2,...,T]$:

$$
p_\theta(x_{0|T}) = p_\theta(x_T) \prod_{t=1}^T p_\theta(x_{t-1}|x_t)
$$

Consider a sub-sequence of the time steps: $\tau = [\tau_1, \tau_2, ..., \tau_S]$
The reverse process for this sub-sequence

$$
p_\theta(x_{\tau}) = p(x_T) \prod_{t=1}^S p_\theta(x_{\tau_{i-1}}|x_{\tau_{i}})
$$

is optimized using the same objective function as in the full sequence.
As smaller time steps are used, the quality of the generated data can worsen. However, quality degradation is mitigated when the DDIM reverse process becomes more deterministic.