# 7. Discrete Diffusion
So far, our focus has been on data represented in continuous space. However, many real-world data modalities—such as graphs, speech, text, and even images—are inherently discrete or can be effectively modeled in a discrete domain. For such modalities, the diffusion process must be defined differently than it is in continuous space.

Among discrete data types, text is one of the most prominent. Traditionally, text modeling has relied heavily on auto-regressive (AR) methods. Given an input text sequence $x = \{x^1x^2,..., x^d\}$, an AR model defines the joint probability $p_\theta(x)$ using the chain rule as:

$$
p_\theta(x) = p_\theta(x^1x^2...x^d) =p_\theta(x^1) p_\theta(x^2|x^1)p_\theta(x^d|x^1x^2...x^{d-1}) 
$$

Auto-regressive models come with several advantages and limitations:
- Pros: 
    - Highly scalable.
    - Capable of modeling complex probability distributions.
    - Provide a strong inductive bias for language modeling tasks. 
- Cons:
    - Prone to sampling drift over long sequences.
    - Require iterative decoding, leading to slow generation speeds.
    - Architecturally constrained for non-sequential or non-language discrete tasks.
    - Not well-suited as a general-purpose inductive bias beyond language modeling.
 
Recently, diffusion-based language models have emerged as strong alternatives, showing competitive performance across several benchmarks. These models address key shortcomings of AR methods, such as temporal quality degradation and the reverse curse of sequence length [Nie et al., 2024].

In the following section, we provide a comprehensive overview of discrete diffusion models, their formulation, and their growing relevance in modeling discrete data modalities.

---

## Discrete Diffusion Process
To model discrete data, we consider probability distributions defined over a finite support $\mathcal{X}=\{1,...,d\}$, where $d$ denotes the number of possible discrete states. These distributions can be represented as probability mass vectors $p \in \mathbb{R}^d$, where all entries are non-negative and sum to 1.

Following [Campbell et al. 2022], the discrete diffusion process evolves a family of distributions $p_t \in \mathbb{R}^d$ over time, governed by a continuous-time Markov process (CTMP) described by a linear ordinary differential equation (ODE):

$$
\frac{dp_t}{dt} = Q_tp_t, \quad p_0\approx p_{data}
$$

Here, $Q_t$ denotes the diffusion rate matrix with the following properties:
- $Q_t \in \mathbb{R}^{d\times d}$
- All off-diagonal elements are non-negative.
- The columns of $Q_t$ sum to zero, which implies that total mass is conserved.
- $Q_t$ controls the transition from one state to another state, i.e. $Q_t(x,y)$ define the transition rate from $x$ to $y$. 
- Typically, $Q_t$ is time scaled as $Q_t = \sigma(t) Q$, ensuring convergence to a stationary distribution $p_{base}$ as $t \to \infty$.

The process can be approximated for a small time step $\Delta t$ using:

$$
p_{t+\Delta t}(y| x) = \delta_{yx} + Q_t(y,x) \Delta t + O(\Delta t^2), \quad \quad \delta: \text{Kronecker delta}
$$

This can be expressed more explicitly as: 

<!-- $$
\begin{align*}
p_{t+\Delta t}(y| x) =
\left\{
    \begin {aligned}
        & Q_t(y,x) \Delta t + o(\Delta t), &&\text{ if } y \neq x,\\
        & 1 + Q_t(y,x) \Delta t + o(\Delta t),  &&\text{ if } y = x,
    \end{aligned}
\end{align*}
$$ -->
$$
p_{t+\Delta t}(y \mid x) =
\begin{cases}
    Q_t(y, x)\, \Delta t + o(\Delta t), & \text{if } y \neq x, \\
    1 + Q_t(y, x)\, \Delta t + o(\Delta t), & \text{if } y = x,
\end{cases}
$$

Inversely, the rate matrix $Q_t$ can be derived from the transition probabilities as

<!-- $$
\begin{align*}
Q_t(y,x) =
\left\{
    \begin {aligned}
        & \lim_{\Delta t \to 0} \frac{p_{t+\Delta t}(y| x)}{\Delta t}  , &&\text{ if } y \neq x,\\
        & \lim_{\Delta t \to 0} \frac{p_{t+\Delta t}(x|x)-1}{\Delta t},  &&\text{ if } y = x,
    \end{aligned}
\end{align*}
$$ -->

$$
Q_t(y, x) =
\begin{cases}
    \lim_{\Delta t \to 0} \dfrac{p_{t+\Delta t}(y \mid x)}{\Delta t}, & \text{if } y \neq x, \\
    \lim_{\Delta t \to 0} \dfrac{p_{t+\Delta t}(x \mid x) - 1}{\Delta t}, & \text{if } y = x,
\end{cases}
$$


The reverse-time diffusion process is governed by a time-reversed diffusion matrix $\bar{Q}_t$, as described in [Sun et al. 2022]. The dynamics are given by:

$$
\begin{align*}
\frac{dp_{T-t}}{dt} &= \bar{Q}_{T-t} p_{T-t}
\end{align*}
$$


The entries of the reverse rate matrix $\bar{Q}_t$ are defined as:

<!-- $$
\begin{align*}
\bar{Q}_t(x, y) &=
\left\{
    \begin {aligned}
        & \frac{p_t(y)}{p_t(x)} Q_t(y,x) , &&\text{ if } y \neq x,\\
        & -\sum_{k\neq x} \bar{Q}_t(k,x)  &&\text{ if } y = x,
    \end{aligned}
\end{align*}
$$ -->

$$
\bar{Q}_t(x, y) =
\begin{cases}
    \frac{p_t(y)}{p_t(x)} Q_t(y, x), & \text{if } y \neq x, \\
    -\sum_{k \neq x} \bar{Q}_t(k, x), & \text{if } y = x,
\end{cases}
$$

Note that computing the reverse transition rates $\bar{Q}_t(x, y)$ requires knowledge of the marginal probability ratio $\frac{p_t(y)}{p_t(x)}$. To estimate this, a neural network can be trained to approximate the score function, which encodes this ratio—analogous to score-based diffusion models in continuous domains.

---

## Discrete Diffusion Models Based on Score Entropy
In the discrete diffusion framework, the key modeling objective is to estimate the transition probability ratio  $\frac{p_t(y)}{p_t(x)}$ between discrete states. To achieve this, score-based methods are adapted to discrete domains using a formulation known as the Concrete Score, which generalizes the continuous score (log-density gradient) to discrete settings.

## Concrete Score
The Concrete Score serves as a discrete analogue of the continuous score function. Instead of using derivatives, it relies on probability ratios within a neighborhood of a state.

The discrete gradient of a function 
f(x) is defined over a neighborhood $N(x)$ as:

$$
\nabla f(x) = [f(y)- f(y)]_{y \in N(x)}
$$

Applying this to the log-density, we get:

$$
\begin{align*}
\nabla_x \log p_t &= \frac{\nabla p_t(x)}{p_t(x)} \quad \quad \text{Using the chain rule of log derivative}\\
& = \underbrace{\left[ \frac{p_t(y)}{p_t(x)}\right]_{y \in N(x)}}_{\text{Concrete Score}} -1\\
\end{align*}
$$

The ratio is referred to as the **concrete score**.

### Discrete Score Entropy 
Several training objectives have been proposed to estimate the Concrete Score in practice.
- **Concrete Score Matching.** 
    In [Meng et al. 2022], authors proposed to train a network $s_\theta$ to approximate $\frac{p_t(y)}{p_t(x)}$, minimizing the loss:
    
    $$ 
    \mathcal{L}_{CSM} = \frac{1}{2} \mathbb{E}_{x\sim p_t} \left[ \sum_{y\neq x} \left(s_\theta(x_t,t)_y - \frac{p_t(y)}{p_t(x)} \right)^2 \right] 
    $$
    
    However, the $L^2$ objective is not robust to negative or zero values, which can lead to divergent behavior.

- **Score entropy.** 
    To address the drawbacks of CSM, Score Entropy introduces a principled modification by minimizing:    

    $$
     \mathcal{L}_{SE} = \mathbb{E}_{x\sim p} \left[ \sum_{y\neq x} \left(s_\theta(x)_y - \frac{p(y)}{p(x)} \log s_\theta (x)_y \right)\right] 
    $$
    
    Taking the derivative with respect to $\theta$:
    
    $$
    \frac{d}{d\theta}\left(s_\theta(x)_y - \frac{p(y)}{p(x)} \log s_\theta (x)_y \right)
    $$
    
    Setting the derivative to zero gives::
    
    $$
    1 - \frac{p(y)}{p(x)} \frac{1}{s_\theta (x)_y} = 0 \quad \quad \implies \quad \quad s_\theta (x)_y = \frac{p(y)}{p(x)}  
    $$
    
    To ensure non-negativity and improve optimization stability, a normalized version of the score entropy is defined:
    
    $$
    \mathcal{L}_{SE} = \mathbb{E}_{x\sim p} \left[ \sum_{y\neq x} w_{xy}\left(s_\theta(x)_y - \frac{p(y)}{p(x)} \log s_\theta (x)_y + K \left(\frac{p(y)}{p(x)} \right) \right) \right] 
    $$
    
    where $K(a) = a(\log a - 1)$ ensures that $\mathcal{L}_{SE} \geq 0$, and $w_{xy} \geq 0$ are weighting terms. 
    
- For the weights $w_{xy} = 1$, the gradient of $\mathcal{L}_{SE}$ simplifies to:
    $$
    \nabla s_\theta(x)_y \mathcal{L}_{SE} = \frac{1}{s_\theta(x)_y} \mathcal{L}_{CSM}
    $$
    
    showing that SE rescales gradients adaptively based on output magnitude, improving convergence behavior.
    
### Implicit and Denoising Score Entropy
Since $\frac{p(y)}{p(x)}$ is typically unknown, two alternatives are proposed to make training tractable:

- **Implicit Score Entropy.**
    Score entropy is reformulated (up to a constant) as:
    
    $$
    \mathcal{L}_{ISE} =    \mathbb{E}_{x\sim p} \left[ \sum_{y\neq x} w_{xy}s_\theta(x)_y -w_{yx}s_\theta(y)_x  \right] 
    $$
    
    Even for implicit score entropy, a typical Monte Carlo estimate would require sampling an $x$ and evaluating $s_\theta(y)_x$ for all other $y$, which is intractable for high dimensions.
        
- **Denoising Score Entropy.** 
    In denoising score entropy, $p$ is assumed to be generated by perturbing a a base density $p_0$: 

    $$
    p(x) = \sum_{x_0} p(x|x_0)p_0(x_0)
    $$
        
    Then,
    
    $$ 
    \begin{align*}
        \mathbb{E}_{x\sim p} \sum_{y\neq x}  \frac{p(y)}{p(x)} \log s_\theta(x)_y
        &= \sum_{x} \sum_{y\neq x} p(y) \log s_\theta(x)_y  \quad \text{using expectation definition} \\
        &= \sum_{x} \sum_{y\neq x} \log s_\theta(x)_y \sum_{x_0} p(y|x_0)p_0(x_0)\\
        &= \sum_{x_0}\sum_{x} \sum_{y\neq x} \log s_\theta(x)_y \frac{p(y|x_0)}{p(x|x_0)}   p(x|x_0)p_0(x_0)\\
        &= \mathbb{E}_{x_0\sim p_0, x\sim p(.|x_0) } \sum_{y\neq x}\frac{p(y|x_0)}{p(x|x_0)}  \log s_\theta(x)_y 
        \end{align*}
    $$

    where $\frac{p(y|x_0)}{p(x|x_0)}$ is possible to compute making the problem tractable.
    The score entropy $\mathcal{L}_{SE}$ is equivalent (up to a constant independent of $\theta$) to the denoising score entropy $\mathcal{L}_{DSE}$ given as 
    
    $$
    \mathcal{L}_{DSE} = 
    \underset{x\sim p(.|x_0)}{\underset{{x\sim p}}{\mathbb{E}}} \left[ \sum_{y\neq x} w_{xy}\left(s_\theta(x)_y - \frac{p(y|x_0)}{p(x|x_0)} \log s_\theta (x)_y \right) \right] 
    $$
    
---

## Practical Considerations
To make the problem tractable, in practice, ratios between the sequences related by only one hamming distance are considered. For example,  If sequences are denoted by $x = x^1x^2...x^d$, the score model is defined as a sequence-to-sequence function:

$$
s_\theta(.,t):{1,...,n}^d \to \mathbb{R}^{d\times n}
$$

with entries:
$$
(s_\theta(x, t))_{i, y^i} \approx \frac{p_t(x^1...y^i...x^d)}{p_t(x^1...x^i...x^d)}
$$

The discrete diffusion matrix $Q$ matrix is defined as follows:

$$
Q_t(x^1...x^i...x^d, x^1...y^i...x^d ) = Q_t^{tok}(x^i, y^i), \quad \quad \text{ for x and y differing at position i}
$$

To keep the transition stable across time for a noise level $\sigma$ and fixed $Q^{tok}$, 

$$
Q_t^{tok} = \sigma(t) Q^{tok}
$$

There are two popular methods to define the diffusion transition matrix $Q$. While in one case the target distribution is a uniform distribution, in the other case the diffusion is designed to lead to a absorbing / mask state represented by **[M]**. $Q$ matrix for both the cases defined as follows: 

$$
Q_{uniform} = \begin{bmatrix}
1-N & 1 & ... & 1\\
1 & 1-N & ... & 1\\
\vdots & \vdots & \ddots & \vdots\\
1 & 1 &... & 1-N
\end{bmatrix}
\quad \quad
Q_{absorb} = \begin{bmatrix}
-1 & 0 & ... & 0 & 0\\
0 & -1 & ... & 0 & 0\\
\vdots & \vdots & \ddots & \vdots & \vdots\\
0 & 0 & ... & -1 & 0\\
1 & 1 & ... & 1 & 0
\end{bmatrix} 
$$

---

## Masked Diffusion Models ( Absorbing State )
Masked diffusion / absorbing state diffusion methods introduce a special token, denoted as **[M]** (for "mask"), into the set of possible states. Each token in the sequence has a certain probability of transitioning into this mask state over time. In absorbing state methods, Once a token transitions to **[M]**, it remains there permanently—hence the term absorbing state.

### Forward Process
Let $\alpha_t$ denote the masking schedule, representing the expected proportion of unmasked tokens at time $t$.
The forward transition probability from time $s<t$ is defined as: 

<!-- $$
\begin{align*}
q(x_t|x_s) = 
\left\{
    \begin {aligned}
        & \frac{\alpha_t}{\alpha_s}, \text{if the token remains unmasked}, \\
        & 1 - \frac{\alpha_t}{\alpha_s}, \text{if the token transition to \textbf{[M]}},\\
    \end{aligned}
\end{align*}
$$ -->

$$
q(x_t \mid x_s) = 
\begin{cases}
    \frac{\alpha_t}{\alpha_s}, & \text{if the token remains unmasked}, \\
    1 - \frac{\alpha_t}{\alpha_s}, & \text{if the token transitions to } \mathbf{[M]},
\end{cases}
$$

The corresponding transition matrix $Q(s,t)$ can be expressed as:

$$
Q(s,t) = \frac{\alpha_t}{\alpha_s} \mathbf{I} + \left(1-\frac{\alpha_t}{\alpha_s}\right) \mathbf{I} e^T_m 
$$

Here:
- $\mathbf{I}$ is the identity matrix of size $d \times d$.
- $e_m$ is a one-hot vector indicating the mask token $\textbf{[M]}$.


### Reverse Process
The reverse process involves recovering an unmasked token from a masked state. Assuming access to the original token $x_0$, the reverse transition probability from time $t$ to $s$ ($s<t$) is given by: 

<!-- $$
\begin{align*}
q(x_s|x_t, x_0)) = 
\left\{
    \begin {aligned}
        & \frac{\alpha_s -\alpha_t}{1-\alpha_t}, \text{if the token is unmasked}, \\
        & \frac{1-\alpha_s}{1-\alpha_t}, \text{if the token remains mask},\\
    \end{aligned}
\end{align*}
$$ -->

$$
\begin{align*}
q(x_s \mid x_t, x_0) = 
\begin{cases}
    \frac{\alpha_s - \alpha_t}{1 - \alpha_t}, & \text{if the token is unmasked}, \\
    \frac{1 - \alpha_s}{1 - \alpha_t}, & \text{if the token remains mask},
\end{cases}
\end{align*}
$$

The corresponding reverse transition matrix, conditioned on $x_0$ is:

$$
\bar{R}^{x_0}(t,s) = \mathbf{I} + \left(\frac{\alpha_s-\alpha_t}{1- \alpha_t}\right) e_m(x_0-e_m)^T 
$$

This formulation leverages the known original token $x_0$ to define a deterministic direction for denoising masked entries.

### Generative Process
In practice, the original clean token $x_0$ is not available during inference. Therefore, the generative model approximates the reverse process by learning a neural network that predicts a distribution over possible original tokens.
The generative process is defined as:

$$
\begin{align*}
p_\theta(x_s|x_t) \overset{\Delta}{=} q(x_s|x_t, \mu_\theta(x_t,t) 
\end{align*}
$$

Here: 
- $\mu_\theta(x_t,t) \in \mathbb{R}^d$ is a probability mass vector over possible tokens, which is parameterized by a neural network.
- This is known as mean-parametrization as neural network predict a mean pf $x_0$.










## References
- Lou, Aaron, Chenlin Meng, and Stefano Ermon. "Discrete diffusion modeling by estimating the ratios of the data distribution." arXiv preprint arXiv:2310.16834 (2023).
- Arriola, Marianne, et al. "Block diffusion: Interpolating between autoregressive and diffusion language models." arXiv preprint arXiv:2503.09573 (2025).
- Nie, Shen, et al. "Scaling up Masked Diffusion Models on Text." arXiv preprint arXiv:2410.18514 (2024).
- Sun, Haoran, et al. "Score-based continuous-time discrete diffusion models." arXiv preprint arXiv:2211.16750 (2022).
- Ou, Jingyang, et al. "Your absorbing discrete diffusion secretly models the conditional distributions of clean data." arXiv preprint arXiv:2406.03736 (2024).