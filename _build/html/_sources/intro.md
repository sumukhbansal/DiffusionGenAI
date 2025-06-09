# Diffusion-Based Generative AI

Generative AI has made groundbreaking strides in recent years, giving machines the power to generate photorealistic images, produce lifelike videos, synthesize expressive speech, and even enhance or rewrite natural language. Among the leading techniques behind this wave of innovation are **diffusion models** — a family of probabilistic generative models that learn to create data by **reversing the process of noise corruption**.

Unlike GANs or VAEs, which attempt to map directly from noise to data, **diffusion models simulate a step-by-step denoising process**, making the generation more controllable, interpretable, and stable to train. This has positioned them as foundational models for multi-modal generation tasks, powering the next generation of AI creativity tools.

In this tutorial, we aim to provide comprehensive details on different aspects of diffusion models.

--- 

The material in this tutorial is based on a several wonderful existing courses and tutorials, which are noted below. 

1. CS492(D): Diffusion Models and Their Applications, Minhyuk Sung, KAIST, Fall 2024
    https://mhsung.github.io/kaist-cs492d-fall-2024/
2. Introduction to Flow Matching and Diffusion Models
    https://diffusion.csail.mit.edu/
3. 6.S183: A Practical Introduction to Diffusion Models, IAP 2025
    https://www.practical-diffusion.org/lectures/ 

---