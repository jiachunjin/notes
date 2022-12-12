[toc]



## Introduction

The Thermodynamic Variational Objective (TVO) builds a tighter lower bound on the log evidence $\log p(\mathbf{x})$ than the common evidence lower bound (ELBO) when using the same variational family. By analyzing the gap between the variational distribution $q$ and the true posterior $p$ in TVO, a the connection between the Fisher information and the Jeffrey divergence (a.k.a the symmetric KL divergence) can be drawn. The TVO also has close connection with several approximation inference methods like annealed importance sampling (AIS), MCMC Variational Inference (MCMC-VI) and Variation Inference with Variational Contrastive Divergence (VCD).



## Lower bound of the log evidence

Variational Inference (VI) tries to find a close approximation of the posterior distribution $p(\mathbf{z}|\mathbf{x}) = p(\mathbf{x}, \mathbf{z})/p(\mathbf{x})$ with a parametric variational distribution $q_\phi(\mathbf{z})$ (or $q_\phi(\mathbf{z}|\mathbf{x})$ in amortized inference). The traditional variational inference tries to minimize the KL divergence between $q_\phi(\mathbf{z}|\mathbf{x})$ and $p(\mathbf{z}|\mathbf{x})$, which is equivalent to maximizing $\log p(\mathbf{x}) - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x})\|p(\mathbf{z}|\mathbf{x})) = \int_\mathbf{z} q_\phi(\mathbf{z}|\mathbf{x})\log \frac{p(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}d\mathbf{z} = \mathbb{E}_{q_\phi}\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right]$. This objective is known as the evidence lower bound (ELBO) because of the non-negativity of the KL divergence, and people always use it as a proxy to approximately do maximum likelihood computation when $p_\theta(\mathbf{x})$ is defined as a generative model with latent variable $\mathbf{z}$. A broad consensus is that the tighter the lower bound is, the better we can approximate the true posterior distribution $p(\mathbf{z}|\mathbf{x})$ with $q_\phi(\mathbf{z}|\mathbf{x})$ (there is some exception when $p_\theta(\mathbf{x})$ is taken as a generative model like VAE and trained with the IWAE objective, see [^tighter not better]).



## Thermodynamic Integration

Thermodynamic Integration (TI) is a technique used to estimate the partition function ratio of two unnormalized distributions. Suppose we have two densities $\pi_0(\mathbf{z}) = {\tilde{\pi_0}(\mathbf{z})}/{Z_0}$ and $\pi_1(\mathbf{z}) = {\tilde{\pi_1}(\mathbf{z})}/{Z_1}$, we would like to compute ${Z_1}/{Z_0}$ but we cannot directly calculate these two constants. To apply TI, we first need to construct a path between $\pi_0(\mathbf{z})$ and $\pi_1(\mathbf{z})$ via a parameter $\beta \in (0, 1)$:
$$
\pi_\beta = \frac{\tilde{\pi}_\beta(\mathbf{z})}{Z_\beta}, \tilde{\pi}_\beta(\mathbf{z}) = \pi_0^{1-\beta}(\mathbf{z})\pi_1^{\beta}(\mathbf{z}), Z_\beta = \int \pi_0^{1-\beta}(\mathbf{z})\pi_1^{\beta}(\mathbf{z}) d\mathbf{z}\label{eq.path},
$$
we can see something interesting when inspecting $\log Z_\beta$'s  derivative with respect to $\beta$:
$$
\begin{aligned}
\frac{\partial \log Z_{\beta}}{\partial \beta} &=\frac{1}{Z_{\beta}} \frac{\partial}{\partial \beta} Z_{\beta} \\
&=\frac{1}{Z_{\beta}} \frac{\partial}{\partial \beta} \int \tilde{\pi}_{\beta}(\mathbf{z}) \mathrm{d} \mathbf{z} \\
&=\int \frac{1}{Z_{\beta}} \frac{\partial}{\partial \beta} \tilde{\pi}_{\beta}(\mathbf{z}) \mathrm{d} \mathbf{z} \\
&=\int \frac{\tilde{\pi}_{\beta}(\mathbf{z})}{Z_{\beta}} \frac{\partial}{\partial \beta} \log \tilde{\pi}_{\beta}(\mathbf{z}) \mathrm{d} \mathbf{z}\\
&= \mathbb{E}_{\pi_\beta}[\frac{\partial}{\partial \beta} \log \tilde{\pi}_{\beta}(\mathbf{z})],
\end{aligned}
$$
then if we integrate both side and apply the fundamental theorem of calculus:
$$
\log \frac{Z_1}{Z_0} = \int_0^1 \frac{\partial \log Z_{\beta}}{\partial \beta} d\beta = \int_0^1 \mathbb{E}_{\pi_\beta}[\frac{\partial}{\partial \beta} \log \tilde{\pi}_{\beta}(\mathbf{z})]\label{eq.TI},
$$
which transforms the log ratio of two partition functions into a 1-D integral from 0 to 1.



## VI with TI

The key insight connecting TI and VI is to set the two unnormalized distribution in TI as the following:
$$
\begin{aligned}
&\tilde\pi_0(\mathbf{z}) = q_\phi(\mathbf{z}|\mathbf{x}), Z_0 = \int q_\phi(\mathbf{z}|\mathbf{x}) d\mathbf{z} = 1,\\
&\tilde\pi_1(\mathbf{z}) = p(\mathbf{x}, \mathbf{z}), Z_1 = \int p(\mathbf{x}, \mathbf{z}) d\mathbf{z} = p(\mathbf{x}),
\end{aligned}
$$
one great thing is the term $\frac{\partial}{\partial \beta} \log \tilde{\pi}_\beta(\mathbf{z})=\log \frac{\tilde{\pi}_1(\mathbf{z})}{\tilde{\pi}_0(\mathbf{z})} = \log \frac{p(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}$ is a constant for every $\beta$. Plug the above setting into $\eqref{eq.TI}$ , we reach the Thermodynamic Variational Identity (TVI):
$$
\log p(\mathbf{x}) = \int_0^1\mathbb{E}_{\pi_\beta}\left[\log\frac{p(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right]d\beta.
$$
When discretizing the integration with $K$ left Riemann sum approximation to the TVI, we come to the TVO:
$$
\frac{1}{K}\left[\sum_{k=0}^{K-1} \mathbb{E}_{\pi_{\beta_{k}}}\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\right]\right] \leq \int_{0}^{1} \mathbb{E}_{\pi_{\beta}}\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\right] d \beta=\log p(\mathbf{x}).
$$
And when $K = 1$, we come back to the ELBO: $\mathbb{E}_{q_\phi}\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right] \leq \log p(\mathbf{x})$.

Next, I'd like to 1) show that the integrands of TI is keep increasing which makes the TVO always tighter than the ELBO and 2) give a visualization of TVO to further analyze the lower bound gap.



## TVO is always tighter than ELBO

### The path exponential family

A distribution from the exponential family can be expressed as:
$$
p_\theta(\mathbf{z}) = h(\mathbf{z})\exp\{\left<\theta, \phi(\mathbf{z})\right> - A(\theta)\},
$$
where $A(\theta) = \log \int h(\mathbf{z})\exp\{\left<\theta, \phi(\mathbf{z})\right>\}d\mathbf{z}$ is called the log partition function, $\phi(\mathbf{z})$ is called the *sufficient statistics* and $\theta$ is called the *canonical parameters*. In the following, I will interchangeably use the term *log evidence* and *log partition function* since they represent similar concept in this note. The key point here is that every intermediate distribution defined in $\eqref{eq.path}$ can be expressed as a distribution from the exponential family, and thus we can analyze the intermediate distribution with the properties of the exponential family. We construct the following intermediate distribution:

$$
\begin{aligned}
\pi_{\beta}(\mathbf{z} | \mathbf{x}):&=\pi_{0}(\mathbf{z} | \mathbf{x}) \exp \{\beta \cdot T(\mathbf{x}, \mathbf{z})-\psi(\mathbf{x} ; \beta)\} \\
&= \frac{1}{Z_\beta}\pi_0(\mathbf{z}|\mathbf{x})^{1-\beta}\pi_1(\mathbf{z}|\mathbf{x})^\beta,
\end{aligned}
$$
where $T(\mathbf{x}, \mathbf{z}):=\log \frac{\pi_1(z|x)}{\pi_0(z|x)}$, and $\psi(\mathbf{x};\beta) = \log Z_\beta(\mathbf{x}) = \log\int_z \pi_0(z|x)\exp\{\beta \cdot T(x,z)\}dz$.

### Local evidence as the mean parameters

Define $\eta_\beta = \frac{\partial}{\partial \beta} \psi(\mathbf{x; \beta}) = \frac{\partial}{\partial \beta} \log Z_\beta(\mathbf{x})$, which is called the *local evidence* by the authors of TVO. This term is also called the *mean parameter* in the exponential family context, then we have the following:
$$
\log p(\mathbf{x}) = \psi(1)-\psi(0)=\int_{0}^{1} \eta_{\beta} d \beta=\int_{0}^{1} \mathbb{E}_{\pi_{\beta}}\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\right] d \beta
$$

### Convexity of the log partition function

>**Claim: the log partition function $A(\theta)$ of exponential family is convex with respect to $\theta$**.
>
>Proof:
>
>Here, we show the Hessian matrix of $A(\theta)$ is a covariance matrix and thus positive-semidefinite:
>$$
>\begin{aligned}
>\frac{\partial}{\partial \theta_\alpha}A(\theta) 
>&= \frac{\partial}{\partial \theta_\alpha}\log \int h(\mathbf{z})\exp\{\left<\theta, \phi(\mathbf{z})\right>\}d\mathbf{z}\\
>&= \frac{\int \phi_\alpha(\mathbf{z}) h(\mathbf{z})\exp\{\left<\theta, \phi(\mathbf{z})\right>\}}{Z}\\
>&= \mathbb{E}_p\left[\phi_\alpha(\mathbf{z})\right],
>\end{aligned}
>$$
>and we further have:
>$$
>\begin{aligned}
>& \quad \frac{\partial^2}{\partial \theta_\beta\partial\theta_\alpha}A(\theta)\\
>&= \int \phi_\alpha(\mathbf{z})\frac{\partial}{\partial \theta_\beta}\frac{h(\mathbf{z})\exp\{\left<\theta, \phi(\mathbf{z})\right>\}}{\int h(\mathbf{u})\exp\{\left<\theta, \phi(\mathbf{u})\right>\}d\mathbf{u}}dx\\
>&= \int \phi_\alpha(\mathbf{z}) h(\mathbf{z}) \frac{\phi_\beta(\mathbf{z})\exp\{\left<\theta, \phi(\mathbf{z})\right>\}Z - \exp\{\left<\theta, \phi(\mathbf{z})\right>\}\int h(\mathbf{u})\phi_\beta(\mathbf{u})\exp\{\left<\theta, \phi(\mathbf{u})\right>\}d\mathbf{u}}{Z^2}d\mathbf{z}\\
>&= \int p_\theta(\mathbf{z})\phi_\alpha(\mathbf{z})\phi_\beta(\mathbf{z})d\mathbf{z} - \int \frac{1}{Z^2}\phi_\alpha(\mathbf{z})h(\mathbf{z}) \exp\{\left<\theta, \phi(\mathbf{z})\right>\int h(\mathbf{u})\phi_\beta(\mathbf{u})\exp\{\left<\theta, \phi(\mathbf{u})\right>\}d\mathbf{u} d\mathbf{z}\\
>&= \int p_\theta(\mathbf{z})\phi_\alpha(\mathbf{z})\phi_\beta(\mathbf{z})d\mathbf{z} - \left[\frac{1}{Z}\int h(\mathbf{u})\phi_\beta(\mathbf{u})\exp\{\left<\theta, \phi(\mathbf{u})\right>\}d\mathbf{u}\right]\left[\frac{1}{Z}\int \phi_\alpha(\mathbf{z})h(\mathbf{z})\exp\{\left<\theta, \phi(\mathbf{z})\right>d\mathbf{z}\right]\\
>&= \mathbb{E}_p[\phi_\alpha(Z)\phi_\beta(Z)] - \mathbb{E}_p[\phi_\alpha(Z)]\mathbb{E}_p[\phi_\beta(Z)]\\
>&= \operatorname{Cov}_p(\phi_\alpha(Z), \phi_\beta(Z)) \\
>& \geq 0,
>\end{aligned}\label{eq.hessianlogZ}
>$$



### The local evidence is non-decreasing in TVO

Since the integrand $\mathbb{E}_{\pi_{\beta}}\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\right]$ can be viewed as the *mean parameter* of the path exponential family, and we can use the convexity of the log partition function with respect to canonical parameter to show the local evidence is non-decreasing.

### The image

With the above discussion, we can easily have $\mathbb{E}_{\pi_{\beta}}\left[\log\frac{p(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right] \leq \mathbb{E}_{\pi_{\beta^\prime}}\left[\log\frac{p(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right]$ for every $0 \leq \beta  \leq \beta^\prime \leq 1,$ thus the TVO is a tighter lower bound than the ELBO and can be visualized as the following:

<img src="what is TVO.assets/image-20221008160914478-1665216565111-1.png" alt="image-20221008160914478" style="zoom:50%;" />

figure source: [^TVO]

Note that in real cases, the ELBO is always a negative term, thus the image will be like the following:

<img src="what is TVO.assets/image-20221009172824055.png" alt="image-20221009172824055" style="zoom:75%;" />

figure source: appendix of [^TVO]



## The gap between TVO and the log evidence

### Annealed Importance Sampling

Annealed importance sampling (AIS) [^AIS] provides an estimate of the log partition function $\log Z$ of an intractable target distribution $\tilde\pi_1(\mathbf{z}) / Z$ by sampling from a distribution path that interpolates between a tractable initial distribution $\pi_0(\mathbf{z})$ and the target. The procedure is given in the following pseudocode (with some transform of notations, $T_k(\mathbf{x}|\mathbf{x}_{k-1})$ denotes the MCMC transition operator which leaves $\pi_{\beta_k}$ invariant):

<img src="what is TVO.assets/image-20221009225129657.png" alt="image-20221009225129657" style="zoom:50%;" />

figure source: [^moment averaging path]

The log of final weight can be expanded as:
$$
\begin{aligned}
\log w_K &=\log w_{K-1}+\log \tilde\pi_{\beta_K}\left(\mathbf{z}_{K-1}\right)-\log\tilde\pi_{\beta_{K-1}}\left(\mathbf{z}_{K-1}\right) \\
&=\log w_{K-2}+\log \tilde\pi_{\beta_{K-1}}\left(\mathbf{z}_{K-2}\right)-\log \tilde\pi_{\beta_{K-2}}\left(\mathbf{z}_{K-2}\right)+\log \tilde\pi_{\beta_K}\left(\mathbf{z}_{K-1}\right)-\log \tilde\pi_{\beta_{K-1}}\left(\mathbf{z}_{K-1}\right) \\
& \cdots \\
&=\log w_0+\sum_{k=1}^K\left[\log \tilde\pi_{\beta_k}\left(\mathbf{z}_{k-1}\right)-\log \tilde\pi_{\beta_{k-1}}\left(\mathbf{z}_{k-1}\right)\right],
\end{aligned}\label{eq.AIS}
$$
Then when we take expectation of $\log w_K$ with respect to $\pi_{\beta_0}, \cdots, \pi_{\beta_{K-1}}$, we will have:
$$
\begin{aligned}
\mathbb{E}_{\mathbf{z}_{0}, \cdots, \mathbf{z}_{K-1}}\left[\log w_K\right] &= \log Z_0 + \sum_{k=1}^{K}\mathbb{E}_{\pi_{\beta_{k-1}(\mathbf{z}_{k-1})}}\left[\log \frac{\tilde\pi_{\beta_k}(\mathbf{z}_{k-1})}{\tilde\pi_{\beta_{k-1}}(\mathbf{z}_{k-1})}\right] \\
&= \log Z_0 + \sum_{k=1}^K \left( \log \frac{\pi_{\beta_k}(\mathbf{z}_{k-1})}{\pi_{\beta_{k-1}}(\mathbf{z}_{k-1})} + \log Z_{k} - \log Z_{k-1} \right)\\
&=  \log Z_K - \sum_{k=1}^K D_{KL}(\pi_{\beta_{k-1}}(\mathbf{z})\|\pi_{\beta_{k}}(\mathbf{z}))\\
&\leq \log Z_K
\end{aligned}\label{eq.AISgap}
$$
The above derivation gives the gap between the AIS-based estimation of the log partition and the true log partition function, equals to the sum of KL-divergence between the intermediate distributions.

### AIS and TVO, equivalence and difference

When we consider a geometric averaging path, $\pi_{\beta_k}(\mathbf{z}) = \pi_{0}(\mathbf{z})^{1-\beta_k}\pi_1(\mathbf{z})^{\beta_k}$, then the $\log \tilde\pi_{\beta_k}\left(\mathbf{z}_{k-1}\right)-\log \tilde\pi_{\beta_{k-1}}\left(\mathbf{z}_{k-1}\right)$ term in $\eqref{eq.AIS}$, simplifies to $\left(\beta_{k}  -\beta_{k-1}\right) \log\frac{\tilde\pi_1(\mathbf{z}_{k-1})}{\tilde \pi_0(\mathbf{z}_{k-1})}$, thus the expected log weight becomes to:
$$
\begin{aligned}
\mathbb{E}_{\mathbf{z}_{0}, \cdots, \mathbf{z}_{K-1}}\left[\log w_K\right] &= \log Z_0 + \sum_{k=1}^K \left(\beta_{k}  -\beta_{k-1}\right) \mathbb{E}_{\pi_{\beta_{k-1}}(\mathbf{z}_{k-1})}\left[ \log\frac{\tilde\pi_1(\mathbf{z}_{k-1})}{\tilde \pi_0(\mathbf{z}_{k-1})} \right]\\
&=  \sum_{k=1}^K \left(\beta_{k}  -\beta_{k-1}\right) \mathbb{E}_{\pi_{\beta_{k-1}}(\mathbf{z}_{k-1})}\left[ \log\frac{\tilde\pi_1(\mathbf{z}_{k-1})}{\tilde \pi_0(\mathbf{z}_{k-1})} \right],
\end{aligned}\label{eq.AISasTVO}
$$
which reproduces the TVO objective.

Some tiny difference is how to get the samples from $\pi_{\beta_{k-1}}$ to practice Monte Carlo estimation of the expectations. AIS uses MCMC based transition kernel to get $\mathbf{z}_{k-1} \sim \pi_{\beta_{k-1}}$, while TVO uses importance sampling based techniques to estimate the objective value [^TVO][^all in exp]. These differences in sampling technique will result to performance difference in estimation accuracy, gradient variance and computational efficiency.

### From left to right Riemann sum

Note that the TVO lower bound results from the left Riemann sum of the TI integration, and we can construct an upper bound of the log evidence by replacing the left Riemann sum with right Riemann sum:
$$
\operatorname{TVO}_{\text{upper}} = \sum_{k=1}^K \left(\beta_{k}  -\beta_{k-1}\right) \mathbb{E}_{\pi_{\beta_{k}}(\mathbf{z}_{k})}\left[ \log\frac{\tilde\pi_1(\mathbf{z}_{k})}{\tilde \pi_0(\mathbf{z}_{k})} \right].
$$
The corresponding modified AIS algorithm of the $\operatorname{TVO}_{\text{upper}}$ bound can be obtained by interchanging the order of moving forward with the MCMC transition operator and the update of the current log weight:

<img src="what is TVO.assets/4167BBEC601A5822DDC9EC4726B29712.png" alt="img" style="zoom:25%;" />

and similar to $\eqref{eq.AISgap}$, in this way the expected log weight becomes to:
$$
\begin{aligned}
\mathbb{E}_{\mathbf{z}_{1}, \cdots, \mathbf{z}_{K}}\left[\log w_K\right] 
&= \log Z_0 + \mathbb{E}_{\mathbf{z}_{1}, \cdots, \mathbf{z}_{K}}\left[\sum_{k=1}^K\left(\log \tilde\pi_{\beta_k}(\mathbf{z}_k) - \log \tilde\pi_{\beta_{k-1}}(\mathbf{z}_k)\right)\right] \\
&= \log Z_0 + \sum_{k=1}^K\mathbb{E}_{ \pi_{\beta_k}(\mathbf{z}_{k})}\left[\log\frac{\pi_{\beta_k}(\mathbf{z}_k)}{\pi_{\beta_{k-1}}(\mathbf{z}_k)} + \log Z_{k} - \log Z_{k-1}\right] \\
&= \log Z_K + \sum_{k=1}^K D_{KL}(\pi_{\beta_{k}}(\mathbf{z})\|\pi_{\beta_{k-1}}(\mathbf{z})) \\
& \geq \log Z_K,
\end{aligned}
$$
and this happened to be the an upper bound of the log evidence, and the upper bound gap is again the sum of KL divergences between the intermediate distributions with the order reversed.

The difference between the TVO lower and upper bound is shown in the following figure:

<img src="what is TVO.assets/EB5BA3482A77C1AE51F4B3C14641B1AD.png" alt="img" style="zoom:20%;" />

### Flatten the TI curve

In classical VI, the optimization goal is actually to increase the value of the start point (which is the ELBO) to make it as high as possible. Note that this is equivalent to minimizing the red area in the following figure:

<img src="what is TVO.assets/BCF11A2BF426B23829E32BCFEA9DA354.png" alt="img" style="zoom:30%;" />



However, for some cases just minimizing the red area or equivalently, $D_{KL}\left[q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}|\mathbf{x})\right]$, can have problems. An example is shown in the following figure:

<img src="what is TVO.assets/37504A6320CC81165F1D44A398BA917F.png" alt="img" style="zoom:30%;" />



The TI curve is still non-decreasing as justified before, but the $D_{KL}\left[q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}|\mathbf{x})\right]$ is much smaller than $D_{KL}\left[p(\mathbf{z}|\mathbf{x})\| q_\phi(\mathbf{z}|\mathbf{x})\right]$. This example inspires us to replace the KL divergence by the Jeffrey divergence (a.k.a. the symmetric KL divergence) to measure the difference between the variational distribution and the true posterior. The Jeffrey divergence can be visualized as the sum of the yellow and red area in the above figure, and it also can be represented by the red line.

Note that minimizing the Jeffrey divergence between $q_\phi(\mathbf{z}|\mathbf{x})$ and $p(\mathbf{z}|\mathbf{x})$ is equivalent to flattening the TI curve, and only when the variational distribution $q_\phi(\mathbf{z}|\mathbf{x})$ equals to the true posterior $p(\mathbf{z}|\mathbf{x}),$ the curve will be a flat line. 

### Jeffrey divergence and Fisher information

To measure the curvature of the TI curve, for every point on the curve, we can sum (integrate) their derivatives with respect to $\beta$. Note that every point's coordinate can be written as $(\beta, \frac{\partial}{\partial \beta} \log Z_\beta= \mathbb{E}_{\pi_{\beta}}\left[\log\frac{p(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right] )$ , and the derivatives with respect to $\beta$ is the second derivative of $\log Z_\beta$ with respect to $\beta$: $\frac{\partial^2}{\partial \beta^2} \log Z_\beta$. Of course we can directly differentiate $\mathbb{E}_{\pi_{\beta}}\left[\log\frac{p(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right]$, but we can again utilize the exponential family property to make our life easier. 

Note by $\eqref{eq.hessianlogZ}$ we already have $\frac{\partial}{\partial \beta} \frac{\partial}{\partial \beta} \log Z_\beta = \operatorname{Var}_{\pi_\beta}\left(\log \frac{p(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right)$, thus we can rewrite the Jeffrey divergence as a one-dimensional integration: 
$$
D_{KL}\left[q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}|\mathbf{x})\right] + D_{KL}\left[p(\mathbf{z}|\mathbf{x}) \| q_\phi(\mathbf{z}|\mathbf{x}) \right] = \int_{\beta = 0}^{\beta=1} \operatorname{Var}_{\pi_\beta}\left(\log \frac{p(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right) d\beta.
$$
One interesting thing is the integrand is actually the Fisher information of $\beta$ with respect to the intermediate distributions as the following claim:

> **Claim: $\operatorname{Var}_{\pi_\beta}\left(\frac{\partial}{\partial \beta}\log\tilde \pi_{\beta}\right) = \operatorname{Var}_{\pi_\beta}\left(\frac{\partial}{\partial \beta}\log \pi_\beta\right)$, which is the Fisher information of $\beta$ for $\pi_\beta(\mathbf{z})$**.
>
> Proof:
> $$
> \begin{aligned}
> \operatorname{Var}_{\pi_\beta}\left(\frac{\partial}{\partial \beta}\log \pi_{\beta}\right) 
> &= \operatorname{Var}_{\pi_\beta}\left(\frac{\partial}{\partial \beta}\log\tilde \pi_{\beta}-\frac{\partial}{\partial \beta}\log Z_\beta\right)\\
> &= \operatorname{Var}_{\pi_\beta}\left(\frac{\partial}{\partial \beta}\log\tilde \pi_{\beta} -\mathbb{E}_{\pi_\beta}\left[\frac{\partial}{\partial\beta} \log \tilde \pi_\beta\right]\right) \\
> &= \mathbb{E}_{\pi_\beta}\left[\left(\frac{\partial}{\partial\beta}\log \tilde\pi_\beta - \mathbb{E}_{\pi_\beta}\left[\frac{\partial}{\partial\beta}\log \tilde\pi_\beta\right] - \mathbb{E}_{\pi_\beta}\left[ \frac{\partial}{\partial\beta}\log \tilde\pi_\beta - \mathbb{E}_{\pi_\beta}\left[\frac{\partial}{\partial\beta}\log \tilde\pi_\beta\right] \right] \right)^2\right] \\
> &= \mathbb{E}_{\pi_\beta}\left[\left( \frac{\partial}{\partial \beta}\log\tilde \pi_{\beta} -\mathbb{E}_{\pi_\beta}\left[\frac{\partial}{\partial\beta} \log \tilde \pi_\beta\right] \right)^2\right] \\
> &= \operatorname{Var}_{\pi_\beta}\left(\frac{\partial}{\partial \beta}\log\tilde \pi_{\beta}\right).
> \end{aligned}
> $$
> 

Thus in this way, we have drawn the connection between the Fisher information of the annealing parameter $\beta$ and the Jeffrey divergence that Jeffrey divergence equals to the integral of the Fisher information along the geometric path. 

The result gives us another interpretation of TVO that the closer $q_\phi(\mathbf{z}|\mathbf{x})$ and $p(\mathbf{z}|\mathbf{x})$ are, the less Fisher information should be contained in $\beta$. And this is natural because the Fisher information denotes how certain about the estimation about $\beta$ we are, and the closer the two distributions are, the less certain we are since it becomes much harder to distinguish.

One thing should be noted is that the ELBO maximizing goal and the TVO curve flattening goal is coupled, in other words we can always achieve one goal by achieving the other in most cases.



## Connections with other VI methods

### VI with Variational Contrastive Divergence

In the above, we have discussed about measuring the difference between the variational distribution and the true posterior with Jeffrey divergence, but we haven't mentioned how to compute this divergence since the forward KL divergence $D_{KL}\left[p(\mathbf{z}|\mathbf{x}) \| q_{\phi}(\mathbf{z}|\mathbf{x})\right]$ is an expectation over the true posterior, which is individually hard to estimate. 

The Variational Contrastive Divergence [^VCD] (VCD) is a divergence originally proposed to combine MCMC methods with VI methods. This [blog](https://www.inference.vc/icml-highlight-contrastive-divergence-for-variational-inference-and-mcmc/) gives an excellent introduction to it. Denote the MCMC transition kernel $\Pi$ as an improvement operator, the main idea is: given any distribution $q$, taking an MCMC steps should take you closer to the posterior, and the VCD is given as:
$$
\mathcal{L}_{VCD} = \mathbb{E}_{\Pi^tq_\phi}\left[\log\frac{p(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right] - \mathbb{E}_{q_\phi}\left[\log\frac{p(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right],
$$
which satisfies all the constraints that a divergence should meet. Both terms resemble the *local evidence* in TVO. The difference is the averaging parameter is now $t \in \{0, 1, \cdots, \infty\}$ rather than $\beta \in (0, 1)$. Note that the larger $t$ is, the closer $\Pi^tq_\theta$ can get to $p(\mathbf{z}|\mathbf{x})$, and when $t \to \infty$, $\Pi^tq_\theta (\mathbf{z}) =  p(\mathbf{z}|\mathbf{x})$, this is similar to increasing $\beta$ to 1. The VCD is shown in the following figure and when $t\to\infty$, it recovers the Jeffrey divergence.

<img src="what is TVO.assets/F1FCF3A956DFFD8407FC40DED479F7DD.png" alt="img" style="zoom:30%;" />

### MCMC-VI

In the above, the family of the variational distribution $q_\phi$ is not discussed. It can be traditional factorizable Gaussian, or normalizing flows. However, the approximation quality depends on both the objective and the expressiveness of the variational distribution family. MCMC-VI [^MCMC-VI], and similar methods [^auxiliaryDGM] define the variational distribution as the marginal distribution of a Markov chain or a mixture distribution:
$$
q_\phi(\mathbf{z}) = \int q_0(\mathbf{z}_0)\prod_{t=1}^TT_\phi(\mathbf{z}_t|\mathbf{z}_{t-1})d\mathbf{z}_{0}\cdots\mathbf{z}_{T-1}\label{eq.AISq}
$$
In [^MCMC-VI], it is shown that when we define the variational distribution $q$ as the marginal distribution of the corresponding AIS algorithm, we can get a looser bound than $\mathbb{E}_{q_\phi}\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z})}\right]$ (note although this is an ELBO, it is intractable since $q_\phi(\mathbf{z})$ is now an implicit distribution that we can only sample from but cannot evaluate the density for a certain point, this is a tradeoff between the expressiveness of the variational distribution and the tightness of the lower bound), and the new bound recovers $\eqref{eq.AISasTVO}$. In other words, regardless of the way we conduct Monte Carlo estimation of the variational objective, using the TVO to optimize an explicit variational distribution $q_\phi(\mathbf{z})$ is almost equivalent to using the MCMC-VI (also the method proposed in [^auxiliaryDGM]) objective with AIS to optimize the implicit variational distribution in $\eqref{eq.AISq}$. This shows that in variational inference, the objective function and the variational family can be coupled.



## references

[^tighter not better]: Rainforth, Tom, et al. "Tighter variational bounds are not necessarily better." *International Conference on Machine Learning*. PMLR, 2018.
[^TVO]:  Masrani, Vaden, Tuan Anh Le, and Frank Wood. "The thermodynamic variational objective." *Advances in Neural Information Processing Systems* 32 (2019).
[^all in exp]: Brekelmans, Rob, et al. "All in the exponential family: Bregman duality in thermodynamic variational inference." *arXiv preprint arXiv:2007.00642* (2020).
[^AIS]: Neal, Radford M. "Annealed importance sampling." *Statistics and computing* 11.2 (2001): 125-139.
[^moment averaging path]: Grosse, Roger B., Chris J. Maddison, and Russ R. Salakhutdinov. "Annealing between distributions by averaging moments." *Advances in Neural Information Processing Systems* 26 (2013).
[^VCD]: Ruiz, Francisco, and Michalis Titsias. "A contrastive divergence for combining variational inference and mcmc." *International Conference on Machine Learning*. PMLR, 2019.
[^MCMC-VI]: Salimans, Tim, Diederik Kingma, and Max Welling. "Markov chain monte carlo and variational inference: Bridging the gap." *International conference on machine learning*. PMLR, 2015.
[^auxiliaryDGM]: Maaløe, Lars, et al. "Auxiliary deep generative models." *International conference on machine learning*. PMLR, 2016.
