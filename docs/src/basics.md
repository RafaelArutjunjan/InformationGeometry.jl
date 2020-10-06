

### Introduction to Information Geometry

For a more detailed discussion of information geometry, see e.g. [my Master's Thesis](https://github.com/RafaelArutjunjan/Master-Thesis), [this paper](https://arxiv.org/abs/1010.1449) or [this book](https://books.google.de/books?id=vc2FWSo7wLUC).

Essentially, information geometry is a combination of the mathematical disciplines of differential geometry and probability theory. The main idea is to rephrase statistical problems in such a way that they can be given a geometric interpretation.


### Information Divergences

In information theory, the dissimilarity between two probability distributions ``p(x)`` and ``q(x)`` is generally quantified using so-called information divergences, which are positive-definite functionals. The most popular choice of information divergence is given by the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) ``D_{\text{KL}}[p,q]`` defined by
```math
D_{\text{KL}}[p,q] \coloneqq \int \mathrm{d}^m y \, p(y) \, \mathrm{ln} \bigg( \frac{p(y)}{q(y)} \bigg) = \mathbb{E}_p \bigg[\mathrm{ln}\bigg( \frac{p}{q} \bigg) \bigg].
```
Intuitively, the Kullback-Leibler divergence corresponds to the relative increase in Shannon entropy (i.e. loss of information) that is incurred by approximating the distribution ``p(x)`` through ``q(x)``.
In addition to its tangible information-theoretic interpretation, the Kullback-Leibler divergence has the following desirable properties:
* reparametrization invariance with respect to the random variable over which the distributions are integrated,
* applicable between any two probability distributions with common support, e.g. a ``\chi^2``-distribution and a Poisson distribution or a normal and a Cauchy distribution.

On the other hand, the disadvantages of using information divergences such as the Kullback-Leibler divergence to measure the dissimilarity of distributions are:
* they are typically not symmetric, i.e. ``D_{\text{KL}}[p,q] \neq D_{\text{KL}}[q,p]``
* they usually do not satisfy a triangle inequality
wherefore they do not constitute distance functions (i.e. metric functions) on the underlying space of probability distributions.


### The Fisher Metric

In practical applications, one is often particularly interested in spaces of probability distributions which form a single overarching family and can be parametrized using a parameter configuration ``\theta \in \mathcal{M}`` where ``\mathcal{M}`` constitutes a smooth manifold. Accordingly, any two members ``p(y;\theta_1)`` and ``p(y;\theta_2)`` of this family can be compared using e.g. the Kullback-Leibler divergence ``D_{\text{KL}}\big[p(\theta_1),p(\theta_2)\big]`` via the formula given above.

While the Kullback-Leibler divergence ``D_{\text{KL}}[p,q]`` does not constitute a proper distance function on ``\mathcal{M}``, it can be expanded in Taylor series around ``\theta_\text{MLE}`` in terms of its derivatives. The zeroth order of this expansion vanishes due to the definiteness of the Kullback-Leibler divergence (i.e. ``D_{\text{KL}}[q,q] = 0`` for all distributions ``q``). Similarly, the first order vanishes since the expectation of the components of the score is nil. Thus, the second order approximation of the Kullback-Leibler divergence is completely determined by its Hessian, which can be computed as
```math
g_{ab}(\theta) \coloneqq \bigg[\frac{\partial^2}{\partial \psi^a \, \partial \psi^b} \, D_\text{KL} \big[p(y;\theta) , p(y;\psi) \big] \bigg]_{\psi = \theta}
= ... = -\mathbb{E}_p\bigg[\frac{\partial^2 \, \mathrm{ln}(p)}{\partial \theta^a \, \partial \theta^b}\bigg] = ... = \mathbb{E}_p\bigg[\frac{\partial \, \mathrm{ln}(p)}{\partial \theta^a} \frac{\partial \, \mathrm{ln}(p)}{\partial \theta^b}\bigg]
```
where it was assumed that the order of the derivative operator and the integration involved in the expectation value can be interchanged.

The Hessian of the Kullback-Leibler divergence is typically referred to as the [Fisher information matrix](https://en.wikipedia.org/wiki/Fisher_information). Moreover, since it can be shown that the Fisher information is not only positive-definite but also exhibits the transformation behaviour associated with a ``(0,2)``-tensor field, it can therefore be used as a [Riemannian metric](https://en.wikipedia.org/wiki/Fisher_information_metric) on the parameter manifold ``\mathcal{M}``.

Clearly, the Riemannian geometry induced on ``\mathcal{M}`` by the Fisher metric is ill-equipped to faithfully capture the behaviour of the Kullback-Leibler divergence in its entirety (e.g. its asymmetry). Nevertheless, this Riemannian approximation already encodes many of the key aspects of the Kullback-Leibler divergence and additionally benefits from the versatility and maturity of the differential-geometric formalism. Therefore, the Fisher metric offers a convenient and powerful tool which can be used to study statistical problems in a coordinate invariant setting which focuses on intrinsic properties of the parameter manifold.

**To be continued...**
