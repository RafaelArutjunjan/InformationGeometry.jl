

## Introduction to Information Geometry

For a more detailed discussion of information geometry, see e.g. my [Master's Thesis](https://github.com/RafaelArutjunjan/Master-Thesis) or [this paper](https://arxiv.org/abs/1010.1449) or alternatively [this book](https://books.google.de/books?id=vc2FWSo7wLUC).

### Information Divergences
Essentially, information geometry is a combination of the mathematical subjects of differential geometry and probability theory. The main idea is to rephrase statistical problems in such a way that they can be given a geometric interpretation.

In information theory, the dissimilarity between two probability distributions ``p(x)`` and ``q(x)`` is generally quantified using so-called information divergences, which are positive-definite functionals. The most popular choice of information divergence is given by the Kullback-Leibler divergence ``D_{\text{KL}}[p,q]`` defined by
```math
D_{\text{KL}}[p,q] \coloneqq \int \mathrm{d}^m y \, p(y) \, \mathrm{ln} \bigg( \frac{p(y)}{q(y)} \bigg) = \mathbb{E}_p \bigg[\mathrm{ln}\bigg( \frac{p}{q} \bigg) \bigg].
```
Intuitively, the Kullback-Leibler divergence corresponds to the relative increase in Shannon entropy (i.e. loss of information) incurred by approximating the distribution ``p(x)`` through ``q(x)``.
In addition to its tangible information-theoretic interpretation, the Kullback-Leibler divergence has the following desirable properties:
* reparametrization invariance with respect to the random variable over which the distributions are integrated,
* applicable for any two probability distributions with common support, e.g. a ``\chi^2``-distribution and a Poisson distribution or a normal and a Cauchy distribution.

On the other hand, the disadvantages of using the Kullback-Leibler divergence (or other information divergences) are:
* they are typically not symmetric, i.e. ``D_{\text{KL}}[p,q] \neq D_{\text{KL}}[q,p]``
* they do not satisfy a triangle inequality
wherefore they do not constitute distance functions (i.e. metric functions).


### The Fisher Metric

While the Kullback-Leibler divergence ``D_{\text{KL}}[p,q]`` does not constitute a proper distance function, its Hessian
```math
g_{ab}(\theta) \coloneqq \bigg[\frac{\partial^2}{\partial \psi^a \, \partial \psi^b} \, D_\text{KL} \big[p(y;\theta) , p(y;\psi) \big] \bigg]_{\psi = \theta}
= ... = -\mathbb{E}_p\bigg[\frac{\partial^2 \, \mathrm{ln}(p)}{\partial \theta^a \, \partial \theta^b}\bigg] = ... = \mathbb{E}_p\bigg[\frac{\partial \, \mathrm{ln}(p)}{\partial \theta^a} \frac{\partial \, \mathrm{ln}(p)}{\partial \theta^b}\bigg]
```
where it was assumed that the order of the derivative operator and the integral involved in the expectation value can be interchanged.

* Moreover, this is a second order Taylor expansion of DKL since score vanishes at MLE and no first order due to definiteness.
