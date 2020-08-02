================================================
Massively Parallel Empirical Bayes Poisson Means
================================================

This package provides GPU-accelerated inference for the *Empirical Bayes
Poisson Means* (EBPM) problem:

.. math::

\\begin{align}
x_{ij} \\mid s_{i}, \\lambda_{ij} &\\sim \\operatorname{Poisson}(s_i \\lambda_{ij})\\\\
\\lambda_{ij} &\\sim g_j(\\lambda_{ij})
\\end{align}

This model can be used to explicitly separate and model variation in scRNA-seq
data due to measurement error and variation in true gene expression values
[Sarkar2020]_.

This implementation readily supports fitting the model for data on the order of
10⁶ cells and 10⁴ genes in parallel. It also supports fitting multiple EBPM
problems per gene in parallel, as arise when e.g., cells have been assigned to
groups (clusters). For example, we have `used the method
<https://aksarkar.github.io/singlecell-ideas/mpebpm.html>`_ to solve 537,678
EBPM problems (5,597 cells from 54 conditions, at 9,957 genes) in parallel in a
few minutes [Sarkar2019]_.

.. toctree::

   mpebpm
   references
