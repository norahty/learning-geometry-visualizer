# Trajectory Auditor: Measure-Based Visualization of Iterative Algorithm Dynamics

**Author:** Nora Han  
**Status:** Research tool — code available upon request  
**Last updated:** October 2025

---

## Overview

This tool visualizes how **probability outputs** from an iterative procedure evolve over time.  
Given checkpoints of any model or process that produces probability vectors on a fixed probe set, it embeds those distributions with **InPCA** (using a **Hellinger** kernel) and reveals the **trajectory** the procedure traces through the space of probability measures.

**What you can see**
- convergence vs. cycling (last-iterate vs. averaged behavior)  
- low-dimensional organization of states  
- phase changes across training/algorithmic conditions

> Works with *any* procedure that emits probability vectors (classifiers, probabilistic algorithms, repeated games).  
> No dependence on any specific architecture.

---

## Why this matters

The goal is **legibility**: when dynamics exhibit regularity (stability, low dimension, repeatable returns), explanations stop being post-hoc stories and start looking like **claims you can justify**.  
This tool provides quantitative, visual evidence for such regularities.

---

## Method (one page)

Let $\(p_t(x)\)$ be the predictive distribution at step $\(t\)$ on a fixed probe set $\(X\)$.  
Define the Hellinger inner product  
$\[
\langle p_i, p_j \rangle_H \;=\; \sum_{x\in X} \sqrt{p_i(x)\,p_j(x)} .
\]$

1. Build the Hellinger similarity matrix \(K\) over checkpoints.  
2. Double-center $\(K\)$ to obtain a Gram matrix.  
3. Eigen-decompose to get **InPCA** coordinates (top components).  
4. Plot trajectories and diagnostics (2D/3D) across time or conditions.

**Notes.**  
- The **measure space** is the underlying probability space $\( (X,\mathcal F,\mu) \)$.  
- “Hellinger geometry” here means distances/angles **on the set of probability measures**, not the base space $\(X\)$.

---

## Quick Start

1. Collect probability vectors at chosen checkpoints on a fixed probe set.  
2. Run `inpca_embed.py` to compute InPCA coordinates (Hellinger + SVD).  
3. Use `plot_trajectory.py` for interactive 2D/3D visualization (Plotly).  

*Environment:* Python ≥3.10, NumPy, SciPy, scikit-learn, Plotly, Pandas.

---

## Examples

### 1) Image classification (MNIST)  
Multiple models trained on different subsets; each curve is the evolution of output probabilities projected into InPCA space.

<img width="1440" height="767" alt="Screenshot 2025-10-21 at 1 07 56 PM" src="https://github.com/user-attachments/assets/1b87e2f3-30d9-4ef6-9505-0c784536a0fa" />



### 2) Toy synthetic data  
Two-class setting (N=4) for controlled tests of convergence and reversibility; distinct endpoints illustrate divergent paths and stabilization.

<img width="1430" height="703" alt="image" src="https://github.com/user-attachments/assets/1ba5a620-1509-49d9-96b2-fc1ec6299aa6" />

### 3) Repeated-game dynamics  
Probability updates from learning in games; comparison of instantaneous vs. time-averaged strategies in a 3D embedding.

<img width="1019" height="448" alt="image" src="https://github.com/user-attachments/assets/f583ac06-88e6-4c54-b321-7d4bbc37bffc" />
<img width="1019" height="448" alt="Screenshot 2025-10-17 at 2 53 07 PM" src="https://github.com/user-attachments/assets/03f809a1-c578-46e7-9737-2907f9f99963" />


---

## 🧩 Reference

K.N. Quinn, C.B. Clement, F. De Bernardis, M.D. Niemack, & J.P. Sethna, Visualizing probabilistic models and data with Intensive Principal Component Analysis, Proc. Natl. Acad. Sci. U.S.A. 116 (28) 13762-13767, https://doi.org/10.1073/pnas.1817218116 (2019).

Also see my implementation of this paper's partial results at: [https://github.com/norahty/inpca-mnist](https://github.com/norahty/inpca-mnist)
