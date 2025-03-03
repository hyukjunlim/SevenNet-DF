
<img src="SevenNet_logo.png" alt="Alt text" height="180">

# SevenNet_dFS

**SevenNet_dFS** is an extended and optimized version of [**SevenNet**](https://github.com/MDIL-SNU/SevenNet), a machine-learning-based force field model tailored for molecular dynamics (MD) simulations.

## Key Differences

```diff
+ Direct force and stress prediction output
+ Integrated derivative values during training for higher accuracy
+ Up to 250x faster MD simulation speed
- non-conservative MD simulation
```

## Model Overview

```python
# SevenNet_dFS directly predicts force and stress
forces, stresses = sevennet_dfs.predict(atomic_coordinates)
```

Previous implementations computed derivatives internally, causing computational bottlenecks:

```python
# Original SevenNet (slower inference due to derivative calculations)
derivatives = sevennet.compute_derivatives(atomic_coordinates)
forces, stresses = post_process(derivatives)
```

## Training Approach

To maintain high accuracy despite direct prediction, derivative values from the original SevenNet were integrated during training:

```python
# Integrated training loop example
loss_force = criterion(direct_forces, derivative_forces)
loss_stress = criterion(direct_stresses, derivative_stresses)
total_loss = loss_force + loss_stress
total_loss.backward()
```

## Performance

```markdown
| Metric                      | Improvement         |
|-----------------------------|---------------------|
| MD Simulation Speed         | ✅ 250x Faster      |
| Prediction Accuracy         | ✅ Enhanced         |
| Inference Computational Cost| ✅ Significantly Reduced |
```

## Quick Start

```bash
# Clone Repository
git clone https://github.com/your-repo/SevenNet_dFS.git
cd SevenNet_dFS

# Install dependencies
pip install sevenn

# Run inference example
cd sevennet_tutorial
python tuto.py
```

## Citation<a name="citation"></a>

If you use this code, please cite our paper:
```txt
@article{park_scalable_2024,
	title = {Scalable Parallel Algorithm for Graph Neural Network Interatomic Potentials in Molecular Dynamics Simulations},
	volume = {20},
	doi = {10.1021/acs.jctc.4c00190},
	number = {11},
	journal = {J. Chem. Theory Comput.},
	author = {Park, Yutack and Kim, Jaesun and Hwang, Seungwoo and Han, Seungwu},
	year = {2024},
	pages = {4857--4868},
}
```