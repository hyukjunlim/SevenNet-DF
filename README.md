
<img src="SevenNet_logo.png" alt="Alt text" height="180">

# SevenNet_dFS

**SevenNet_dFS** is an extended and optimized version of [**SevenNet**](https://github.com/MDIL-SNU/SevenNet), a machine-learning-based force field model tailored for molecular dynamics (MD) simulations.

## Key Differences

```diff
+ Direct force and stress prediction output
+ Integrated derivative values during training for higher accuracy
+ Up to 5x faster inference speed
+ Up to 250x faster MD simulation speed
- Non-conservative MD simulation
- Slower training speed
```

## Performance

| Metric                      | Improvement         |
|-----------------------------|---------------------|
| Training Speed         | ✅ 1.59x Slower      |
| Inference Speed         | ✅ 4.91x Faster      |
| MD Simulation Speed         | ✅ 255.28x Faster      |
| Prediction Accuracy         | ✅ Enhanced         |
| Inference Computational Cost| ✅ Significantly Reduced |

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
```bibtex
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