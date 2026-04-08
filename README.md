# Flow Field Reconstruction via Voronoi-Enhanced Physics-Informed Neural Networks with End-to-End Sensor Placement Optimization

VSOPINN is a differentiable framework for **high-fidelity field reconstruction from sparse measurements** and **adaptive sensor layout learning**. It integrates **soft Voronoi rasterization** of sparse sensor observations with a **geometry-adaptive convolutional PINN**, and further supports **multi-condition learning** through a shared-encoder multi-decoder design.

This repository contains the code and selected examples associated with our VSOPINN study on flow and thermal field reconstruction.

## Overview

Sparse sensor measurements are often incomplete, irregularly distributed, and sensitive to sensor failures. VSOPINN addresses these challenges by combining physics-informed learning, differentiable Voronoi-based data projection, and learnable sensor optimization in a unified end-to-end framework.

The main ideas of this repository include:

- **Voronoi-assisted data projection** for converting sparse point measurements into grid-based representations.
- **Geometry-adaptive field reconstruction** on irregular domains.
- **End-to-end sensor optimization** with differentiable learning and CVT-enhanced relocation.
- **Multi-condition layout transfer** for improving robustness and generalization across different operating conditions.
- **Robustness under sensor failures** through learned spatial representations.

## Representative results

### Lid-driven cavity flow

VSOPINN improves the reconstruction of the velocity field and yields lower local errors than baseline models in the benchmark lid-driven cavity case.

<p align="center">
  <img src="case1.jpg" width="900" alt="Lid-driven cavity flow results"/>
</p>

### Multi-condition lid-driven cavity flow

The learned layout generalizes to both interpolation and extrapolation regimes, providing more accurate reconstructions than the random layout under multiple Reynolds numbers.

<p align="center">
  <img src="case4_multi.jpg" width="900" alt="Multi-condition cavity flow results"/>
</p>

### Thermal field reconstruction in a casing domain

The framework also extends beyond Navier--Stokes flow reconstruction and can be applied to scalar thermal-field reconstruction in complex engineering geometries.

<p align="center">
  <img src="case5_T.jpg" width="900" alt="Thermal casing reconstruction results"/>
</p>

## Benchmarks covered in this work

The study validates VSOPINN through several representative numerical cases:

- Lid-driven cavity flow
- Vascular flow in irregular domains
- Rotational flow in annulus geometry
- Multi-condition lid-driven cavity flow
- Thermal field reconstruction in a casing domain

## What is included in this repository

Depending on the released version of the repository, the codebase may include:

- core VSOPINN model components
- geometry-adaptive reconstruction modules
- differentiable soft Voronoi projection modules
- sensor layout optimization routines
- case-specific scripts and example data
- visualization scripts for reconstructed fields and profiles

## Notes

- This repository is intended to accompany our VSOPINN study.
- The manuscript is currently under review, so this README focuses on the method and code rather than a formal citation block.
- If figure paths differ in your repository, simply update the image paths in this README accordingly.

## Contact

For questions regarding the code or the associated study, please contact the repository maintainers through GitHub.
