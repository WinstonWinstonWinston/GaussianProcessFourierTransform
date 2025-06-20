# Gaussian Process Fourier Transform (GPFT)

We present a nonparametric Bayesian framework to infer radial distribution functions from experimental scattering measurements with uncertainty quantification using **non-stationary Gaussian processes**. The Gaussian process prior mean and kernel functions are designed to resolve well-known numerical problems with the Fourier transform, including discrete measurement binning and detector windowing, while encoding fundamental yet minimal physical knowledge of liquid structure. We demonstrate uncertainty propagation of the Gaussian process posterior to unmeasured quantities of interest. The methodology is applied to **liquid argon** and **water** as a proof of principle.

---

![Overview Diagram](GPFT.drawio_fig.png)

---

## Directory Structure

The repository is organized by **material type** (e.g., `Argon/`, `H2O/`), with subfolders further separated into:

- `experimental/`: Experimental dataset analyis
- `simulated/`: Simulated dataset analysis

## Core Code

The primary implementation of the Gaussian process Fourier transform framework is contained in:
`gptransform.py`
