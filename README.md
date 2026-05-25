# BrainShiftNet: Physics-Informed Diffeomorphic MRI-US Registration

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**BrainShiftNet** is a deep learning framework designed to predict and correct intraoperative brain shift in real-time. By performing diffeomorphic, multi-modal registration between pre-operative MRI (T1-CE & FLAIR) and intraoperative Ultrasound (iUS), this model provides neurosurgeons with continuously updated navigation coordinates.

> **Key Performance:** Achieves a mean Target Registration Error (TRE) of 14.28mm (down from >20mm) with an ultra-fast inference time of **0.32 seconds** on clinical test cases.

![Brain Shift Alignment Example](link_to_a_gif_or_image_showing_alignment.gif) *(Note: Add a visual here showing your before/after registration grids)*

## 🧠 Core Architecture
This project reframes deformable image registration as a physics-informed learning problem, mitigating the severe cross-modal intensity differences between MRI and US.

* **TransMorph & Attention U-Net:** Leverages Swin-Transformer and CNN encoder-decoder paradigms with attention-gated skip connections.
* **Diffeomorphic Integration:** Predicts a stationary velocity field (SVF) integrated via scaling-and-squaring, guaranteeing topology-preserving deformations (no unbiological tissue folding).
* **Physics-Informed Neural Network (PINN):** Supervised by a composite loss function integrating Normalised Mutual Information (NMI), Bending Energy, and Navier-Cauchy biomechanical constraints.

## ⚙️ Installation & Usage

### Prerequisites
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/1jayaram/BrainShiftNet-Physics-Informed-Diffeomorphic-MRI-US-Registration.git](https://github.com/1jayaram/BrainShiftNet-Physics-Informed-Diffeomorphic-MRI-US-Registration.git)
cd BrainShiftNet
pip install -r requirements.txt


* **GitHub Tags/Topics:** On your repository page, add topics like `medical-imaging`, `deep-learning`, `pytorch`, `image-registration`, `neurosurgery`, and `physics-informed-neural-networks`. This helps researchers find your repo via GitHub search.
* **Use GitHub Issues for "Help Wanted":** Create an issue titled "Feature Request: Implement Monte Carlo Dropout for Uncertainty Maps" and tag it with `help wanted` and `good first issue`. This gives visiting developers a concrete starting point if they want to contribute.
* **Share on Academic Networks:** Once the repo is live, share the link on platforms like ResearchGate, LinkedIn, and relevant subreddits (like r/MachineLearning or r/MedicalImaging) with a short video or GIF of your results.
