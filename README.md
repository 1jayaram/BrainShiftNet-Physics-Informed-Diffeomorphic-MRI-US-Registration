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

## 💾 Dataset Requirements: The RESECT Database

Due to patient privacy and data usage agreements, the training data cannot be hosted in this repository. To train or evaluate BrainShiftNet, you **must download the RESECT database manually**.

**How to get the data:**
1. Create a free account on the Neuroimaging Informatics Tools and Resources Clearinghouse (NITRC).
2. Navigate to the **[RESECT Project Page](https://www.nitrc.org/projects/resect/)**.
3. Accept the data usage agreement and download the dataset (22 clinical cases containing pre-operative MRI, intra-operative US, and landmark `.tag` files).
4. Extract the dataset into the `data/raw/` directory of this repository.
5. Run the provided pre-processing script to normalize the ultrasound data:
   ```bash
   python training/fix1_diagnose_us.py --mri data/raw/mri.nii.gz --us data/raw/us.nii.gz --out data/processed/

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
