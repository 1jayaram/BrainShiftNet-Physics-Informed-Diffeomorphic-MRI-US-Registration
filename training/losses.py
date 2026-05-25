"""
training/losses.py

Loss functions for brain shift / image registration training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─── IMAGE SIMILARITY LOSSES ───────────────────────────────────────────────

class NCC(nn.Module):
    """
    Local Normalised Cross-Correlation.
    Best for mono-modal (MRI-MRI) or carefully tuned MRI-US registration.
    window_size: local patch size (default 9 means 9^3 voxels)
    """
    def __init__(self, window_size: int = 9):
        super().__init__()
        self.win = window_size

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        I = y_true
        J = y_pred
        ndims = len(I.shape) - 2
        assert ndims == 3, "Only 3D supported"

        win = [self.win] * ndims
        sum_filt = torch.ones([1, 1, *win], device=I.device, dtype=I.dtype)
        pad_no = self.win // 2
        stride, padding = [1] * ndims, [pad_no] * ndims

        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum  = F.conv3d(I,  sum_filt, stride=stride, padding=padding)
        J_sum  = F.conv3d(J,  sum_filt, stride=stride, padding=padding)
        I2_sum = F.conv3d(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = F.conv3d(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = F.conv3d(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)
        return -torch.mean(cc)   # negative because we minimise


class MutualInformation(nn.Module):
    """
    Differentiable Mutual Information via soft histograms.
    Better than NCC for multi-modal (MRI-US) registration.
    """
    def __init__(self, bins: int = 64, sigma: float = 0.5, normalised: bool = True):
        super().__init__()
        self.bins       = bins
        self.sigma      = 2 * sigma ** 2
        self.normalised = normalised

    def _parzen_window(self, x: torch.Tensor, bin_centres: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((x.unsqueeze(-1) - bin_centres) ** 2) / self.sigma)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        bin_centres = torch.linspace(0, 1, self.bins, device=y_true.device)

        # Flatten spatial dims
        I = y_true.view(y_true.shape[0], -1)
        J = y_pred.view(y_pred.shape[0], -1)

        # Soft histograms
        p_I  = self._parzen_window(I, bin_centres).mean(1)   # (B, bins)
        p_J  = self._parzen_window(J, bin_centres).mean(1)
        p_IJ = torch.bmm(
            self._parzen_window(I, bin_centres).permute(0, 2, 1),
            self._parzen_window(J, bin_centres)
        ) / I.shape[1]  # (B, bins, bins)

        # Normalise
        p_I  = p_I  / (p_I.sum(dim=1, keepdim=True)  + 1e-9)
        p_J  = p_J  / (p_J.sum(dim=1, keepdim=True)  + 1e-9)
        p_IJ = p_IJ / (p_IJ.sum(dim=[1, 2], keepdim=True) + 1e-9)

        H_I  = -(p_I  * (p_I  + 1e-9).log()).sum(1)
        H_J  = -(p_J  * (p_J  + 1e-9).log()).sum(1)
        H_IJ = -(p_IJ * (p_IJ + 1e-9).log()).sum([1, 2])

        mi = H_I + H_J - H_IJ
        if self.normalised:
            mi = 2 * mi / (H_I + H_J + 1e-9)   # NMI in [0, 1]
        return -mi.mean()  # negative = minimise


# ─── REGULARISATION LOSSES ─────────────────────────────────────────────────

class BendingEnergy(nn.Module):
    """
    Bending energy regularisation on the deformation field.
    Penalises second-order derivatives (ensures smooth deformation).
    """
    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """flow: (B, 3, D, H, W)"""
        dy = flow[:, :, 1:, :, :]  - flow[:, :, :-1, :, :]
        dx = flow[:, :, :, 1:, :]  - flow[:, :, :, :-1, :]
        dz = flow[:, :, :, :, 1:]  - flow[:, :, :, :, :-1]

        d2y = dy[:, :, 1:, :, :]  - dy[:, :, :-1, :, :]
        d2x = dx[:, :, :, 1:, :]  - dx[:, :, :, :-1, :]
        d2z = dz[:, :, :, :, 1:]  - dz[:, :, :, :, :-1]

        return (d2y.pow(2).mean() + d2x.pow(2).mean() + d2z.pow(2).mean()) / 3


class GradientLoss(nn.Module):
    """
    Gradient (first-order) regularisation.
    Penalises large spatial gradients in the deformation field.
    """
    def __init__(self, penalty: str = "l2"):
        super().__init__()
        self.penalty = penalty

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        dy = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
        dx = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
        dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])
        if self.penalty == "l2":
            dy = dy.pow(2)
            dx = dx.pow(2)
            dz = dz.pow(2)
        return (dy.mean() + dx.mean() + dz.mean()) / 3


class JacobianLoss(nn.Module):
    """
    Penalises negative Jacobian determinants (folding).
    Ensures the deformation field is locally invertible.
    """
    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        J = self._jacobian_determinant(flow)
        neg_penalty = torch.clamp(-J, min=0)
        return neg_penalty.mean()

    def _jacobian_determinant(self, flow: torch.Tensor) -> torch.Tensor:
        """Approximate Jacobian determinant via finite differences."""
        B, _, D, H, W = flow.shape
        dy = flow[:, :, 1:, :, :]  - flow[:, :, :-1, :, :]
        dx = flow[:, :, :, 1:, :]  - flow[:, :, :, :-1, :]
        dz = flow[:, :, :, :, 1:]  - flow[:, :, :, :, :-1]
        # Central crop to match sizes
        s = (slice(None), slice(None), slice(0, D-1), slice(0, H-1), slice(0, W-1))
        J00 = 1 + dy[s][:, 0]
        J11 = 1 + dx[s][:, 1]
        J22 = 1 + dz[s][:, 2]
        J01 = dy[s][:, 1]
        J02 = dy[s][:, 2]
        J10 = dx[s][:, 0]
        J12 = dx[s][:, 2]
        J20 = dz[s][:, 0]
        J21 = dz[s][:, 1]
        det = (J00 * (J11*J22 - J12*J21)
             - J01 * (J10*J22 - J12*J20)
             + J02 * (J10*J21 - J11*J20))
        return det


# ─── LANDMARK SUPERVISION LOSS ─────────────────────────────────────────────

class LandmarkLoss(nn.Module):
    """
    Supervised landmark loss.
    Penalises the distance between:
      - predicted landmark positions (MRI landmarks warped by flow)
      - ground truth landmark positions in US space
    
    landmarks: (B, N, 6) tensor [x1,y1,z1, x2,y2,z2] in voxel coords
      where (x1,y1,z1) = MRI coords, (x2,y2,z2) = US coords
    """
    def forward(self, flow: torch.Tensor,
                landmarks: torch.Tensor,
                img_size: tuple) -> torch.Tensor:
        if landmarks is None or landmarks.shape[1] == 0:
            return torch.tensor(0.0, device=flow.device)

        B, N, _ = landmarks.shape
        device  = flow.device

        # Source landmarks (MRI space)
        src_lm = landmarks[:, :, :3]   # (B, N, 3)
        tgt_lm = landmarks[:, :, 3:]   # (B, N, 3)

        # Sample flow at source landmark positions
        # Normalise to [-1, 1] for grid_sample
        norm_lm = src_lm.clone()
        for i, s in enumerate(img_size):
            norm_lm[:, :, i] = 2.0 * src_lm[:, :, i] / (s - 1) - 1.0

        # grid_sample expects (B, C, D, H, W) and grid (B, 1, 1, N, 3)
        grid = norm_lm[:, :, [2, 1, 0]].unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N, 3)
        sampled_flow = F.grid_sample(flow, grid, align_corners=True, mode="bilinear")
        sampled_flow = sampled_flow.squeeze(2).squeeze(2)           # (B, 3, N)
        sampled_flow = sampled_flow.permute(0, 2, 1)                # (B, N, 3)

        # Warped landmark positions
        warped_lm = src_lm + sampled_flow

        # TRE = Euclidean distance
        tre = torch.sqrt(((warped_lm - tgt_lm) ** 2).sum(dim=-1) + 1e-8)
        return tre.mean()


# ─── COMBINED REGISTRATION LOSS ────────────────────────────────────────────

class RegistrationLoss(nn.Module):
    """
    Combined loss for MRI-to-US deformable registration.
    
    L = lambda_sim  * L_similarity
      + lambda_reg  * L_regularisation
      + lambda_lm   * L_landmark
      + lambda_jac  * L_jacobian
    """
    def __init__(
        self,
        lambda_sim:  float = 1.0,
        lambda_reg:  float = 2.0,
        lambda_lm:   float = 5.0,
        lambda_jac:  float = 0.5,
        similarity:  str   = "mi",      # "ncc" | "mi" | "mse"
        regulariser: str   = "bending", # "bending" | "gradient"
        img_size:    tuple = (160, 192, 160),
        ncc_window:  int   = 9,
        mi_bins:     int   = 64,
    ):
        super().__init__()
        self.lambda_sim  = lambda_sim
        self.lambda_reg  = lambda_reg
        self.lambda_lm   = lambda_lm
        self.lambda_jac  = lambda_jac
        self.img_size    = img_size

        # Similarity
        if similarity == "ncc":
            self.sim_loss = NCC(window_size=ncc_window)
        elif similarity == "mi":
            self.sim_loss = MutualInformation(bins=mi_bins)
        else:
            self.sim_loss = nn.MSELoss()

        # Regularisation
        if regulariser == "bending":
            self.reg_loss = BendingEnergy()
        else:
            self.reg_loss = GradientLoss()

        self.jac_loss = JacobianLoss()
        self.lm_loss  = LandmarkLoss()

    def forward(
        self,
        mri_warped:  torch.Tensor,
        us_fixed:    torch.Tensor,
        flow:        torch.Tensor,
        landmarks:   torch.Tensor = None,
    ) -> dict:
        L_sim  = self.sim_loss(us_fixed, mri_warped[:, :1])  # compare first channel
        L_reg  = self.reg_loss(flow)
        L_jac  = self.jac_loss(flow)
        L_lm   = self.lm_loss(flow, landmarks, self.img_size) if landmarks is not None else torch.tensor(0.0)

        total = (self.lambda_sim * L_sim
               + self.lambda_reg * L_reg
               + self.lambda_jac * L_jac
               + self.lambda_lm  * L_lm)

        return {
            "total":    total,
            "sim":      L_sim.item(),
            "reg":      L_reg.item(),
            "jac":      L_jac.item(),
            "landmark": L_lm.item() if isinstance(L_lm, torch.Tensor) else 0.0,
        }
