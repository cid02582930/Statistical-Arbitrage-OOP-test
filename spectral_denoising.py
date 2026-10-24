"""
Module: spectral_denoising
Implements Marčenko-Pastur analytical fitting, spectral boundary extraction,
and constant trace shrinkage for empirical correlation matrices.
"""

from dataclasses import dataclass
import numpy as np
import scipy.optimize as opt


@dataclass(frozen=True)
class SpectralBounds:
    lambda_min: float
    lambda_max: float
    noise_variance: float


class MarchenkoPasturFilter:
    def __init__(self, q_factor: float) -> None:
        """
        Parameters
        ----------
        q_factor : float
            Ratio of sample observations to variables (T / N). Must be >= 1.0.
        """
        if q_factor < 1.0:
            raise ValueError(f"Q factor must be >= 1.0, provided: {q_factor: .2f}")
        self.q = q_factor

    def compute_theoretical_bounds(self, sigma_sq: float = 1.0) -> tuple[float, float]:
        inv_q_sqrt = np.sqrt(1.0 / self.q)
        lambda_min = sigma_sq * (1.0 - inv_q_sqrt) ** 2
        lambda_max = sigma_sq * (1.0 + inv_q_sqrt) ** 2
        return lambda_min, lambda_max

    def denoise(self, empirical_corr: np.ndarray) -> np.ndarray:
        n_assets = empirical_corr.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(empirical_corr)

        # sort eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[idx], 1e-10)
        eigenvectors = eigenvectors[:, idx]

        # iterative check via noise trace conservation
        k_signals = 0
        for candidate_k in range(n_assets):
            if candidate_k == 0:
                sigma_sq = 1.0
            else:
                # distribute residual trace uniformly over candidate noise modes
                sigma_sq = float((n_assets - np.sum(eigenvalues[:candidate_k]))/ (n_assets - candidate_k))

            _, lambda_max = self.compute_theoretical_bounds(sigma_sq)
            if eigenvalues[candidate_k] <= lambda_max:
                k_signals = candidate_k
                break

        # apply constant trace shrinkage across noise subspace
        cleaned_eigenvalues = eigenvalues.copy()
        if k_signals < n_assets:
            cleaned_eigenvalues[k_signals:] = np.mean(eigenvalues[k_signals:])

        # reconstruct correlation matrix
        cleaned_corr = eigenvectors @ np.diag(cleaned_eigenvalues) @ eigenvectors.T
        inv_sqrt_diag = 1.0 / np.sqrt(np.diag(cleaned_corr))
        cleaned_corr = cleaned_corr * np.outer(inv_sqrt_diag, inv_sqrt_diag)
        np.fill_diagonal(cleaned_corr, 1.0)
        return cleaned_corr