"""
Module: spectral_denoising
Implements Marčenko-Pastur analytical fitting, spectral boundary extraction,
and constant trace shrinkage for empirical correlation matrices.
"""

from dataclasses import dataclass
import numpy as np
import scipy.optimize as opt
from scipy.spatial.distance import pdist, squareform


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
            raise ValueError(f"Q factor must be >= 1.0, provided: {q_factor}")
        self.q = q_factor

    def compute_theoretical_bounds(self, sigma_sq: float = 1.0) -> SpectralBounds:
        inv_q_sqrt = np.sqrt(1.0 / self.q)
        lambda_min = sigma_sq * (1.0 - inv_q_sqrt) ** 2
        lambda_max = sigma_sq * (1.0 + inv_q_sqrt) ** 2
        return SpectralBounds(lambda_min=lambda_min, lambda_max=lambda_max, noise_variance=sigma_sq)

    def fit_empirical_sigma(self, eigenvalues: np.ndarray) -> float:
        """
        Fits the underlying noise variance parameter sigma^2 by minimizing the
        squared distance between empirical noise eigenvalues and the MP law.
        """
        n = len(eigenvalues)

        def objective(sigma_sq_candidate: float) -> float:
            bounds = self.compute_theoretical_bounds(sigma_sq_candidate)
            noise_eigs = eigenvalues[
                (eigenvalues >= bounds.lambda_min) & (eigenvalues <= bounds.lambda_max)
            ]
            if len(noise_eigs) == 0:
                return 1e6
            # Theoretical variance of MP density
            theoretical_var = (sigma_sq_candidate**2) * (
                (1.0 + 1.0 / self.q) - (1.0 - 1.0 / self.q) ** 2 / 2.0
            )
            empirical_var = float(np.var(noise_eigs))
            return (empirical_var - theoretical_var) ** 2

        res = opt.minimize_scalar(objective, bounds=(1e-4, 2.0), method="bounded")
        return float(res.x)

    def denoise_correlation_matrix(self, empirical_corr: np.ndarray) -> np.ndarray:
        """
        Executes Constant Trace Shrinkage on the correlation matrix.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(empirical_corr)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[idx], 1e-12)
        eigenvectors = eigenvectors[:, idx]

        sigma_sq = self.fit_empirical_sigma(eigenvalues)
        bounds = self.compute_theoretical_bounds(sigma_sq)

        # Identify signals versus noise bulk
        signal_mask = eigenvalues > bounds.lambda_max
        num_signals = int(np.sum(signal_mask))

        cleaned_eigenvalues = eigenvalues.copy()
        if num_signals < len(eigenvalues):
            noise_average = np.mean(eigenvalues[~signal_mask])
            cleaned_eigenvalues[~signal_mask] = noise_average

        # Reconstruct regularized matrix
        reconstructed = eigenvectors @ np.diag(cleaned_eigenvalues) @ eigenvectors.T

        # Re-scale diagonal to maintain exact unit correlation matrix properties
        inv_sqrt_diag = 1.0 / np.sqrt(np.diag(reconstructed))
        cleaned_corr = reconstructed * np.outer(inv_sqrt_diag, inv_sqrt_diag)
        np.fill_diagonal(cleaned_corr, 1.0)
        return cleaned_corr