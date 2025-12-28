import numpy as np
from ..links import IdentityLink
from .family import Family

class Gaussian(Family):
    """Implementation of the Gaussian class"""
    def __init__(self, link=None):
        # Set default if link is None
        if link is None:
            link = IdentityLink()
        
        # Validate the link 
        if not hasattr(link, 'link') or not callable(link.link):
            raise AttributeError("link object must have 'link' method")
        if not hasattr(link, "inverse") or not callable(link.inverse):
            raise AttributeError("link object must have 'inverse' method")
        if not hasattr(link, "derivative") or not callable(link.derivative):
            raise AttributeError("link object must have 'derivative' method")
        if not hasattr(link, "inverse_derivative") or not callable(link.inverse_derivative):
            raise AttributeError("link object must have 'inverse_derivative' method")
        
        self.link = link

        def variance(mu):
            """Returning the theoretical value of V(mu)"""
            return np.ones_like(mu)
        
        #Deviance
        def deviance(y, mu, freq_weights=1.0):
            """Returning the theoretical equation of the deviance"""
            return np.sum(freq_weights * (y - mu)**2)
        
        def starting_mu(y):
            y_valid = self._validate_y(y)
            # Median for robustness
            mu_start = np.median(y_valid)
            return np.full_like(y_valid, mu_start)
        
        domain = (-np.inf, np.inf)

        # Initialize parent class
        super().__init__(link, variance, deviance, starting_mu, domain)
        
        # Additional attributes
        self._initialized = True

    def _validate_y(self, y):
        """Validating the gaussian response"""
        y = np.asarray(y, dtype=np.float64)

        if y.ndim != 1:
            raise ValueError(f"y must be 1D for Gaussian, got shape {y.shape}")
        
        if np.any(~np.isfinite(y)):
            non_count = np.sum(~np.isfinite(y))
            raise ValueError(f"y contains {non_count} non-finite values (NAN/inf)")

        # Warnings for extreme values
        if len(y) > 10:
            z_scores = np.abs((y - np.mean(y)) / (np.std(y) + 1e-10))
            extreme_idx = np.where(z_scores > 5)[0]
            if len(extreme_idx) > 0:
                print(f"Warning: {len(extreme_idx)} extreme values in y (|z| > 5)")
        
        return y 

    def log_likelihood(self, y, mu, scale=1.0, freq_weights=1.0, include_constant=True):
        """Log likelihood of the gaussian"""
        y = self._validate_y(y)
        
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        
        n = len(y)
        
        # log-likelihood calculation
        ll = -0.5 * n * np.log(2 * np.pi * scale)
        ll -= 0.5 * np.sum(freq_weights * (y - mu)**2) / scale

        if not include_constant:
            # without constant terms
            ll = -0.5 * np.sum(freq_weights * (y - mu)**2) / scale

        return ll
    
    def check_endog(self, y):
        """Validate endogenous variable (response) for this family"""
        y = np.asarray(y, dtype=np.float64)

        if y.ndim != 1:
            raise ValueError(f"y must be 1D for Gaussian, got shape {y.shape}")
        
        if np.any(~np.isfinite(y)):
            nan_count = np.sum(~np.isfinite(y))
            raise ValueError(f"y contains {nan_count} non-finite values (NAN/inf)")
        
        # Warning for constant y
        if np.allclose(y, y[0], rtol=1e-10):
            print(f"Warning: y is constant (all values = {y[0]:.3f})")
        
        return y
    
    def check_exog(self, X):
        """Validate exogenous variables """
        try:
            return super().check_exog(X)
        
        except AttributeError:
            X = np.asarray(X, dtype=np.float64)
            
            if X.ndim != 2:
                raise ValueError(f"X must be 2D, got shape {X.shape}")
            
            if X.shape[1] < 1:
                raise ValueError(f"X must have at least one column, got shape {X.shape}")
            
            if np.any(~np.isfinite(X)):
                raise ValueError("X contains NaN or infinite values")
            
            return X
    
    def __repr__(self):
        return f"Gaussian(link={self.link})"
    
    def __str__(self):
        return f"Gaussian family with {self.link} link"