import numpy as np
from ..links import PowerLink
from .family import Family

class Gamma(Family):
    """Implementation of the gamma function"""
    
    def __init__(self, link=None, theta=1.0):
        """Initialization of the class"""
        if link is None:
            link = PowerLink(power=-1)

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

        # dispersion parameter
        if theta <= 0:
            raise ValueError(f"Dispersion parameter must be positive, got{theta}")
        self.theta = float(theta)

        # V(u): u^2
        def variance(mu):
            """ Variance V(u) """
            return mu**2
        
        # deviance
        def deviance(y, mu, freq_weights=1.0):         
            y = np.asarray(y)
            mu = np.asarray(mu)

            y_clipped = np.clip(y, self.eps, None)
            mu_clipped = np.clip(mu, self.eps, None)

            dev = 2 * ((y_clipped - mu_clipped) / mu_clipped - np.log(y_clipped / mu_clipped))

            # using weights
            if hasattr(freq_weights, '__len__'):
                freq_weights = np.asarray(freq_weights, dtype=np.float64)
                dev *= freq_weights

            return np.sum(dev)
        
        def starting_mu(y):
            y = np.asarray(y, dtype=np.float64)
            if np.any(y < 0):
                raise ValueError(f"y must > 0 for Gamma")
            mu_start = np.mean(y)

            if mu_start == 0:
                mu_start = .5
            return np.full_like(y, mu_start)
        
        # Domain for Gamma
        domain = (0, np.inf)

        # Initialize parent class
        super().__init__(link, variance, deviance, starting_mu, domain)

        self._initialized = True
    
    def _lgamma(self, x):
        """Log gamma function using Stirling's approximation or math.lgamma"""
        try:
            # Try to use math.lgamma if available (more accurate)
            import math
            return math.lgamma(x)
        except ImportError:
            # Stirling's approximation for large x: log(Γ(x)) ≈ (x-0.5)*log(x) - x + 0.5*log(2π)
            # For smaller x, we can use the recursive property
            if x > 10:
                return (x - 0.5) * np.log(x) - x + 0.5 * np.log(2 * np.pi)
            else:
                # Use recursion for smaller values: Γ(x+1) = xΓ(x)
                # So log(Γ(x)) = log(Γ(x+n)) - Σ_{i=0}^{n-1} log(x+i)
                n = int(10 - x) + 1
                log_gamma_x_plus_n = (x + n - 0.5) * np.log(x + n) - (x + n) + 0.5 * np.log(2 * np.pi)
                sum_logs = np.sum(np.log(x + np.arange(n)))
                return log_gamma_x_plus_n - sum_logs
    
    def log_likelihood(self, y, mu, scale=1.0, freq_weights=1.0, include_constant=True):
        """Log likelihood of the Gamma distribution """
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        
        if y.shape != mu.shape:
            raise ValueError(f"y shape {y.shape} must match mu shape {mu.shape}")
        
        # validating mu
        if np.any(mu <= 0):
            invalid = mu[mu <= 0]
            raise ValueError(f"mu must be positive for Gamma. Found {len(invalid)} non-positive values")
        
        # validating y 
        if np.any(y <= 0):
            raise ValueError(f"y must be > 0 for Gamma distribution")
        
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        
        # clipping mu to avoid log(0)
        mu_clipped = np.clip(mu, self.eps, None)
        y_clipped = np.clip(y, self.eps, None)
        
        # k = shape parameter = 1/scale
        k = 1.0 / scale
        
        ll = (k * np.log(k / mu_clipped) + 
              (k - 1) * np.log(y_clipped) - 
              k * y_clipped / mu_clipped)
        
        # Subtract logΓ(k) for the constant term
        if include_constant:
            ll -= self._lgamma(k)
        
        # Apply frequency weights
        if hasattr(freq_weights, '__len__'):
            freq_weights = np.asarray(freq_weights, dtype=np.float64)
            if len(freq_weights) != len(ll):
                raise ValueError(f"freq_weights length must match y length")
            if np.any(freq_weights < 0):
                raise ValueError("freq_weights cannot be negative")
            ll *= freq_weights
        
        return np.sum(ll)
    
    def check_exog(self, X):
        """Validate exogenous variables"""
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
        
    def check_endog(self, y):
        """Validate endogenous variable (response) for this family"""
        y = np.asarray(y, dtype=np.float64)

        if y.ndim != 1:
            raise ValueError(f"y must be 1D for Gamma, got shape {y.shape}")
        
        if np.any(~np.isfinite(y)):
            nan_count = np.sum(~np.isfinite(y))
            raise ValueError(f"y contains {nan_count} non-finite values (NAN/inf)")
        
        # check for positive values
        if np.any(y <= 0):
            non_pos_count = np.sum(y <= 0)
            raise ValueError(f"y must be > 0 for Gamma, found {non_pos_count} non-positive values")
        
        # Warning for constant y
        if np.allclose(y, y[0], rtol=1e-10):
            print(f"Warning: y is constant (all values = {y[0]:.3f})")

        return y
    
    def __repr__(self):
        return f"Gamma(link={self.link}, theta={self.theta})"
    
    def __str__(self):
        return f"Gamma family with {self.link} link (theta={self.theta})"