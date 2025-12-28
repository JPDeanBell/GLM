import numpy as np
from ..links import LogLink
from .family import Family

class NegativeBinomial(Family):
    """Implementation for the Negative binomial class"""
    
    def __init__(self, link=None, theta=1.0):
        """ Initialize Negative Binomial family."""
        # Set default link if None
        if link is None:
            link = LogLink()
        
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
        
        # Validate theta 
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        self.theta = float(theta)
        
        # Variance function:
        def variance(mu):
            mu = np.asarray(mu, dtype=np.float64)
            if np.any(mu < 0):
                raise ValueError(f"mu must be non-negative for Negative Binomial")
            return mu + mu**2 / self.theta
        
        # Deviance function
        def deviance(y, mu, freq_weights=1.0):
            return self._deviance(y, mu, freq_weights)
        
        # Starting values
        def starting_mu(y):
            return self._starting_mu(y)
        
        # Domain
        domain = (0, np.inf)
        
        super().__init__(link, variance, deviance, starting_mu, domain)
        
        self._initialized = True
    
    def _deviance(self, y, mu, freq_weights=1.0):

        """Negative Binomial deviance"""
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        
        if y.shape != mu.shape:
            raise ValueError(f"y shape {y.shape} must match mu shape {mu.shape}")
        
        # Validate y and mu
        if np.any(y < 0) or np.any(np.mod(y, 1) != 0):
            raise ValueError("y must be non-negative integers for Negative Binomial")
        if np.any(mu <= 0):
            raise ValueError("mu must be positive for Negative Binomial")
        
        y_safe = np.clip(y, self.eps, None)
        mu_safe = np.clip(mu, self.eps, None)
        theta_safe = max(self.theta, self.eps)
        
        # Deviance formula 
        term1 = y_safe * np.log(y_safe / mu_safe)
        term2 = (y_safe + theta_safe) * np.log((y_safe + theta_safe) / (mu_safe + theta_safe))
        
        dev = 2 * (term1 - term2)
        
        # Handling y = 0 case
        mask_zero = (y == 0)
        if np.any(mask_zero):
            mu_zero = mu_safe[mask_zero]
            dev[mask_zero] = 2 * theta_safe * np.log((theta_safe + mu_zero) / theta_safe)
        
        # frequency weights
        if hasattr(freq_weights, '__len__'):
            freq_weights = np.asarray(freq_weights, dtype=np.float64)
            if len(freq_weights) != len(dev):
                raise ValueError(f"freq_weights length must match y length")
            if np.any(freq_weights < 0):
                raise ValueError("freq_weights cannot be negative")
            dev *= freq_weights
        
        return np.sum(dev)
    
    def _starting_mu(self, y):
        """Compute starting values for mu"""
        y = np.asarray(y, dtype=np.float64)
        
        if np.any(y < 0):
            raise ValueError("y must be non-negative for Negative Binomial")
        
        # Starting with mean of y
        mu_start = np.mean(y)
        if mu_start == 0:
            mu_start = 0.5
        
        # Clipping
        mu_start = np.clip(mu_start, 1e-4, 1e6)
        return np.full_like(y, mu_start)
    
    def log_likelihood(self, y, mu, scale=1.0, freq_weights=1.0, include_constant=True):
        
        """Log likelihood"""
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        
        if y.shape != mu.shape:
            raise ValueError(f"y shape {y.shape} must match mu shape {mu.shape}")
        
        # Validate inputs
        if np.any(y < 0) or np.any(np.mod(y, 1) != 0):
            raise ValueError("y must be non-negative integers for Negative Binomial")
        if np.any(mu <= 0):
            raise ValueError("mu must be positive for Negative Binomial")
        
        # Scale parameter 
        if scale != 1.0:
            print(f"Warning: Negative Binomial has fixed scale=1.0, ignoring scale={scale}")
        
        y_safe = np.clip(y, self.eps, None)
        mu_safe = np.clip(mu, self.eps, None)
        theta_safe = max(self.theta, self.eps)
        
        # Main terms of log-likelihood
        t1 = self._log_gamma(y_safe + theta_safe)  
        t2 = self._log_gamma(theta_safe)           
        t3 = theta_safe * np.log(theta_safe)       
        t4 = theta_safe * np.log(theta_safe + mu_safe)
        t5 = y_safe * np.log(mu_safe)              
        t6 = y_safe * np.log(theta_safe + mu_safe)
        
        # Log-likelihood
        loglike_val = t1 - t2 + t3 - t4 + t5 - t6
        
        # Subtract log(y!) if requested
        if include_constant:
            loglike_val -= self._log_factorial(y)
        
        # Applying frequency weights
        if hasattr(freq_weights, '__len__'):
            freq_weights = np.asarray(freq_weights, dtype=np.float64)
            if len(freq_weights) != len(loglike_val):
                raise ValueError(f"freq_weights length must match y length")
            if np.any(freq_weights < 0):
                raise ValueError("freq_weights cannot be negative")
            loglike_val *= freq_weights
        
        return np.sum(loglike_val)
    
    def _log_gamma(self, x):

        """Compute log(Γ(x)) using Stirling's approximation"""
        x = np.asarray(x, dtype=np.float64)
        
        # For x 
        mask_zero = x <= 0
        res = np.full_like(x, -np.inf)
        
        # For small x
        mask_small = (x > 0) & (x < 1)
        if np.any(mask_small):
            x_small = x[mask_small]
            res[mask_small] = self._log_gamma(x_small + 1) - np.log(x_small)
        
        # For larger x
        mask_large = x >= 1
        if np.any(mask_large):
            x_large = x[mask_large]
        
            res[mask_large] = (x_large - 0.5) * np.log(x_large) - x_large + 0.5 * np.log(2 * np.pi)
        
        return res
    
    def _log_factorial(self, n):

        """Compute log(n!) using Stirling's approximation"""
        n = np.asarray(n, dtype=np.float64)
        
        # For n = 0 or 1 log(0!) = log(1!) = 0
        mask_small = (n == 0) | (n == 1)
        result = np.zeros_like(n, dtype=np.float64)
        
        # For larger n Stirling's approximation
        mask_large = ~mask_small
        if np.any(mask_large):
            n_large = n[mask_large]
            result[mask_large] = (n_large * np.log(n_large + 1e-15) - n_large + 
                                 0.5 * np.log(2 * np.pi * n_large))
        
        return result
    
    def check_endog(self, y):

        """Validate response variable for Negative Binomial family"""
        y = np.asarray(y, dtype=np.float64)
        
        if y.ndim != 1:
            raise ValueError(f"y must be 1D for Negative Binomial, got shape {y.shape}")
        
        if np.any(~np.isfinite(y)):
            nan_count = np.sum(~np.isfinite(y))
            raise ValueError(f"y contains {nan_count} non-finite values")
        
        if np.any(y < 0):
            raise ValueError(f"y must be non-negative for Negative Binomial")
        
        if np.any(np.mod(y, 1) != 0):
            print("Warning: y contains non-integer values for Negative Binomial family")
        
        # Check for excessive zero
        zero_prop = np.mean(y == 0)
        if zero_prop > 0.8:
            print(f"Warning: {zero_prop:.1%} of y values are zero (zero-inflation)")
        
        return y
    
    def check_exog(self, X):
        """Validate exogenous variables"""
        try:
            return super().check_exog(X)
        except AttributeError:
            # Fallback implementation
            X = np.asarray(X, dtype=np.float64)
            
            if X.ndim != 2:
                raise ValueError(f"X must be 2D, got shape {X.shape}")
            
            if X.shape[1] < 1:
                raise ValueError(f"X must have at least one column, got shape {X.shape}")
            
            if np.any(~np.isfinite(X)):
                raise ValueError("X contains NaN or infinite values")
            
            return X
    
    def get_theta(self):
        """Get dispersion parameter θ"""
        return self.theta
    
    def set_theta(self, theta):
        """Update dispersion parameter θ"""
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        self.theta = float(theta)
        self.variance = lambda mu: mu + mu**2 / self.theta
    
    def estimate_theta(self, y, mu):
        """ Estimate theta from data using method of moments"""
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        
        # Calculate variance
        residuals = y - mu
        var_resid = np.mean(residuals**2)
        var_mu = np.mean(mu)
        
        # Estimate theta
        if var_resid > var_mu:
            theta_hat = var_mu**2 / (var_resid - var_mu)
            theta_hat = max(theta_hat, 1e-4)  # Ensure positive
            return theta_hat
        else:
            # Underdispersed or equidispersed
            return np.inf  
    
    def __repr__(self):
        return f"NegativeBinomial(link={self.link}, theta={self.theta:.3f})"
    
    def __str__(self):
        return f"Negative Binomial family with {self.link} link, θ={self.theta:.3f}"
    
    # Alternative parameterization methods
    def get_alpha(self):

        """Get alpha """
        return 1.0 / self.theta
    
    def set_alpha(self, alpha):

        """Set alpha"""
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.theta = 1.0 / alpha