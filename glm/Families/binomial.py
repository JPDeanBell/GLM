import numpy as np
from ..links import LogitLink
from .family import Family

class Binomial(Family):
    """Implementation of the binomial class for GLMs"""
    def __init__(self, link=None, n_trials=1):
        # Set default link if None
        if link is None:
            link = LogitLink()
        
        # Validate the link 
        if not hasattr(link, 'link') or not callable(link.link):
            raise AttributeError("link object must have 'link' method")
        if not hasattr(link, "inverse") or not callable(link.inverse):
            raise AttributeError("link object must have 'inverse' method")
        if not hasattr(link, "derivative") or not callable(link.derivative):
            raise AttributeError("link object must have 'derivative' method")
        if not hasattr(link, "inverse_derivative") or not callable(link.inverse_derivative):
            raise AttributeError("link object must have 'inverse_derivative' method")
        
        # Test that link properly maps (0,1) to reels 
        try:
            test_mu = np.array([.1, .5, .9])
            test_eta = link.link(test_mu)
            test_back = link.inverse(test_eta)
            
            if not np.all(np.isfinite(test_eta)):
                raise ValueError("link function must produce finite values for mu ∈ (0,1)")
            
            if not np.allclose(test_mu, test_back, rtol=1e-7, atol=1e-10):
                raise ValueError('link.inverse(link(mu)) should recover mu')  
        except Exception as e:
            raise ValueError(f"Invalid link function: {e}")

        self.link = link 

        # Validating n_trials
        if isinstance(n_trials, (int, np.integer)):
            if n_trials <= 0:
                raise ValueError(f"n_trials must be positive, got {n_trials}")
            self.n_trials = n_trials
            self._n_array = None
        elif hasattr(n_trials, '__len__'):
            n_trials = np.asarray(n_trials, dtype=np.float64)
            if n_trials.ndim != 1:
                raise ValueError(f"n_trials must be 1D, got shape {n_trials.shape}")
            if np.any(n_trials <= 0):
                raise ValueError("All n_trials must be > 0")
            if np.any(np.mod(n_trials, 1) != 0):
                raise ValueError("n_trials must be integers")
            
            self.n_trials = np.mean(n_trials)
            self._n_array = n_trials.astype(int)
        else:
            raise TypeError(f"n_trials must be int or array-like, got {type(n_trials)}")
    
        # Variance function 
        def variance(mu):
            """V(mu) with validation"""
            eps = 1e-15
            mu = np.asarray(mu, dtype=np.float64)
            
            # Check domain
            if np.any(mu < 0) or np.any(mu > 1):
                raise ValueError(f"mu must be in [0,1], got min={mu.min():.3f}, max={mu.max():.3f}")
            
            # Safe calculation
            mu = np.clip(mu, eps, 1 - eps)
            if self._n_array is None:
                return self.n_trials * mu * (1 - mu)  
            else:
                # Handle case where n_trials is an array
                if len(mu) != len(self._n_array):
                    raise ValueError(f"mu length ({len(mu)}) must match n_trials length ({len(self._n_array)})")
                return self._n_array * mu * (1 - mu)
    
        # Deviance function 
        def deviance(y, mu, freq_weights=1.0):
            """Deviance function with validation"""
            return self._deviance(y, mu, freq_weights)  
    
        # Starting values function 
        def starting_mu(y):
            """Starting values with validation"""
            return self._starting_mu(y)  
    
        # Domain for binomial: probabilities between 0 and 1
        domain = (0, 1)

        # Initialize parent class
        super().__init__(link, variance, deviance, starting_mu, domain)
        
        # Additional attributes
        self._initialized = True

    def _validate_y(self, y, allow_matrix=False):
        """Validate response variable y"""
        # Convert y to np.array
        y = np.asarray(y, dtype=np.float64)

        if y.ndim == 2 and y.shape[1] == 2 and allow_matrix:
            # Two column format [successes, failures]
            successes = y[:, 0]
            failures = y[:, 1]

            # Validation
            if np.any(successes < 0):
                raise ValueError("Success counts cannot be negative")
            if np.any(failures < 0):
                raise ValueError("Failure counts cannot be negative")
            if np.any(np.mod(successes, 1) != 0):
                raise ValueError("Success counts must be integers")
            if np.any(np.mod(failures, 1) != 0):
                raise ValueError("Failure counts must be integers")
            
            # Total trials
            total = successes + failures
            if np.any(total <= 0):
                raise ValueError("Total trials (successes + failures) must be positive")
            
            if self._n_array is not None:
                if not np.array_equal(total, self._n_array):
                    raise ValueError("Total trials must match provided n_trials")
                
            proportions = successes / total
            return proportions, successes, {'format': 'matrix', 'total': total}
        
        elif y.ndim == 1:
            # Check if y contains integer counts (for Binomial with n_trials)
            is_counts = False
            if self._n_array is not None or self.n_trials > 1:
                # Check if values look like counts (integers)
                if np.all(np.mod(y, 1) == 0) and np.all(y >= 0):
                    is_counts = True
            
            if is_counts and self._n_array is not None:
                # y are counts, n_trials is an array
                successes = y.astype(int)
                if len(y) != len(self._n_array):
                    raise ValueError(f"y length ({len(y)}) must match n_trials length ({len(self._n_array)})")
                if np.any(successes > self._n_array):
                    raise ValueError("Successes cannot exceed n_trials")
                if np.any(successes < 0):
                    raise ValueError("Successes cannot be negative")
                proportions = successes / self._n_array
                return proportions, successes, {'format': 'counts_array'}
            
            elif is_counts and self.n_trials > 1:
                # y are counts, n_trials is a single integer
                successes = y.astype(int)
                if np.any(successes > self.n_trials):
                    raise ValueError(f"Successes cannot exceed n_trials={self.n_trials}")
                if np.any(successes < 0):
                    raise ValueError("Successes cannot be negative")
                proportions = successes / self.n_trials
                return proportions, successes, {'format': 'counts_scalar'}
            
            else:
                # y are proportions
                if np.any(y < 0) or np.any(y > 1):
                    raise ValueError(f"y must be in [0,1] for proportions, got min={y.min():.3f}, max={y.max():.3f}")
                
                if self._n_array is not None:
                    # Convert proportions to counts
                    successes = np.round(y * self._n_array).astype(int)
                    if np.any(successes < 0) or np.any(successes > self._n_array):
                        raise ValueError(f"y*n_trials must give valid counts in [0, n_trials]")
                    return y, successes, {"format": "proportions_array"}
                elif self.n_trials > 1:
                    # Binomial with single n_trials
                    successes = np.round(y * self.n_trials).astype(int)
                    return y, successes, {"format": "proportions_scalar"}
                else:
                    # Bernoulli
                    return y, None, {"format": "proportions"}
        else:
            raise ValueError(f"y must be 1D or 2D shape (n,2), got shape {y.shape}")

    
    def _deviance(self, y, mu, freq_weights=1.0):
        """Calculate binomial deviance"""
        # Input validation
        if not hasattr(y, '__len__'):
            raise TypeError(f"y must be array-like, got {type(y)}")
        
        if not hasattr(mu, '__len__'):
            raise TypeError(f"mu must be array-like, got {type(mu)}")
        
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)

        if y.shape != mu.shape:
            raise ValueError(f"y shape {y.shape} must match mu shape {mu.shape}")
        
        # Validate y
        y_prop, y_success, y_info = self._validate_y(y, allow_matrix=True)

        # Validate mu
        if np.any(mu < 0) or np.any(mu > 1):
            invalid_idx = np.where((mu < 0) | (mu > 1))[0]
            raise ValueError(f"mu must be in [0,1]. Invalid at indices {invalid_idx[:5]}, "
                           f"values: {mu[invalid_idx[:5]]}")
        
        # Safe calculation with clipping
        mu_clipped = np.clip(mu, self.eps, 1 - self.eps)

        if y_info['format'] == 'matrix':
            total = y_info['total']
            successes = y_success
            mu_success = mu_clipped * total

            # Initialize arrays for terms
            t1 = np.zeros_like(successes, dtype=np.float64)
            t2 = np.zeros_like(successes, dtype=np.float64)
            
            # Handle different cases separately for stability
            
            # Case 1: 0 < successes < total
            mask_middle = (successes > 0) & (successes < total)
            if np.any(mask_middle):
                successes_m = successes[mask_middle]
                total_m = total[mask_middle]
                mu_success_m = mu_success[mask_middle]
                
                # Use clipped successes for log calculation
                successes_clipped = np.clip(successes_m, self.eps, total_m - self.eps)
                t1[mask_middle] = successes_clipped * np.log(successes_clipped / mu_success_m)
                t2[mask_middle] = (total_m - successes_clipped) * np.log(
                    (total_m - successes_clipped) / (total_m - mu_success_m)
                )
            
            # Case 2: successes = 0
            mask_zero = (successes == 0)
            if np.any(mask_zero):
                total_0 = total[mask_zero]
                mu_success_0 = mu_success[mask_zero]
                # Only t2 term contributes
                t2[mask_zero] = total_0 * np.log(total_0 / (total_0 - mu_success_0))
            
            # Case 3: successes = total
            mask_total = (successes == total)
            if np.any(mask_total):
                total_T = total[mask_total]
                mu_success_T = mu_success[mask_total]
                # Only t1 term contributes
                t1[mask_total] = total_T * np.log(total_T / mu_success_T)
            
            dev = 2 * (t1 + t2)
        else:
            # Single column format
            y_clipped = np.clip(y_prop, self.eps, 1 - self.eps)
            
            # Initialize arrays
            t1 = np.zeros_like(y_prop, dtype=np.float64)
            t2 = np.zeros_like(y_prop, dtype=np.float64)
            
            # Handle different cases
            mask_middle = (y_prop > 0) & (y_prop < 1)
            if np.any(mask_middle):
                y_m = y_clipped[mask_middle]
                mu_m = mu_clipped[mask_middle]
                t1[mask_middle] = y_m * np.log(y_m / mu_m)
                t2[mask_middle] = (1 - y_m) * np.log((1 - y_m) / (1 - mu_m))
            
            mask_zero = (y_prop == 0)
            if np.any(mask_zero):
                mu_zero = mu_clipped[mask_zero]
                t2[mask_zero] = np.log(1 / (1 - mu_zero))
            
            mask_one = (y_prop == 1)
            if np.any(mask_one):
                mu_one = mu_clipped[mask_one]
                t1[mask_one] = np.log(1 / mu_one)
            
            dev = 2 * (t1 + t2)
            
            # Scale by n_trials for binomial (not Bernoulli)
            if y_info['format'] in ['counts_scalar', 'proportions_scalar'] and self.n_trials > 1:
                dev *= self.n_trials
            elif y_info['format'] in ['counts_array', 'proportions_array']:
                dev *= self._n_array
        
        # Apply frequency weights
        if hasattr(freq_weights, '__len__'):
            freq_weights = np.asarray(freq_weights, dtype=np.float64)
            if len(freq_weights) != len(dev):
                raise ValueError(f"freq_weights length ({len(freq_weights)}) must match y length ({len(dev)})")
            if np.any(freq_weights < 0):
                raise ValueError("freq_weights cannot be negative")
            
            dev *= freq_weights
        
        return np.sum(dev)

    def _starting_mu(self, y):
        """Compute starting values for μ"""
        # Validate y
        y_prop, y_success, y_info = self._validate_y(y, allow_matrix=True)

        # Calculate starting values based on format
        if y_info['format'] == 'matrix':
            total = y_info['total']
            successes = y_success
            mu_start = (successes + 0.5) / (total + 1)
        
        elif y_info['format'] == 'counts_scalar':
            # y are counts, n_trials is a single integer
            successes = y_success
            mu_start = (successes + 0.5) / (self.n_trials + 1)
        
        elif y_info['format'] == 'counts_array':
            # y are counts, n_trials is an array
            successes = y_success
            mu_start = (successes + 0.5) / (self._n_array + 1)
        
        elif y_info['format'] == 'proportions_scalar' and self.n_trials > 1:
            # y are proportions, n_trials is a single integer > 1
            successes = y_success if y_success is not None else np.round(y_prop * self.n_trials)
            mu_start = (successes + 0.5) / (self.n_trials + 1)
        
        elif y_info['format'] == 'proportions_array':
            # y are proportions, n_trials is an array
            successes = y_success if y_success is not None else np.round(y_prop * self._n_array)
            mu_start = (successes + 0.5) / (self._n_array + 1)
        
        else:
            # Bernoulli case or single proportion with n_trials=1
            mu_start = (y_prop + 0.5) / 2
        
        # Clip to safe range
        mu_start = np.clip(mu_start, self.eps, 1 - self.eps)
        return mu_start

    def log_likelihood(self, y, mu, scale=1.0, freq_weights=1.0, include_constant=True):
        """Binomial log-likelihood - FIXED WITHOUT SCIPY"""
        # Input validation
        if not hasattr(y, '__len__'):
            raise TypeError(f"y must be array-like, got {type(y)}")
        
        if not hasattr(mu, '__len__'):
            raise TypeError(f"mu must be array-like, got {type(mu)}")
        
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)

        if y.shape != mu.shape:
            raise ValueError(f"y shape {y.shape} must match mu shape {mu.shape}")
        
        # Validate mu
        if np.any(mu <= 0) or np.any(mu >= 1):
            invalid_mu = mu[(mu <= 0) | (mu >= 1)]
            raise ValueError(f"mu must be in (0,1) for log-likelihood. "
                           f"Found {len(invalid_mu)} invalid values "
                           f"(min={mu.min():.3e}, max={mu.max():.3e})")
        
        # Validate scale
        if not np.isscalar(scale):
            raise TypeError(f"scale must be scalar, got {type(scale)}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        
        # Safe calculation
        mu_clipped = np.clip(mu, self.eps, 1 - self.eps)
        
        # Validate y and get format info
        y_prop, y_success, y_info = self._validate_y(y, allow_matrix=True)

        loglike_val = None
        
        if y_info['format'] == 'matrix':
            total = y_info['total']
            successes = y_success

            # Probability terms
            loglike_val = (successes * np.log(mu_clipped) + 
                          (total - successes) * np.log(1 - mu_clipped))
            
            # Add combinatorial constant if requested
            if include_constant:
                loglike_val += self._log_comb(total, successes)
        else:
            # y are proportions or counts
            if self._n_array is None and self.n_trials == 1:
                # Bernoulli case
                # Use y_prop directly (no clipping for y in Bernoulli)
                y_safe = np.clip(y_prop, self.eps, 1 - self.eps)
                loglike_val = (y_safe * np.log(mu_clipped) + 
                              (1 - y_safe) * np.log(1 - mu_clipped))
            else:
                # Binomial case
                if y_info['format'] in ['counts_scalar', 'counts_array']:
                    successes = y_success
                elif self._n_array is not None:
                    successes = np.round(y_prop * self._n_array).astype(int)
                else:
                    # This is the key case: proportions with scalar n_trials
                    successes = np.round(y_prop * self.n_trials).astype(int)
                
                # Get n_trials for each observation
                if self._n_array is not None:
                    n_trials = self._n_array
                else:
                    # Create array of n_trials with same length as y
                    n_trials = np.full_like(y_prop, self.n_trials, dtype=int)
                
                # Clip successes to valid range
                successes = np.clip(successes, 0, n_trials)
                
                # Probability terms
                loglike_val = (successes * np.log(mu_clipped) + 
                              (n_trials - successes) * np.log(1 - mu_clipped))
                
                # Add combinatorial constant if requested
                if include_constant:
                    loglike_val += self._log_comb(n_trials, successes)
        
        # Apply frequency weights
        if hasattr(freq_weights, '__len__'):
            freq_weights = np.asarray(freq_weights, dtype=np.float64)
            if len(freq_weights) != len(loglike_val):
                raise ValueError(f"freq_weights length ({len(freq_weights)}) must match y length ({len(loglike_val)})")
            if np.any(freq_weights < 0):
                raise ValueError("freq_weights cannot be negative")
        loglike_val *= freq_weights
        
        return np.sum(loglike_val)

    def _log_comb(self, n, k):
        """Compute log binomial coefficient using Stirling's approximation"""
        # Ensure n and k are arrays
        n = np.asarray(n, dtype=np.float64)
        k = np.asarray(k, dtype=np.float64)
        
        # Handle scalar inputs
        if n.ndim == 0:
            n = np.array([n])
        if k.ndim == 0:
            k = np.array([k])
        
        # Initialize result array
        result = np.zeros_like(k, dtype=np.float64)
        
        # Edge cases: k = 0 or k = n
        mask_edge = (k == 0) | (k == n)
        result[mask_edge] = 0.0
        
        # Other cases: use Stirling's approximation
        mask_other = ~mask_edge
        if np.any(mask_other):
            n_other = n[mask_other]
            k_other = k[mask_other]
            
            # Avoid log(0) by adding epsilon
            k_safe = np.maximum(k_other, self.eps)
            nk_safe = np.maximum(n_other - k_other, self.eps)
            
            # Stirling's approximation for log(C(n,k)):
            # log(C(n,k)) ≈ log(√(n/(2πk(n-k)))) + n*log(n) - k*log(k) - (n-k)*log(n-k)
            
            # Better formulation to avoid overflow:
            # log(C(n,k)) = 0.5*log(n/(2πk(n-k))) + n*log(n) - k*log(k) - (n-k)*log(n-k)
            
            # First term: log(√(n/(2πk(n-k))))
            term1 = 0.5 * np.log(n_other / (2 * np.pi * k_safe * nk_safe))
            
            # Second term: n*log(n) - k*log(k) - (n-k)*log(n-k)
            term2 = (n_other * np.log(n_other) - 
                    k_safe * np.log(k_safe) - 
                    nk_safe * np.log(nk_safe))
            
            result[mask_other] = term1 + term2
        
        return result

    def fitted(self, eta):
        """Apply inverse link to get probabilities with validation"""
        if not hasattr(eta, '__len__'):
            raise TypeError(f"eta must be array-like, got {type(eta)}")
        
        eta = np.asarray(eta, dtype=np.float64)

        try:
            mu = self.link.inverse(eta)
        except Exception as e:
            raise RuntimeError(f"Error applying inverse link: {e}")
        
        # Clip to avoid numerical issues
        mu = np.clip(mu, self.eps, 1 - self.eps)

        # Check for NaN or inf
        if np.any(~np.isfinite(mu)):
            nan_idx = np.where(~np.isfinite(mu))[0]
            raise ValueError(f"Inverse link produced non-finite values at indices {nan_idx[:5]}, "
                           f"eta values: {eta[nan_idx[:5]]}")
        
        return mu

    def predict(self, eta, type="response"):
        """Make predictions"""
        if type not in ["response", "link", "class"]:
            raise ValueError(f"type must be 'response', 'link', or 'class', got '{type}'")
        
        if type == "link":
            return np.asarray(eta, dtype=np.float64)
        elif type == "class":
            prob = self.fitted(eta)
            return (prob >= .5).astype(int)
        else:
            return self.fitted(eta)
    
    def check_endog(self, y):
        """Validate endogenous variable (response) for this family"""
        try:
            y_clean, _, _ = self._validate_y(y, allow_matrix=True)
            return y_clean
        except Exception as e:
            raise ValueError(f"Invalid response for Binomial family: {e}")
    
    def __repr__(self):
        n_str = self._n_array if self._n_array is not None else self.n_trials
        return f"Binomial(link={self.link}, n_trials={n_str})"  
    
    def __str__(self):
        if self._n_array is None:
            n_desc = f"n={self.n_trials}"
        else:
            n_desc = f"n varies ({len(self._n_array)} values, mean={self.n_trials:.1f})"
        
        return f"Binomial family with {self.link} link, {n_desc}"