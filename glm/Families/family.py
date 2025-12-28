import numpy as np
from abc import abstractmethod
 

class Family:
    """Base class for Exponential families"""
    def __init__(self, link, variance, deviance, starting_mu, domain, eps=1e-15):
        self.link = link
        self.variance = variance
        self.deviance = deviance
        self.starting_mu = starting_mu
        self.domain = domain
        self.eps = eps

    def fitted(self, eta):
        mu = self.link.inverse(eta)
        return np.clip(mu, self.domain[0] + self.eps, self.domain[1] - self.eps)
    

    def predict(self,eta,type="response"):

        if type not in["response","link"]:
            raise ValueError(f"type must be 'response' or 'link' ,got {type} instead")
        if type=='link':
            return np.asarray(eta,dtype=np.float64)
        else:
            return self.fitted(eta)
        
    def check_exog(self, X):
        """Validate design matrix X (common to all families)"""
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        
        if X.shape[1] < 1:
            raise ValueError(f"X must have at least one column, got shape {X.shape}")
        
        # Check for NaN/inf (common to all families)
        if np.any(~np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values")
        
        # Check for constant columns (common issue for all families)
        col_std = np.std(X, axis=0)
        constant_cols = np.where(col_std < self.eps)[0]
        
        if len(constant_cols) > 1:  # Allow one constant column (intercept)
            for col in constant_cols[1:]:
                print(f"Warning: Column {col} of X is constant (std={col_std[col]:.2e})")
        
        # Optional: Check rank deficiency (common to all families)
        if X.shape[0] > X.shape[1]:
            rank = np.linalg.matrix_rank(X)
            if rank < X.shape[1]:
                print(f"Warning: X is rank deficient (rank={rank} < n_features={X.shape[1]})")
        
        return X
    
    @abstractmethod
    def log_likelihood(self, y, mu, scale=1.0, freq_weights=1.0):
        pass

    @abstractmethod
    def check_endog(self,y):
        pass
