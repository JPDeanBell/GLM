import numpy as np
from ..links import LogLink
from .family import Family

class Poisson(Family):
    """Implementation of the Gaussian class"""
    def __init__(self, link=None):
        # Set default if link is None
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

        def variance(mu):
            """Returning the theoretical value of V(mu)"""
            mu= np.asarray(mu,dtype=np.float64)
            return np.where(mu<0,self.eps*1e5,mu)
        
        
        def deviance(y, mu, freq_weights=1.0):
            """Returning the theoretical equation of the deviance"""
            
            y=np.asarray(y,dtype=np.float64)
            mu= np.asarray(mu,dtype=np.float64)

            #clipping for safe computation
            mu_clipped=np.clip(mu,self.eps*1e5,None)
            y_clipped= np.clip(y,self.eps,None)

            #handling y=0 
            dev= np.where(y>0,
                          2*(y_clipped *np.log(y_clipped/mu_clipped)-(y_clipped-mu_clipped)),
                          2*mu_clipped)

            #using weights
            if hasattr(freq_weights, '__len__'):
                freq_weights=np.asarray(freq_weights,dtype=np.float64)
                dev*= freq_weights

            return np.sum(dev)  


        def starting_mu(y):
            y =np.asarray(y,dtype=np.float64)
            if np.any(y<0):
                raise ValueError(f"y must >=0 for poisson")
            mu_start=np.mean(y)

            if mu_start==0:
                mu_start=.5
            return np.full_like(y,mu_start)
        
        #Domain for Poisson: mu>=0
        domain = (0, np.inf)

        # Initialize parent class
        super().__init__(link, variance, deviance, starting_mu, domain)
        
        # Additional attributes
        self._initialized = True

    def log_likelihood(self, y, mu, scale=1.0, freq_weights=1.0, include_constant=True):
        
        """Log likelihood of the Poisson"""
        y = np.asarray(y,dtype=np.float64)
        mu= np.asarray(mu,dtype=np.float64)
        
        if y.shape != mu.shape:
            raise ValueError(f"y shape {y.shape} must match mu shape {mu.shape}")
        
       #validating mu
        if np.any(mu<=0):
           invalid= mu[mu<=0]
           raise ValueError(f"mu must be positive for Poisson." 
                             f"Found {len(invalid)} non-positive values")
        
        #validating y
        if np.any(y<0) or np.any(np.mod(y,1)!=0):
            raise ValueError(f"y must be >=0 ")
        
        # clipping mu to avoid log(0)
        mu_clipped=np.clip(mu,self.eps*1e5,None)

        #lilelihood function
        ll= y*np.log(mu_clipped)-mu_clipped

        #add log factorial term if include_constant
        if include_constant:
            ll-=self._log_factorial(y)

        # Applying frequency weights
        if hasattr(freq_weights, '__len__'):
            freq_weights = np.asarray(freq_weights, dtype=np.float64)
            if len(freq_weights) != len(ll):
                raise ValueError(f"freq_weights length must match y length")
            if np.any(freq_weights < 0):
                raise ValueError("freq_weights cannot be negative")
            ll*= freq_weights

        return np.sum(ll)
    
    def _log_factorial(self,n):

        """Computes log(n!) with stirling's approximation"""
        n=np.asarray(n,dtype=np.float64)

        res= np.zeros_like(n,dtype=np.float64)

        #for n=0 or 1 log(0!)= log(1!)=0
        mask_small= (n==0) | (n==1)
        res[mask_small]=0.0

        #for larger n stirling's approx
        mask_large= ~mask_small
        if np.any(mask_large):
            n_large= n[mask_large]

            #approximation
            res[mask_large]=(n_large*np.log(n_large+self.eps)-n_large+
                             .5*np.log(2 * np.pi * n_large))
        return res
    
    def check_endog(self, y):
        """Validate endogenous variable (response) for this family"""
        y = np.asarray(y, dtype=np.float64)

        if y.ndim != 1:
            raise ValueError(f"y must be 1D for Gaussian, got shape {y.shape}")
        
        if np.any(~np.isfinite(y)):
            nan_count = np.sum(~np.isfinite(y))
            raise ValueError(f"y contains {nan_count} non-finite values (NAN/inf)")
        
        # Warning for constant y
        if np.any(y <0):
            raise ValueError(f"y contains {nan_count} non-finite values")
        
        if np.any(np.mod(y,1) !=0):
            print("Warning: y contains non-integer values for Poisson family")
        
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
