import numpy as np
import warnings
from typing import Optional, Tuple,Dict, Any, Union
from scipy import linalg
from scipy.special import erf
from glm import Gaussian

class GLM:
    """Glm implementation"""
    def __init__(self, endog, exog, family=None, offset=None,
                 exposure=None,freq_weights=None, var_weights=None,
                 missing='none', hasconst=False):
        
        #Set family 
        if family is None:
            family= Gaussian()
        self.family= family
        
        #initialiaz basic attributes
        self.endog=self.family.check_endog(endog)
        self.exog= self.family.check_exog(exog)
        
        #some params
        self.nobs = self.endog.shape[0]
        self.nvars = self.exog.shape[1]
        

        #Initialization of the rest
        self.offset = self._validate_offset(offset)
        self.exposure = self._validate_exposure(exposure)
        self.freq_weights = self._validate_weights(freq_weights, 'freq_weights')
        self.var_weights = self._validate_weights(var_weights, 'var_weights')
        self.hasconst = hasconst
        self.missing = missing

        # Initialize model parameters
        self.nobs = self.endog.shape[0]
        self.nvars = self.exog.shape[1]
        self.params = None
        self.scale = None
        self.cov_params = None
        self.bse = None
        self.llf = None
        self.deviance = None
        self.null_deviance = None
        self.df_resid = None
        self.df_model = None
        self.aic = None
        self.bic = None
        self.converged = False
        self.iterations = 0
        self.fit_history = []
    
        #convergence criterias
        self.tol= 1e-6
        self.maxiter= 100

        #check if constant should be added
        if not hasconst and self.add_constant:
            self._add_constant()
            self.hasconst= True

    @property
    def add_constant(self):

        """If constant should be added to exog"""

        if self.hasconst:
            return False
        
        #check if first column is all 1s
        if self.exog.shape[1]>0:
            first_col= self.exog[:,0]
            if np.allclose(first_col,1.0):
                return False
        
        return True
    
    def _validate_family_data(self):

        """Validate that data is compatible with family"""
        # Check endog using family's check_endog method
        try:
            self.family.check_endog(self.endog)
        except AttributeError:
            # If family doesn't have check_endog, use our own validation
            if hasattr(self.family, 'domain'):
                domain = self.family.domain
                if domain is not None:
                    if domain[0] is not None and np.any(self.endog < domain[0]):
                        warnings.warn(f"endog values below domain minimum {domain[0]}")
                    if domain[1] is not None and np.any(self.endog > domain[1]):
                        warnings.warn(f"endog values above domain maximum {domain[1]}")


    def _validate_offset(self,offset):

        """Validating offset term"""
        if offset is None:
            return np.zeros(self.nobs)
        
        offset= np.asarray(offset, dtype=np.float64)

        if offset.shape ==(self.nobs,):
            return offset
        elif offset.shape==(self.nobs,1):
            return offset.ravel()
        else:
            raise ValueError(f"offset must have shape ({self.nobs},) or shape ({self.nobs},1)")
    
    def _validate_exposure(self,exposure):

        """Validate exposure term"""
        if exposure is None:
            return np.ones(self.nobs)
        
        exposure= np.asarray(exposure, dtype=np.float64)

        if exposure.shape == (self.nobs,):
            if np.any(exposure <= 0):
                raise ValueError("exposure must be positive")
            return exposure
        elif exposure.shape == (self.nobs, 1):
            exposure = exposure.ravel()
            if np.any(exposure <= 0):
                raise ValueError("exposure must be positive")
            return exposure
        else:
            raise ValueError(f"exposure must have shape ({self.nobs},) or ({self.nobs}, 1)")
    
    def _validate_weights(self, weights, name):

        """Validate weights"""
        if weights is None:
            return np.ones(self.nobs)
        
        weights = np.asarray(weights, dtype=np.float64)
        
        if weights.shape == (self.nobs,):
            if np.any(weights < 0):
                raise ValueError(f"{name} must be non-negative")
            return weights
        elif weights.shape == (self.nobs, 1):
            weights = weights.ravel()
            if np.any(weights < 0):
                raise ValueError(f"{name} must be non-negative")
            return weights
        else:
            raise ValueError(f"{name} must have shape ({self.nobs},) or ({self.nobs},1)") 
        
    def _add_constant(self):

        """Add constant term to exog"""
        if not self.hasconst:
            const = np.ones((self.nobs, 1))
            self.exog = np.hstack([const, self.exog])
    
    def fit(self,start_params=None, tol=None,maxiter=None,scale=None,
            cov_type='nonrobust',cov_kwds=None):
        """Using IRLS"""
        #Setting up convergence parameters
        if tol is not None:
            self.tol= tol
        if maxiter is not None:
            self.maxiter= maxiter
        #Initialize parameters
        if start_params is None:
            start_params=self._get_start_params()
        else:
            start_params= np.asarray(start_params,dtype=np.float64)
            if start_params.shape != (self.nvars,):
                raise ValueError(f"start_params must have shape ({self.nvars},)")
            
        self.params= start_params.copy()
        #Tracking
        self.iterations=0
        self.fit_history=[]
        self.converged= False
        #link function
        link= self.family.link

        #Initialize mu and eta- use better starting values
        mu= self.family.starting_mu(self.endog)
        eta=link.link(mu)

        #main IRLS loop
        for i in range(self.maxiter):
            old_params= self.params.copy()

            #Linear predictor wth offset and exposure
            eta_current= np.dot(self.exog,self.params)
            #add offset
            if self.offset is not None:
                eta_current+= self.offset
            # Update mu using inverse link
            mu= link.inverse(eta_current)
            mu= np.maximum(mu,1e-8)
            #For Poisson-like models with exposure
            if self.exposure is not None and hasattr(self.family,'scale'):
                mu= mu*self.exposure
            # Calculate working weights and response
            mu_deriv= link.derivative(mu)
            var_func= self.family.variance(mu)

            #combining weights
            weights= self._get_weights()
            if weights is not None:
                var_func= var_func/weights
            #Working weights
            denominator= var_func * mu_deriv**2
            denominator= np.maximum(denominator,1e-8)
            w= weights/ denominator

            # working response
            z= eta_current+(self.endog-mu)*mu_deriv

            #Weighted least squares
            XW= self.exog * w[:, np.newaxis]
            XWX= np.dot(self.exog.T, XW)
            XWz= np.dot(self.exog.T, w*z)

            try:
                #solving normal equations
                params_new= linalg.solve(XWX,XWz,assume_a='sym')
            except linalg.LinAlgError:
                #Pseudo-inverse if singular
                params_new=np.dot(linalg.pinv(XWX),XWz)
            #update params
            self.params= params_new
            #check convergence
            param_change= np.max(np.abs(self.params-old_params))
            if param_change <self.tol:
                self.converged= True
                break
            #Store iteration info
            self.fit_history.append({
                'iteration':i+1,
                'params': self.params.copy(),
                'change':param_change
            })
        self.iterations=i+1
        #Calculate final fitted values
        self.eta= np.dot(self.exog,self.params)
        if self.offset is not None:
            self.eta+= self.offset
        
        self.mu= link.inverse(self.eta)
        self.mu= np.maximum(self.mu,1e-8)

        if self.exposure is not None and hasattr(self.family,'scale'):
            self.mu= self.mu* self.exposure
        
        #scale parameter
        self.scale= self.estimate_scale(scale)

        #cov matrix
        self.cov_params= self.calc_covariance(cov_type,cov_kwds)
        self.bse= np.sqrt(np.diag(self.cov_params))

        #calculate diagnostics
        self._calculate_diagnostics()

        return self
    
    def _get_start_params(self):
        """Get starting parameters for Poisson regression"""
        if hasattr(self.family.link,'__class__') and self.family.link.__class__.__name__ =='LogLink':
            # start with intercept-only model
            y=self.endog

            #if we have offset,adjust for it
            if self.offset is not None:
                mean_y= np.maximum(np.mean(y),1e-8) #to avoid log(0)
                intercept= np.log(mean_y)

                #if we have an intercept column (first column is all ones)
                if np.allclose(self.exog[:,0],1.0):
                    params= np.zeros(self.nvars)
                    params[0]=intercept
                    if self.offset is not None:
                        params[0]-=np.mean(self.offset) #adjust for average offset
                    return params
            #fallback: use OLS on transformed data
            try:
                #transform y to log scale (add small constant to avoid log(0))
                y_transformed= np.log(y+.5)

                #adjust for offset if present
                if self.offset is not None:
                    y_transformed-=self.offset
                # use QR decomposition for stability
                Q,R= np.linalg.qr(self.exog)
                params= linalg.solve_triangular(R,Q.T @ y_transformed)
                #Ensure intercept is reasonable
                if self.offset is not None:
                    #adjust intercept for average offset
                    if np.allclose(self.exog[:,0],1.0):
                        params[0]-=np.mean(self.offset)
                return params
            except linalg.LinAlgError:
                # Fall back to normal equations
                XTX= np.dot(self.exog.T,self.exog)
                XTy= np.dot(self.exog.T,y_transformed)
                try:
                    params=linalg.solve(XTX,XTy,assume_a='sym')
                    return params
                except linalg.LinAlgError:
                    # use zero starting values
                    return np.zeros(self.nvars)
        else:
            #for other link functions,use the original method
            try:
                Q,R=np.linalg.qr(self.exog)
                params=linalg.solve_triangular(R,Q.T @ self.endog)
            except linalg.LinAlgError:
                XTX= np.dot(self.exog.T,self.exog)
                XTy=np.dot(self.exog.T,self.endog)
                try:
                    params= linalg.solve(XTX,XTy,assume_a='sym')
                except linalg.LinAlgError:
                    params=np.zeros(self.nvars)
            return params
    
    def _get_weights(self):
        """Combine frequency and variance weights"""
        weights = np.ones(self.nobs)
        
        if self.freq_weights is not None:
            weights = weights * self.freq_weights
        
        if self.var_weights is not None:
            weights = weights * self.var_weights
        
        return weights
    
    def estimate_scale(self, scale=None):
        
        if scale is not None:
            if isinstance(scale, (int, float)):
                return float(scale)
            elif isinstance(scale, str):
                if scale.lower() == 'pearson':
                    return self._pearson_chi2()
                elif scale.lower() == 'dev':
                    return self._deviance_scale()
        
        # Default: use family's scale if fixed
        if hasattr(self.family, 'scale') and self.family.scale is not None:
            return self.family.scale
        
        # Estimate from data
        return self._deviance_scale()
    
    def _pearson_chi2(self):
        """Pearson chi-squared estimator of scale"""
        weights = self._get_weights()
        if weights is None:
            weights = 1.0
        
        # Pearson residuals
        pearson_resid = (self.endog - self.mu) / np.sqrt(self.family.variance(self.mu))
        
        # Weighted sum
        chi2 = np.sum(weights * pearson_resid**2)
        
        # Degrees of freedom
        df = self.nobs - self.nvars
        
        if df > 0:
            return chi2 / df
        else:
            warnings.warn("Degrees of freedom <= 0, returning 1.0")
            return 1.0
    
    def _deviance_scale(self):
        """Deviance-based estimator of scale"""
        weights = self._get_weights()
        if weights is None:
            weights = 1.0
        
        dev = self.family.deviance(self.endog, self.mu, freq_weights=weights)
        
        # Degrees of freedom
        df = self.nobs - self.nvars
        
        if df > 0:
            return dev / df
        else:
            warnings.warn("Degrees of freedom <= 0, returning 1.0")
            return 1.0
    
    def calc_covariance(self, cov_type='nonrobust', cov_kwds=None):

        """Calculating covariance matrix"""
        if cov_kwds is None:
            cov_kwds={}
        
        #Basic GLM cov
        link= self.family.link
        mu_deriv= link.derivative(self.mu)
        var_func= self.family.variance(self.mu)
        weights= self._get_weights()

        if weights is not None:
            var_func= var_func/weights

        #working weights
        w = weights /(var_func* mu_deriv**2)
        
        #information matrix
        XW= self.exog*w[:, np.newaxis]
        XWX= np.dot(self.exog.T, XW)

        try:
            #inverse of the information matrix
            cov= linalg.inv(XWX)*self.scale
        except linalg.LinAlgError:
            cov= linalg.pinv(XWX)*self.scale
        
        #apply robust covariance corrections if needed
        if cov_type.lower()!='nonrobust':
            cov= self._robust_covariance(cov,cov_type,cov_kwds)
        
        return cov
    def _robust_covariance(self, cov_base, cov_type, cov_kwds):
        """Calculate robust covariance matrix"""
        # Calculate residuals
        if cov_type.upper() in ['HC0', 'HC1', 'HC2', 'HC3']:
            # Sandwich covariance: (X'X)^-1 X' Ω X (X'X)^-1
            
            # Get residuals
            if hasattr(self, 'resid_response'):
                resid = self.resid_response
            else:
                resid = self.endog - self.mu
            
            # Calculate middle part of sandwich
            link = self.family.link
            mu_deriv = link.derivative(self.mu)
            var_func = self.family.variance(self.mu)
            weights = self._get_weights()
            
            if weights is not None:
                var_func = var_func / weights
            
            # For HC estimators
            if cov_type.upper() == 'HC0':
                # No adjustment
                h = np.ones_like(resid)
            elif cov_type.upper() == 'HC1':
                # Small sample adjustment
                h = np.sqrt(self.nobs / (self.nobs - self.nvars))
            elif cov_type.upper() == 'HC2':
                # Leverage-based adjustment
                XW = self.exog * (weights / (var_func * mu_deriv**2))[:, np.newaxis]
                XWX_inv = linalg.inv(np.dot(self.exog.T, XW))
                leverage = np.sum(self.exog * np.dot(self.exog, XWX_inv), axis=1)
                h = 1 / np.sqrt(1 - leverage)
            elif cov_type.upper() == 'HC3':
                # More conservative adjustment
                leverage = self._get_leverage()
                h = 1 / (1 - leverage)
            
            # Weighted squared residuals
            omega = (resid**2) * (h**2)
            
            # Sandwich covariance
            XOmega = self.exog * omega[:, np.newaxis]
            XOmegaX = np.dot(self.exog.T, XOmega)
            sandwich = np.dot(np.dot(cov_base, XOmegaX), cov_base)
            
            return sandwich
        
        else:
            warnings.warn(f"Unknown cov_type '{cov_type}', returning non-robust")
            return cov_base
    
    def _get_leverage(self):
        """Calculate leverage (hat) values"""
        link = self.family.link
        mu_deriv = link.derivative(self.mu)
        var_func = self.family.variance(self.mu)
        weights = self._get_weights()
        
        if weights is not None:
            var_func = var_func / weights
        
        w = weights / (var_func * mu_deriv**2)
        
        # Weighted X
        sqrt_w = np.sqrt(w)
        X_weighted = self.exog * sqrt_w[:, np.newaxis]
        
        # Hat matrix diagonal
        try:
            H = np.dot(X_weighted, linalg.pinv(X_weighted))
            leverage = np.diag(H)
        except:
            # Fallback
            leverage = np.zeros(self.nobs)
        
        return leverage
    
    def _calculate_diagnostics(self):
        """Calculate model diagnostics"""
        # Log-likelihood
        weights = self._get_weights()
        self.llf = self.family.log_likelihood(
            self.endog, self.mu, 
            scale=self.scale, 
            freq_weights=weights
        )
        
        # Deviance
        self.deviance = self.family.deviance(
            self.endog, self.mu, 
            freq_weights=weights
        )
        
        # Null deviance (intercept-only model)
        null_mu = np.full_like(self.endog, np.mean(self.endog))
        self.null_deviance = self.family.deviance(
            self.endog, null_mu,
            freq_weights=weights
        )
        
        # Degrees of freedom
        self.df_model = self.nvars - (1 if self.hasconst else 0)
        self.df_resid = self.nobs - self.nvars
        
        # Information criteria
        self.aic = -2 * self.llf + 2 * self.nvars
        self.bic = -2 * self.llf + np.log(self.nobs) * self.nvars
        
        # Calculate residuals
        self.resid_response = self.endog - self.mu
        self.resid_pearson = self.resid_response / np.sqrt(self.family.variance(self.mu))
        self.resid_deviance = self._deviance_residuals()
        
        # Calculate working residuals
        link = self.family.link
        mu_deriv = link.derivative(self.mu)
        self.resid_working = (self.endog - self.mu) * mu_deriv
    
    def _deviance_residuals(self):
        """Calculate deviance residuals"""
        weights = self._get_weights()
        
        # Individual deviance contributions
        dev_contrib = self.family.deviance(
            self.endog, self.mu,
            freq_weights=1.0  # Don't weight here, we'll sign them
        )
        
        # Actually, we need to compute sign and sqrt of deviance
        # This is family-specific, so we'll do a simple version
        sign = np.sign(self.endog - self.mu)
        resid_dev = sign * np.sqrt(
            np.clip(dev_contrib, 0, None)  # Ensure non-negative
        )
        
        return resid_dev
    
    def predict(self, exog=None, offset=None, exposure=None,type='response'):
        if exog is None:
            exog = self.exog
        else:
            exog = np.asarray(exog, dtype=np.float64)
            if exog.ndim == 1:
                exog = exog.reshape(1, -1)
        
        if exog.shape[1] != self.nvars:
            raise ValueError(f"exog must have {self.nvars} columns")
        
        # Linear predictor
        eta = np.dot(exog, self.params)
        
        # Add offset if provided
        if offset is not None:
            offset = np.asarray(offset, dtype=np.float64)
            if offset.shape == (exog.shape[0],):
                eta += offset
            else:
                raise ValueError(f"offset must have shape ({exog.shape[0]},)")
        
        # Handle prediction type
        if type.lower() == "link":
            return eta
        elif type.lower() == "response":
            mu = self.family.link.inverse(eta)
            
            # Apply exposure if provided
            if exposure is not None:
                exposure = np.asarray(exposure, dtype=np.float64)
                if exposure.shape == (exog.shape[0],):
                    if np.any(exposure <= 0):
                        raise ValueError("exposure must be positive")
                    mu = mu * exposure
                else:
                    raise ValueError(f"exposure must have shape ({exog.shape[0]},)")
            
            return mu
        elif type.lower() == "var":
            # Predict variance
            mu = self.family.link.inverse(eta)
            if exposure is not None:
                exposure = np.asarray(exposure, dtype=np.float64)
                mu = mu * exposure
            return self.family.variance(mu) * self.scale
        else:
            raise ValueError(f"Unknown prediction type: {type}")
    
    def summary(self):
        """Return summary of the fitted model"""
        from tabulate import tabulate
        
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append(f"Generalized Linear Model Regression Results")
        summary_lines.append("=" * 80)
        summary_lines.append(f"Model Family:          {type(self.family).__name__}")
        summary_lines.append(f"Link Function:         {self.family.link}")
        summary_lines.append(f"Dependent Variable:    y")
        summary_lines.append(f"No. Observations:      {self.nobs}")
        summary_lines.append(f"Model:                 {self.df_model}")
        summary_lines.append(f"Residual:              {self.df_resid}")
        summary_lines.append(f"Scale:                 {self.scale:.6f}")
        summary_lines.append(f"Method:                IRLS")
        summary_lines.append(f"Converged:             {self.converged}")
        summary_lines.append(f"No. Iterations:        {self.iterations}")
        summary_lines.append("-" * 80)
        
        # Coefficient table
        coef_table = []
        for i in range(self.nvars):
            coef = self.params[i]
            se = self.bse[i] if self.bse is not None else np.nan
            t = coef / se if se > 0 else np.nan
            p = 2 * (1 - _norm_cdf(np.abs(t))) if not np.isnan(t) else np.nan
            
            coef_table.append([
                f"x{i}",
                f"{coef:10.6f}",
                f"{se:10.6f}",
                f"{t:10.4f}" if not np.isnan(t) else "      nan",
                f"{p:10.4f}" if not np.isnan(p) else "      nan",
                f"[{coef - 1.96*se:6.4f}, {coef + 1.96*se:6.4f}]" if se > 0 else "[ nan, nan]"
            ])
        
        summary_lines.append(tabulate(
            coef_table,
            headers=["", "coef", "std err", "z", "P>|z|", "[0.025  0.975]"],
            tablefmt="plain"
        ))
        
        summary_lines.append("-" * 80)
        
        # Model diagnostics
        if self.deviance is not None:
            summary_lines.append(f"Deviance:              {self.deviance:12.4f}")
        if self.null_deviance is not None:
            summary_lines.append(f"Null Deviance:         {self.null_deviance:12.4f}")
        if self.llf is not None:
            summary_lines.append(f"Log-Likelihood:        {self.llf:12.4f}")
        if self.aic is not None:
            summary_lines.append(f"AIC:                   {self.aic:12.4f}")
        if self.bic is not None:
            summary_lines.append(f"BIC:                   {self.bic:12.4f}")
        
        summary_lines.append("=" * 80)
        
        return "\n".join(summary_lines)
    
    def conf_int(self, alpha=0.05):
        if self.bse is None:
            raise ValueError("Model must be fitted first")
        
        z = _norm_ppf(1 - alpha / 2)
        ci_lower = self.params - z * self.bse
        ci_upper = self.params + z * self.bse
        
        return np.column_stack([ci_lower, ci_upper])
    
    def t_test(self, restriction_matrix):
         raise NotImplementedError("t_test not implemented yet")
    
    def f_test(self, restriction_matrix):
        raise NotImplementedError("f_test not implemented yet")
    
    def __repr__(self):
        return f"GLM(family={type(self.family).__name__}, n_obs={self.nobs}, n_vars={self.nvars})"
    
    def __str__(self):
        return f"GLM model with {type(self.family).__name__} family"

# Helper functions for normal distribution
def _norm_cdf(x):
    """Cumulative distribution function of standard normal"""
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def _norm_ppf(p):
    """Percent point function (inverse CDF) of standard normal"""
    # Simple approximation
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")
    
    # Cornish-Fisher expansion
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        num = (((((((2.5090809287301226727e3 * r +
                    3.3430575583588128105e4) * r +
                    6.7265770927008700853e4) * r +
                    4.5921953931549871457e4) * r +
                    1.3731693765509461125e4) * r +
                    1.9715909503065514427e3) * r +
                    1.3314166789178437745e2) * r +
                    3.3871328727963666080e0) * q
        den = (((((((5.2264952788528545610e3 * r +
                    2.8729085735721942674e4) * r +
                    3.9307895800092710610e4) * r +
                    2.1213794301586595867e4) * r +
                    5.3941960214247511077e3) * r +
                    6.8718700749205790830e2) * r +
                    4.2313330701600911252e1) * r +
                    1.0)
        return num / den
    else:
        if q > 0:
            r = 1 - p
        else:
            r = p
        
        r = np.sqrt(-np.log(r))
        if r <= 5:
            r = r - 1.6
            num = (((((((7.74545014278341407640e-4 * r +
                        2.27238449892691845833e-2) * r +
                        2.41780725177450611770e-1) * r +
                        1.27045825245236838258e0) * r +
                        3.64784832476320460504e0) * r +
                        5.76949722146069140550e0) * r +
                        4.63033784615654529590e0) * r +
                        1.42343711074968357734e0)
            den = (((((((1.05075007164441684324e-9 * r +
                        5.47593808499534494600e-4) * r +
                        1.51986665636164571966e-2) * r +
                        1.48103976427480074590e-1) * r +
                        6.89767334985100004550e-1) * r +
                        1.67638483018380384940e0) * r +
                        2.05319162663775882187e0) * r +
                        1.0)
        else:
            r = r - 5
            num = (((((((2.01033439929228813265e-7 * r +
                        2.71155556874348757815e-5) * r +
                        1.24266094738807843860e-3) * r +
                        2.65321895265761230930e-2) * r +
                        2.96560571828504891230e-1) * r +
                        1.78482653991729133580e0) * r +
                        5.46378491116411436990e0) * r +
                        6.65790464350110377720e0)
            den = (((((((2.04426310338993978564e-15 * r +
                        1.42151175831644588870e-7) * r +
                        1.84631831751005468180e-5) * r +
                        7.86869131145613259100e-4) * r +
                        1.48753612908506148525e-2) * r +
                        1.36929880922735805310e-1) * r +
                        5.99832206555887937690e-1) * r +
                        1.0)
        
        if q < 0:
            return -num / den
        else:
            return num / den
        
def glm(endog,exog, family=None,**kwargs):
    """convenience for quick GLM fitting"""
    model= GLM(endog,exog,family,**kwargs)
    return model.fit()

class StableGLM(GLM):
    """Stable version of GLM for complex situations"""
    def calc_covariance(self, cov_type='nonrobust', cov_kwds=None):
        if cov_kwds is None: 
            cov_kwds={}
        #Accessing attributes
        link= self.family.link
        mu_deriv= link.derivative(self.mu)
        var_func= self.family.variance(self.mu)
        weights= self._get_weights()

        #preventing division by zero and clean NANs
        denom=np.clip(var_func*mu_deriv**2,1e-10,None)
        w= np.nan_to_num(weights/denom,nan=.0,posinf=1e8)

        XW= self.exog * w[:,np.newaxis]
        XWX= np.dot(self.exog.T,XW)

        try:
            cov= linalg.inv(XWX) *self.scale
        except linalg.LinAlgError:
            cov= linalg.pinv(XWX)*self.scale
        
        if cov_type.lower() !='nonrobust':
            cov= self._robust_covariance(cov,cov_type=cov_type,cov_kwds=cov_kwds)
        return cov
    
    def fit(self, start_params=None, tol=None, maxiter=None, scale=None, **kwargs):
        # We override fit to add Step-Halving so the results don't explode to 1e147
        if tol is not None: self.tol = tol
        if maxiter is not None: self.maxiter = maxiter
        
        if start_params is None:
            self.params = self._get_start_params()
        else:
            self.params = np.asarray(start_params, dtype=np.float64)

        link = self.family.link
        eta = np.dot(self.exog, self.params) + self.offset
        mu = np.clip(link.inverse(eta), 1e-8, 1e10)
        
        # Initial deviance to track if we are getting better or worse
        current_dev = self.family.deviance(self.endog, mu, freq_weights=self._get_weights())
        #Tracking
        self.iterations=0
        self.fit_history=[]
        self.converged= False

        for i in range(self.maxiter):
            old_params = self.params.copy()
            
            mu_deriv = link.derivative(mu)
            var_func = self.family.variance(mu)
            denom = np.clip(var_func * mu_deriv**2, 1e-10, None)
            w = self._get_weights() / denom
            z = (eta - self.offset) + (self.endog - mu) * mu_deriv
            
            XW = self.exog * w[:, np.newaxis]
            direction = np.linalg.pinv(np.dot(self.exog.T, XW)) @ np.dot(self.exog.T, w * z)
            step = direction - old_params

            # --- STEP HALVING ---
            # This prevents the "Scale: 2.49e+147" error
            alpha = 1.0
            for _ in range(10):
                test_params = old_params + alpha * step
                test_eta = np.dot(self.exog, test_params) + self.offset
                test_mu = np.clip(link.inverse(test_eta), 1e-8, 1e10)
                test_dev = self.family.deviance(self.endog, test_mu, freq_weights=self._get_weights())
                
                if test_dev <= current_dev:
                    self.params, mu, eta, current_dev = test_params, test_mu, test_eta, test_dev
                    break
                alpha *= 0.5
                #Storing info
                self.fit_history.append({
                'iteration':i+1,
                'params': self.params.copy(),
                'change':self.params - old_params
            })
            
            if np.max(np.abs(self.params - old_params)) < self.tol:
                self.converged = True
                break
        
        # Finalizing using the regular diagnostics method
        self.iterations = i + 1
        self.mu = mu
        self.scale = self.estimate_scale(scale)
        self.cov_params = self.calc_covariance(**kwargs)
        self.bse = np.sqrt(np.diag(self.cov_params))
        self._calculate_diagnostics()
        return self
        

        

    

    





