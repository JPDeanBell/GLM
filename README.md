# Custom Generalized Linear Models (GLM) Implementation

## Overview
Built from-scratch implementation of Generalized Linear Models in Python with robust numerical stability, supporting multiple distributions (Poisson, Gaussian, Binomial) and link functions. Features include IRLS optimization, covariance estimation, model diagnostics, and comprehensive validation against statsmodels.

## Key Features
✅ **Full GLM Framework**: IRLS algorithm with convergence tracking and iteration history  
✅ **Multiple Families**: Poisson, Gaussian, Binomial with appropriate link functions  
✅ **Robust Numerical Stability**: Ridge regularization, overflow protection, NaN/Inf handling  
✅ **Comprehensive Diagnostics**: AIC, BIC, deviance, Pearson residuals, confidence intervals  
✅ **Validation Suite**: Direct comparison with statsmodels benchmark (coefficients within 0.001)  
✅ **Epidemiological Application**: Diabetes mortality analysis with offset-adjusted rates  

## Technical Highlights
• **IRLS Implementation**: Custom weighted least squares with QR decomposition fallback  
• **Numerical Safeguards**: Automatic eta clipping, denominator bounding, ridge regularization  
• **Matrix Operations**: Efficient NumPy implementation with pseudo-inverse fallbacks  
• **Statistical Testing**: Wald tests, likelihood ratio tests, model comparison  

## Case Study: Diabetes Mortality Analysis
Replicated published epidemiological analysis (Table 6.3) using offset-adjusted Poisson regression:
• **Finding**: 85+ age group has 74× higher mortality risk than 45-54 reference (p<0.0001)  
• **Gender Effect**: Females have 41% lower mortality than males after age adjustment  
• **Validation**: Coefficients matched published results within 0.001 accuracy  

## Files
• `glm.py` - Core GLM implementation with IRLS fitting  
• `families.py` - Distribution families (Poisson, Gaussian, Binomial)  
• `links.py` - Link function implementations (log, identity, logit)  
• `PoissonRegression.ipynb` - Complete case study with visualization and diagnostics  
• `tests/` - Unit tests and validation against statsmodels  
