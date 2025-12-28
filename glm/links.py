import numpy as np
from scipy.special import erf,erfinv
from abc import ABC, abstractmethod

class Link(ABC):
    @abstractmethod
    def link(self,mu):
        """ g(u)=> X'B """
        pass

    @abstractmethod
    def inverse(self,eta):
        """ g^(-1)(X'B)=> u """
        pass 

    @abstractmethod
    def derivative(self,mu):
        "derivative of the link"
        pass

    def inverse_derivative(self,eta):
        mu=self.inverse(eta)
        return 1.0/self.derivative(mu)
    
    def __repr__(self):
        return self.__class__.__name__

class IdentityLink(Link):

    def link(self,mu):
        return mu
    
    def inverse(self,eta):
        return eta
    
    def derivative(self, mu):
        return np.ones_like(mu)
    

class LogLink(Link):
  
    def link(self,mu):
        mu_safe= np.maximum(mu,1e-15)
        return np.log(mu)
    
    def inverse(self,eta):
        eta_clipped=np.clip(eta,-700,700)
        return np.exp(eta_clipped)
    
    def derivative(self, mu):
        mu_safe=np.maximum(mu,1e-10)
        return 1.0/mu_safe
    
            
class LogitLink(Link):

    def link(self,mu):
        if np.any((mu<=0) | (mu>=1)):
            raise ValueError("mu must be strictly between 0 and 1")
        return np.log(mu/(1.0-mu))
    
    def inverse(self,eta):
         return 1.0 / (1.0 + np.exp(-eta))
    
    def derivative(self, mu):
        if np.any((mu<=0) | (mu>=1)):
            raise ValueError("M(u) must be strictly between 0 and 1")
        return 1.0/(mu*(1.0-mu))


class PowerLink:
    
    def __init__(self,power=1.0):
        self.power=power

    def link(self,mu):
        """g(u)=u^p or log(u) if p==0"""
        if self.power==0:
            link_=LogLink()
            return link_.link()
        elif self.power==1:
            """Gaussian if p==1"""
            link_=IdentityLink()
            return link_.inverse()
        else:
            return np.power(mu,self.power)
    
    def inverse(self,eta):
        if self.power==0:
            link_=LogLink()
            return link_.inverse()
        elif self.power==1:
            """Gaussian if p==1"""
            link_=IdentityLink()
            return link_.inverse()
        else:
            return np.power(eta,1/self.power)
    
    def derivative(self,mu):
        if self.power==0:
            link_=LogLink()
            return link_.derivative()
        elif self.power==1:
            """Gaussian if p==1"""
            link_=IdentityLink()
            return link_.derivative()
        else:
            return self.power*np.power(mu,self.power-1)
    

class ProbitLink(Link):

    def __init__(self,eps=1e-15):
        self.eps=eps
    
    def link(self,mu):
        mu= np.clip(mu,self.eps,1-self.eps)
        return np.sqrt(2)* erfinv(2 * mu - 1)
    
    def inverse(self,eta):
        return .5*(1+erf(eta/np.sqrt(2)))
    
    def derivative(self, mu):
        mu= np.clip(mu,self.eps,1-self.eps)
        eta= self.link(mu)
        phi=np.exp(-eta**2/2)/np.sort(2*np.pi)
        return 1.0/phi
    
    
        
        


    
