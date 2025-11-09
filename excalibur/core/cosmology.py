import numpy as np
from scipy.integrate import quad
from excalibur.core.constants import *


class LCDM_Cosmology:
    def __init__(self, H0, Omega_m, Omega_r, Omega_lambda, Omega_k = 0):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_lambda = Omega_lambda
        self.Omega_r = Omega_r
        self.Omega_k = 0


    def E(self, z):
        """Fonction de Hubble normalisée."""
        return np.sqrt(self.Omega_m * (1 + z)**3 + self.Omega_r * (1 + z)**4 + self.Omega_k * (1 + z)**2 + self.Omega_lambda)
    
    def a_of_z(self, z):
        """Calcule le facteur d'échelle a à partir du décalage vers le rouge z."""
        return 1.0 / (1.0 + z)
    
    def a_of_t(self, t):
        """Calcule le facteur d'échelle a à partir du temps cosmologique t (approximation numérique)."""
        def integrand(a):
            return 1.0 / (a * self.E(1.0/a - 1.0))
        
        integral, _ = quad(integrand, 0, 1)
        t0 = integral / self.H0
        
        def time_to_a(a):
            integral, _ = quad(integrand, 0, a)
            return integral / self.H0
        
        # Handle both scalar and array inputs
        t = np.asarray(t)
        scalar_input = t.ndim == 0
        t = np.atleast_1d(t)
        
        a_result = np.zeros_like(t)
        
        for i, t_val in enumerate(t):
            # Recherche numérique de a pour lequel time_to_a(a) = t_val
            a_guess = 1.0
            for _ in range(100):
                t_guess = time_to_a(a_guess)
                if abs(t_guess - t_val) < 1e-6:
                    break
                a_guess *= t_val / t_guess
            a_result[i] = a_guess
        
        return a_result.item() if scalar_input else a_result
    
    def a_of_eta(self, eta):
        """Calcule le facteur d'échelle a à partir du temps conforme η (approximation numérique)."""
        def integrand(a):
            return 1.0 / self.E(1.0/a - 1.0)
        
        integral, _ = quad(integrand, 0, 1)
        eta0 = integral / self.H0
        
        def eta_to_a(a):
            integral, _ = quad(integrand, 0, a)
            return integral / self.H0
        
        # Handle both scalar and array inputs
        eta = np.asarray(eta)
        scalar_input = eta.ndim == 0
        eta = np.atleast_1d(eta)
        
        a_result = np.zeros_like(eta)
        
        for i, eta_val in enumerate(eta):
            # Recherche numérique de a pour lequel eta_to_a(a) = eta_val
            a_guess = 1.0
            for _ in range(100):
                eta_guess = eta_to_a(a_guess)
                if abs(eta_guess - eta_val) < 1e-6:
                    break
                a_guess *= eta_val / eta_guess
            a_result[i] = a_guess
        
        return a_result.item() if scalar_input else a_result

    def comoving_distance(self, z):
        """Calcule la distance comobile jusqu'à un décalage vers le rouge z."""
        integral, _ = quad(lambda zp: 1.0 / self.E(zp), 0, z)
        return (c / self.H0) * integral
    
