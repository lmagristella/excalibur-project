# integration/integrator.py
import numpy as np

class Integrator:
    """Intègre la géodésique d’un photon dans une métrique donnée."""
    def __init__(self, metric, dt=1e-3):
        self.metric = metric
        self.dt = dt

    def integrate(self, photon, steps):
        state = photon.state.copy()
        for _ in range(steps):
            k1 = self.metric.geodesic_equations(state)
            k2 = self.metric.geodesic_equations(state + 0.5*self.dt*k1)
            k3 = self.metric.geodesic_equations(state + 0.5*self.dt*k2)
            k4 = self.metric.geodesic_equations(state + self.dt*k3)
            state += (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)
            photon.x = state[:4]
            photon.u = state[4:]
            photon.state_quantities(self.metric.metric_physical_quantities)
            photon.record()
