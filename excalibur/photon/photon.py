# photon/photon.py
import numpy as np
from .photon_history import PhotonHistory

class Photon:
    """Objet représentant l’état instantané d’un photon."""
    def __init__(self, position, direction, weight=1.0):
        self.x = np.asarray(position, dtype=float)
        self.u = np.asarray(direction, dtype=float)
        self.quantities = np.array([])
        self.weight = weight
        self.history = PhotonHistory()

    def null_condition(self, metric):
        g = metric.metric_tensor(self.x)
        norm = np.dot(self.u, np.dot(g, self.u))
        return norm

    def state_quantities(self,relevant_quantities):
        self.quantities = relevant_quantities(self.x)

    @property
    def state(self):
        return np.concatenate([self.x, self.u, self.quantities])

    def record(self):
        self.history.append(self.state)
