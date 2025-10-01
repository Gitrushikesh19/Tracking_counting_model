import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanTracker:
    def __init__(self, dt=1.0):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.set_dt(dt)

        # We measure [cx, cy, w, h]
        self.kf.H = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0]
        ], dtype=float)

        # Covariances (tune as needed)
        self.kf.P *= 1000.0          # large initial uncertainty
        self.kf.R *= 10.0            # measurement noise
        self.kf.Q *= 0.01            # process noise

        # Initial state
        self.kf.x = np.zeros((8, 1))
        self.last_result = None

    def set_dt(self, dt):
        self.dt = float(dt)
        F = np.eye(8)
        F[0,4] = dt; F[1,5] = dt; F[2,6] = dt; F[3,7] = dt
        self.kf.F = F

    @staticmethod
    def xyxy_to_cxcywh(b):
        x1,y1,x2,y2 = b
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = x1 + w * 0.5
        cy = y1 + h * 0.5
        return np.array([cx, cy, w, h], dtype=float)

    @staticmethod
    def cxcywh_to_xyxy(s):
        cx, cy, w, h = s
        x1 = cx - w * 0.5; y1 = cy - h * 0.5
        x2 = cx + w * 0.5; y2 = cy + h * 0.5
        return [int(x1), int(y1), int(x2), int(y2)]

    def update(self, box_xyxy):
        z = self.xyxy_to_cxcywh(box_xyxy).reshape(4,1)
        self.kf.update(z)
        self.last_result = self.get_state()

    def predict(self):
        self.kf.predict()
        self.last_result = self.get_state()
        return self.last_result

    def get_state(self):
        cx, cy, w, h = self.kf.x[:4].reshape(-1)
        return self.cxcywh_to_xyxy([cx, cy, w, h])
