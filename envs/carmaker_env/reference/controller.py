import scipy
import numpy as np
import matplotlib.pyplot as plt

class LQR:
    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K = self.compute_gain()

    def compute_gain(self):
        P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        K = np.linalg.inv(self.R) @ self.B.T @ P
        return K

    def compute_control(self, x, x_ref):
        u = -self.K @ (x - x_ref)
        return u

if __name__ == "__main__":
    A = np.array([[0, 1], [0, -1]])
    B = np.array([[0], [1]])
    Q = np.eye(2)
    R = np.array([[1]])

    lqr = LQR(A, B, Q, R)

    x = np.array([2, 0])
    x_ref = np.array([0, 0])

    dt = 0.01
    time = np.arange(0, 10, dt)
    x_trajectory = []
    u_trajectory = []

    for t in time:
        u = lqr.compute_control(x, x_ref)
        x = x + (A @ x + B @ u) * dt
        x_trajectory.append(x)
        u_trajectory.append(u)

    x_trajectory = np.array(x_trajectory)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, x_trajectory[:, 0], label='Position')
    plt.plot(time, x_trajectory[:, 1], label='Velocity')
    plt.xlabel('Time [s]')
    plt.ylabel('States')
    plt.title('State Trajectories')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time, u_trajectory, label='Control Input')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Input')
    plt.title('Control Input Trajectory')
    plt.legend()

    plt.tight_layout()
    plt.show()