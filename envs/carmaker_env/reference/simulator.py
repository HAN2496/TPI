import numpy as np
from envs.carmaker_env.reference.system import System
from envs.carmaker_env.reference.controller import LQR

class Simulator:
    def __init__(self, system, lqr, x_ref):
        self.system = system
        self.lqr = lqr
        self.x_ref = x_ref

    def run_simulation(self, max_simulations=1):
        for _ in range(max_simulations):
            self.system.start()
            while True:
                state = self.system.get_state()
                if state is None:
                    break

                state = np.array(state)[:2]
                action = self.lqr.compute_control(np.array(state), self.x_ref)

                action = np.array([0, action[0]])
                self.system.send_action(action)
                
                # print(f"State: {state}, Action: {action}")

if __name__ == "__main__":
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = np.array([[1.0]])
    x_ref = np.array([0.0, 0.0])

    sim_rcv_num = 6
    sim_send_num = 2
    system = System(sim_rcv_num, sim_send_num, port=80)
    lqr = LQR(A, B, Q, R)
    
    simulator = Simulator(system, lqr, x_ref)
    simulator.run_simulation(max_simulations=2)