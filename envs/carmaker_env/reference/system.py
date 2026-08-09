import matlab.engine
import socket
import struct

"""
Python - MATLAB 간의 TCP/IP 통신을 위한 시스템 클래스
"""

class System:
    def __init__(self, simul_rcv_num, simul_send_num, host='127.0.0.1', port=10000,
                 matlab_path='C:/carmaker_matlab/Sydney/src_cm4sl', simul_path='pythonCtrl'):
        self.host = host
        self.port = port
        self.matlab_path = matlab_path
        self.simul_path = simul_path
        self.simul_rcv_num = simul_rcv_num
        self.simul_send_num = simul_send_num
        self._setup_sim()

    def __del__(self):
        self.kill()

    def kill(self):
        self.conn.close()
        self.server_socket.close()
        self.eng.quit()
        print("System closed.")

    def _setup_sim(self):
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.matlab_path)
        self.eng.cd(self.matlab_path)
        self.model = self.eng.load_system(self.simul_path)

        carmaker_gui_path = f"{self.simul_path}/Open CarMaker GUI"
        self.eng.open_system(carmaker_gui_path, nargout=0)
        self.eng.eval("cmguicmd('LoadTestRun \"straight\"')", nargout=0)
        
        self.eng.set_param(f'{self.simul_path}/CarMaker/tcpiprcv', 'Port', str(self.port), nargout=0)
        self.eng.set_param(f'{self.simul_path}/CarMaker/tcpipsend', 'Port', str(self.port), nargout=0)
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

    def start(self):
        self.initiate = False
        self.eng.set_param(self.model, 'SimulationCommand', 'start', nargout=0)
        self.conn, addr = self.server_socket.accept()
        print("Connected to Simulink:", addr)

    def stop(self):
        self.eng.set_param(self.model, 'SimulationCommand', 'stop', nargout=0)
        print("Simulation stopped")

    def get_state(self):
        if not self.initiate:
            self.send_action([0] * self.simul_send_num)
            self.initiate = True
        data = self.conn.recv(self.simul_rcv_num * 8)
        if data:
            state = struct.unpack(f'!{self.simul_rcv_num}d', data)
            return state
        else:
            print("No data received. Stopping simulation.")
            return self.stop()

    def send_action(self, action):
        if len(action) != self.simul_send_num:
            raise ValueError(f"Action must have {self.simul_send_num} elements.")
        packed_data = struct.pack(f'!{self.simul_send_num}d', *action)
        self.conn.sendall(packed_data)

if __name__ == "__main__":
    sim_rcv_num = 6
    simul_send_num = 2

    system = System(sim_rcv_num, simul_send_num, port=80)
    
    try:
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            system.start()
            
            while True:
                action = [0.0, 0.0]
                system.send_action(action)
                state = system.get_state()
                
                if state is None:
                    retry_count += 1
                    print(f"Retry {retry_count} of {max_retries}")
                    break

            if retry_count >= max_retries:
                print("Maximum retries reached. Exiting.")
                break

    except Exception as e:
        print("An error occurred:", e)
    
    finally:
        system.kill()
