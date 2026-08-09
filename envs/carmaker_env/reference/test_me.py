import matlab.engine

MATLAB_PATH='C:/carmaker_matlab/Sydney/src_cm4sl'
SIMUL_PATH='pythonCtrl'
port=10003
# MATLAB 엔진 시작 및 Simulink 모델 열기
eng = matlab.engine.start_matlab()
eng.addpath(MATLAB_PATH)
eng.cd(MATLAB_PATH)
model = eng.load_system(SIMUL_PATH)  # Simulink 모델 이름을 넣어주세요
carmaker_gui_path = SIMUL_PATH+'/Open CarMaker GUI'
eng.open_system(carmaker_gui_path,nargout=0)
eng.eval("cmguicmd('LoadTestRun \"straight\"')", nargout=0)
eng.set_param('{}/CarMaker/tcpiprcv'.format(SIMUL_PATH), 'Port', str(port), nargout=0)
eng.set_param('{}/CarMaker/tcpipsend'.format(SIMUL_PATH), 'Port', str(port), nargout=0)
import socket
host = '127.0.0.1'

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)
print("Waiting for Simulink connection...")

eng.set_param(model, 'SimulationCommand', 'start', nargout=0)
conn, addr = server_socket.accept()
print("Connected to Simulink:", addr)

import struct  # 데이터를 패킹하고 언패킹하는 데 사용

# 데이터 송수신 루프 시작
try:
    while True:
        # [0, 0] 데이터 전송 (예시로 사용)
        send_data = [0, 0]
        # float 형식으로 두 개의 값을 보내기 위해 패킹
        conn.send(struct.pack('!2d', *send_data))
        
        # Simulink로부터 6개의 float 데이터를 수신
        data = conn.recv(6 * 8)  # 6 * 4 bytes = 24 bytes (float는 4바이트)
        if not data:
            print("No data received.")
            break  # 데이터가 없으면 연결 종료
        
        # 수신된 데이터를 float 형식으로 언패킹
        unpacked_data = struct.unpack('!%dd' % 6, data)
        print("Received data from Simulink:", data)
        
        # 원하는 추가 로직 수행 가능
        # 예: 받은 데이터에 따라 새로운 action 생성, 로깅, 상태 업데이트 등

except Exception as e:
    print("An error occurred:", e)

finally:
    # 연결 종료
    conn.close()
    server_socket.close()
    eng.quit()
    print("Connection closed.")