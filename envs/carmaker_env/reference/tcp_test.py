"""
시뮬링크를 별도로 열어 테스트할 경우에 사용하는 TCP/IP 통신을 위한 테스트 서버
"""

import socket
import struct
import time
import numpy as np
import time
import random

TCP_IP = '127.0.0.1'
TCP_PORT = 80
data_num = 6
BUFFER_SIZE = 8 * data_num

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))

s.listen(1)

print("Waiting for connection...", TCP_PORT)
conn, addr = s.accept()
print("Connection from:", addr)
time.sleep(1)

i=0
try:
    while True:
        i+=1
        # 데이터 전송
        a = random.randrange(-1, 1)
        data_to_send = [0, 0.01]
        data = struct.pack('!2d', *data_to_send)

        conn.send(data)
        print("Data sent:", data_to_send)

        # 데이터 수신
        received_data = conn.recv(BUFFER_SIZE)
        if not received_data:
            print("No data received.")
            # break

        try:
            unpacked_data = struct.unpack('!%dd' % data_num, received_data)
            #print("Received data:", unpacked_data[:])
        except:
            print("except: ", received_data)

        #time.sleep(0.2)

finally:
    # 연결 종료
    conn.close()

# a, s' s_t0는 빼고
