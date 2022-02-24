import cv2
import numpy as np
import socket
import struct  # 바이트(bytes) 형식의 데이터 처리 모듈
import pickle  # 바이트(bytes) 형식의 데이터 변환 모듈

ip = '127.0.0.1'
port = 50002

class VideoCamera(object):
    def __init__(self):
        # 소켓 객체 생성
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 소켓 주소 정보 할당
        self.server_socket.bind((ip, port))
        # 연결 리스닝(동시 접속) 수 설정
        self.server_socket.listen(10)
        print('클라이언트 연결 대기')
        # 연결 수락(클라이언트 (소켓, 주소 정보) 반환)
        self.client_socket, self.address = self.server_socket.accept()
        print('클라이언트 ip 주소 :', self.address[0])
        # 수신한 데이터를 넣을 버퍼(바이트 객체)
        self.data_buffer = b""
        # calcsize : 데이터의 크기(byte)
        # - L : 부호없는 긴 정수(unsigned long) 4 bytes
        self.data_size = struct.calcsize("L")

    def __del__(self):
        # 소켓 닫기
        self.client_socket.close()
        self.server_socket.close()
        print('연결 종료')
        # 모든 창 닫기
        cv2.destroyAllWindows()

    def get_frame(self):
        # 설정한 데이터의 크기보다 버퍼에 저장된 데이터의 크기가 작은 경우
        while len(self.data_buffer) < self.data_size:
            # 데이터 수신
            self.data_buffer += self.client_socket.recv(4096)

        # 버퍼의 저장된 데이터 분할
        packed_data_size = self.data_buffer[:self.data_size]
        self.data_buffer = self.data_buffer[self.data_size:]

        # struct.unpack : 변환된 바이트 객체를 원래의 데이터로 변환
        # - > : 빅 엔디안(big endian)
        #   - 엔디안(endian) : 컴퓨터의 메모리와 같은 1차원의 공간에 여러 개의 연속된 대상을 배열하는 방법
        #   - 빅 엔디안(big endian) : 최상위 바이트부터 차례대로 저장
        # - L : 부호없는 긴 정수(unsigned long) 4 bytes
        frame_size = struct.unpack(">L", packed_data_size)[0]

        # 프레임 데이터의 크기보다 버퍼에 저장된 데이터의 크기가 작은 경우
        while len(self.data_buffer) < frame_size:
            # 데이터 수신
            self.data_buffer += self.client_socket.recv(4096)

        # 프레임 데이터 분할
        frame_data = self.data_buffer[:frame_size]
        self.data_buffer = self.data_buffer[frame_size:]

        print("수신 프레임 크기 : {} bytes".format(frame_size))

        # loads : 직렬화된 데이터를 역직렬화
        # - 역직렬화(de-serialization) : 직렬화된 파일이나 바이트 객체를 원래의 데이터로 복원하는 것
        frame = pickle.loads(frame_data)

        # imdecode : 이미지(프레임) 디코딩
        # 1) 인코딩된 이미지 배열
        # 2) 이미지 파일을 읽을 때의 옵션
        #    - IMREAD_COLOR : 이미지를 COLOR로 읽음
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        frame_flip = cv2.flip(frame, 1)
        ret, frame = cv2.imencode('.jpg', frame_flip)
        return frame.tobytes()




'''

import cv2
import numpy as np
import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 9505

class VideoCamera(object):
    def __init__(self):
        #self.cap = cv2.VideoCapture(0)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))

        self.s = b''
    def __del__(self):
        pass
        #self.cap.release()
    def get_frame(self):
        data, addr = sock.recvfrom(46080)
        s += data

        if len(s) == (46080 * 20):
            frame = numpy.fromstring(s, dtype=numpy.uint8)
            frame = frame.reshape(480, 640, 3)
            cv2.imshow('test', frame)
            return frame.tobytes()
            
        ret, frame = self.cap.read()
        cv2.putText(frame, "test", (0, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0))
        frame_flip = cv2.flip(frame, 1)
        ret, frame = cv2.imencode('.jpg', frame_flip)
        return frame.tobytes()
        
'''