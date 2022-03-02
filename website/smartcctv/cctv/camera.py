import cv2
import numpy as np
import socket
import struct  # 바이트(bytes) 형식의 데이터 처리 모듈
import pickle  # 바이트(bytes) 형식의 데이터 변환 모듈
import random
from threading import Thread
import threading

# ip = '192.168.0.223'
ip = '127.0.0.1'
port = 50002

def yolo(frame, size, score_threshold, nms_threshold):
    # YOLO 네트워크 불러오기
    net = cv2.dnn.readNet("data/yolov4.weights", "data/yolov4.cfg")
    layer_names = net.getLayerNames()
    #print(layer_names)
    # for i in net.getUnconnectedOutLayers() :
    # yolo 관련 예시 파일들은 i[0]으로 하지만 직접 찍어본 봐 i로 하는 게 맞음
    # https://www.codespeedy.com/yolo-object-detection-from-image-with-opencv-and-python/

    # ㅛ
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()] # i[0]
    #print(output_layers)
    # 클래스의 갯수만큼 랜덤 RGB 배열을 생성
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # 이미지의 높이, 너비, 채널 받아오기
    height, width, channels = frame.shape

    # 네트워크에 넣기 위한 전처리
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (size, size), (0, 0, 0), True, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(blob)

    # 결과 받아오기
    outs = net.forward(output_layers)

    # 각각의 데이터를 저장할 빈 리스트
    class_ids = []
    confidences = []
    boxes = []
    # print(outs)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.1:
                # 탐지된 객체의 너비, 높이 및 중앙 좌표값 찾기
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 객체의 사각형 테두리 중 좌상단 좌표값 찾기
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 후보 박스(x, y, width, height)와 confidence(상자가 물체일 확률) 출력
    #print(f"boxes: {boxes}")
    #print(f"confidences: {confidences}")

    # Non Maximum Suppression (겹쳐있는 박스 중 confidence 가 가장 높은 박스를 선택)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=score_threshold, nms_threshold=nms_threshold)

    # 후보 박스 중 선택된 박스의 인덱스 출력
    #print(f"indexes: ", end='')
    #for index in indexes:
    #    print(index, end=' ')
    print("\n\n============================== classes ==============================")

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            class_name = classes[class_ids[i]]
            label = f"{class_name} {confidences[i]:.2f}"
            color = colors[class_ids[i]]

            # 사각형 테두리 그리기 및 텍스트 쓰기
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x - 1, y), (x + len(class_name) * 13 + 65, y - 25), color, -1)
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)

            # 탐지된 객체의 정보 출력
            #print(f"[{class_name}({i})] conf: {confidences[i]} / x: {x} / y: {y} / width: {w} / height: {h}")

    return frame

# 클래스 리스트
classes = ["person", "bicycle", "car", "motorcycle",
           "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
           "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
           "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
           "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
           "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
           "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
           "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
           "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# 입력 사이즈 리스트 (Yolo 에서 사용되는 네크워크 입력 이미지 사이즈)
size_list = [320, 416, 608]
# 320×320 it’s small so less accuracy but better speed
# 609×609 it’s bigger so high accuracy and slow speed
# 416×416 it’s in the middle and you get a bit of both.


class VideoCamera(object):
    def __init__(self):
        self.threads = []


    def __del__(self):
        cv2.destroyAllWindows()

    def run_server(self):
        with socket.socket() as sock:
            sock.bind((ip, port))
            while True:
                # 소켓 스레드 통신
                sock.listen(5)
                conn, addr = sock.accept()
                f = Frame(conn)
                self.threads.append(f)
            sock.close()
            print('server shutdown')

class Frame:
    def __init__(self, client_socket):
        self.client_socket = client_socket
        self.data_buffer = b""
        self.data_size = struct.calcsize("L")

    def get_frame(self):
        # 설정한 데이터의 크기보다 버퍼에 저장된 데이터의 크기가 작은 경우
        while len(self.data_buffer) <self.data_size:
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

        frame = yolo(frame=frame, size=size_list[0], score_threshold=0.4, nms_threshold=0.4)

        frame_flip = cv2.flip(frame, 1)
        ret, frame = cv2.imencode('.jpg', frame_flip)
        return frame.tobytes()

'''
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