import cv2
import numpy as np
import math
# from data.proto_file_path import *
# from data.weights_file_path import *
# from data.BODY_PARTS import *
# from data.POSE_PAIRS import *
# from data.yolo_classes import classes

def yolo(frame, size, score_threshold, nms_threshold):
    # YOLO 네트워크 불러오기
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    # print(net.getUnconnectedOutLayers())  -> 출력해서 리스트 확인해보고
      ## ** cuda 버전에 따라서 layer_names[i[0] - 1] 써야할 수도 있음
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

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
    # print(f"boxes: {boxes}")
    # print(f"confidences: {confidences}")

    # Non Maximum Suppression (겹쳐있는 박스 중 confidence 가 가장 높은 박스를 선택)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=score_threshold, nms_threshold=nms_threshold)

    # 후보 박스 중 선택된 박스의 인덱스 출력
    print(f"indexes: ", end='')
    for index in indexes:
        print(index, end=' ')
    print("\n\n============================== classes ==============================")

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            try:
                class_name = classes[class_ids[i]]

            except:
                break

            label = f"{class_name} {confidences[i]:.2f}"
            color = colors[class_ids[i]]

            # 사각형 테두리 그리기 및 텍스트 쓰기
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x - 1, y), (x + len(class_name) * 13 + 65, y - 25), color, -1)
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)

            # 탐지된 객체의 정보 출력
            print(f"[{class_name}({i})] conf: {confidences[i]} / x: {x} / y: {y} / width: {w} / height: {h}")

    return frame

def output_keypoints(frame, proto_file, weights_file, threshold, model_name, BODY_PARTS):
    global points

    # 이미지 읽어오기
    # frame = cv2.imread(image_path)

    # 네트워크 불러오기
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
    layer_names = net.getUnconnectedOutLayers()
    print("layer_names :", layer_names)
    # 입력 이미지의 사이즈 정의
    image_height = 368
    image_width = 368

    # 네트워크에 넣기 위한 전처리
    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False )
    print("input_blob :", type(input_blob))
    print(len(input_blob))

    # 전처리된 blob 네트워크에 입력
    net.setInput(input_blob)
    print(type(net))

    # 결과 받아오기
    out = net.forward()
    # The output is a 4D matrix :
    # The first dimension being the image ID ( in case you pass more than one image to the network ).
    # The second dimension indicates the index of a keypoint.
    # The model produces Confidence Maps and Part Affinity maps which are all concatenated.
    # For COCO model it consists of 57 parts – 18 keypoint confidence Maps + 1 background + 19*2 Part Affinity Maps. Similarly, for MPI, it produces 44 points.
    # We will be using only the first few points which correspond to Keypoints.
    # The third dimension is the height of the output map.
    out_height = out.shape[2]
    # The fourth dimension is the width of the output map.
    out_width = out.shape[3]

    # 원본 이미지의 높이, 너비를 받아오기
    frame_height, frame_width = frame.shape[:2]

    # 포인트 리스트 초기화
    points = []

    print(f"\n========== {model_name} ==========")
    for i in range(len(BODY_PARTS)):

        # 신체 부위의 confidence map
        prob_map = out[0, i, :, :]

        # 최소값, 최대값, 최소값 위치, 최대값 위치
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        # 원본 이미지에 맞게 포인트 위치 조정
        x = (frame_width * point[0]) / out_width
        x = int(x)
        y = (frame_height * point[1]) / out_height
        y = int(y)

        if prob > threshold:  # [pointed]
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)

            points.append((x, y))
            # print(f"[pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

        else:  # [not pointed]
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)

            points.append(None)
            # print(f"[not pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

    return frame

def calculate_degree(point_1, point_2, frame):
    # 역탄젠트 구하기
    global string
    dx = point_2[0] - point_1[0]
    dy = point_2[1] - point_1[1]
    rad = math.atan2(abs(dy), abs(dx))

    # radian 을 degree 로 변환
    deg = rad * 180 / math.pi

    # degree 가 45'보다 작으면 허리가 숙여졌다고 판단
    if deg < 45:
        string = "Bend Down"
        cv2.putText(frame, string, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255))
        print(f"[degree] {deg} ({string})")
    else:
        string = "Stand"
        cv2.putText(frame, string, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255))
        print(f"[degree] {deg} ({string})")

def output_keypoints_with_lines(POSE_PAIRS, frame):
    # 프레임 복사
    frame_line = frame.copy()

    # Neck 과 MidHeap 의 좌표값이 존재한다면
    if (points[1] is not None) and (points[8] is not None):
        calculate_degree(point_1=points[1], point_2=points[8], frame=frame_line)

    for pair in POSE_PAIRS:
        part_a = pair[0]  # 0 (Head)
        part_b = pair[1]  # 1 (Neck)
        if points[part_a] and points[part_b]:
            # print(f"[linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")
            # Neck 과 MidHip 이라면 분홍색 선
            if part_a == 1 and part_b == 8:
                cv2.line(frame, points[part_a], points[part_b], (255, 0, 255), 3)
            else:  # 노란색 선
                cv2.line(frame, points[part_a], points[part_b], (0, 255, 0), 3)
        # else:
        #     print(f"[not linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")

    # 포인팅 되어있는 프레임과 라인까지 연결된 프레임을 가로로 연결
    # frame_horizontal = cv2.hconcat([frame, frame_line])
    # cv2.imshow("Output_Keypoints_With_Lines", frame_horizontal)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return frame


classes = ["person", "bicycle", "car"," motorcycle",
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
           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "wheelchair"]

classes = ["person", "bed", "chair"]
size_list = [320, 416, 608]  # yolo 버전


###################################

## open pose test
BODY_PARTS_BODY_25 = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                      5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
                      10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
                      15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
                      20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"}

POSE_PAIRS_BODY_25 = [[0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [8, 9], [8, 12], [9, 10], [12, 13], [2, 3],
                      [3, 4], [5, 6], [6, 7], [10, 11], [13, 14], [15, 17], [16, 18], [14, 21], [19, 21], [20, 21],
                      [11, 24], [22, 24], [23, 24]]

# 신경 네트워크의 구조를 지정하는 prototxt 파일 (다양한 계층이 배열되는 방법 등)
protoFile_body_25 = "pose_deploy.prototxt"

# 훈련된 모델의 weight 를 저장하는 caffemodel 파일
weightsFile_body_25 = "pose_iter_584000.caffemodel"

path_list = []
walk_1 = "man_walk_1.jpg"
walk_2 = "man_walk_2.jpg"
stand = "man_stand.jpg"
sit = "man_sit.jpg"
down = "man_down.jpg"
path_list.extend([walk_1, walk_2, stand, sit, down])

# 이미지로 테스트
## 키포인트를 저장할 빈 리스트
# points = []
#
# for path in path_list:
#     img = cv2.imread(path)
#     frame = yolo(frame=img, size=size_list[1], score_threshold=0.4, nms_threshold=0.4)
#     frame_man = output_keypoints(frame=frame, proto_file=protoFile_body_25, weights_file=weightsFile_body_25,
#                                  threshold=0.1, model_name=path.split("\\")[-1], BODY_PARTS=BODY_PARTS_BODY_25)
#     frame_man_pose = output_keypoints_with_lines(POSE_PAIRS=POSE_PAIRS_BODY_25, frame=frame_man)
#     cv2.imshow(path.split("\\")[-1] + string, frame_man_pose)
#     print(string)
# cv2.waitKey()


# # # 웹캠으로 실시간 테스트
cap = cv2.VideoCapture(0)
points = []
while(True):
    ret, frame = cap.read()    # Read 결과와 frame

    if(ret) :
        frame = yolo(frame=frame, size=size_list[1], score_threshold=0.4, nms_threshold=0.4)
        frame_pose = output_keypoints(frame, proto_file=protoFile_body_25, weights_file=weightsFile_body_25,
                                 threshold=0.1, model_name="model", BODY_PARTS=BODY_PARTS_BODY_25)
        frame_pose_line = output_keypoints_with_lines(POSE_PAIRS=POSE_PAIRS_BODY_25, frame=frame_pose)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.imshow('test', frame_pose_line)

cap.release()
cv2.destroyAllWindows()


