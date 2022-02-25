''' float32 -> float16로 변경하기
    용량은 늘어나지만 정확도는 올라감 '''

'''입력 이미지 사이즈 모두 32->36으로 변결'''
import cv2
import tensorflow as tf
import os
import numpy as np
import glob
import cv2 as cv
import matplotlib.pyplot as plt

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)

# CIFAR-10 데이터를 다운로드 받기 위한 keras의 helper 함수인 load_data 함수를 임포트합니다.
from tensorflow.keras.datasets.cifar10 import load_data


# 다음 배치를 읽어오기 위한 next_batch 유틸리티 함수를 정의합니다.
def next_batch(num, data, labels):
    '''
    `num` 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴합니다.
    '''
    idx = np.arange(0, len(data))
    # print(data.shape[0])
    # idx = np.arange(0, int(data.shape[0]))  # 수정수정
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    # print(np.asarray(data_shuffle).shape)
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)



# 인풋 아웃풋 데이터, 드롭아웃 확률을 입력받기위한 플레이스홀더를 정의합니다.

x = tf.placeholder(tf.float16, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float16, shape=[None, 10])
keep_prob = tf.placeholder(tf.float16)

# CIFAR-10 데이터를 다운로드하고 데이터를 불러옵니다.
(x_train, y_train), (x_test, y_test) = load_data()
print(len(x_train))

# Custom training dataset
# x_train = [cv.imread(file) for file in glob.glob(r"C:/Users/SY/CSB/wild_boar/dataset/all/*.png")]
#
# del x_train[52]
# del x_train[649]

# 살짝 바꿔여 됨 <- custom 이든 뭐든
for i in range(len(x_train)):
    # print(i)
    # x_train[i].astype('uint8')
    if x_train[i] is None:
        # print(f'{i}, check')
        continue
    # 여기는 주석해줘여 함
    # x_train[i] = cv.cvtColor(x_train[i], cv.COLOR_BGR2RGB)

    #통과
    x_train[i] = cv.resize(x_train[i], (32, 32), interpolation=cv.INTER_AREA)
    # print(x_train)
    # x_train[i] = x_train[i] / 255.0


    # x_train[i] = cv.copyMakeBorder(x_train[i].copy(), 2, 2, 2, 2, cv.BORDER_CONSTANT, value=(0, 0, 0))

    # x_train[i] = np.pad(x_train[i], ((2, 2), (2, 2)), 'constant', constant_value = 0)
    # print(x_train[i])
    # cv.imshow('test', x_train[i])
    # cv.waitKey(0)

# print(x_train[0].shape)
# x_train = np.pad(x_train, [[0,0], [2,2], [2,2], [0,0]])/255
# x_test = np.pad(x_test, [[0,0], [2,2], [2,2], [0,0]])/255


# custom 관련
# print(x_train[0].shape)
# y_train = []
# for i in range(0, 95):
#     y_train.append([0])
# for i in range(95, 164):
#     y_train.append([1])
# for i in range(164, 260):
#     y_train.append([2])
# for i in range(260, 356):
#     y_train.append([3])
# for i in range(356, 456):
#     y_train.append([4])
# for i in range(456, 551):
#     y_train.append([5])
# for i in range(551, 619):
#     y_train.append([6])
# for i in range(619, 719):
#     y_train.append([7])
# for i in range(719, 787):
#     y_train.append([8])
# for i in range(787, 874):
#     y_train.append([9])
#
# x_train = np.array(x_train)
# y_train = np.array(y_train)
#

# 오리지널 일때 x_test 어찌해야 하나
x_test = x_train
y_test = y_train
# print(x_test.shape)


# # scalar 형태의 레이블(0~9)을 One-hot Encoding 형태로 변환합니다.
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_train_one_hot = tf.cast(y_train_one_hot, tf.float16)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)
y_test_one_hot = tf.cast(y_test_one_hot, tf.float16)
print("y_test one hot",y_test_one_hot.shape)
print(y_test_one_hot[0])

# Convolutional Neural Networks(CNN) 그래프를 생성합니다.
# CNN 모델을 정의합니다.
# 첫번째 convolutional layer
W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 16], stddev=0.1))
W_conv1 = tf.cast(W_conv1, tf.float16)
print(f"w_conf : {W_conv1.dtype}")
b_conv1 = tf.Variable(tf.constant(0.1, shape=[16]))
b_conv1 = tf.cast(b_conv1, tf.float16)
print(f"b_conv1 : {b_conv1.dtype}")
# b_conv1 = tf.Variable(tf.zeros([16]))
#h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_conv1 = tf.compat.v2.nn.leaky_relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
print(f"h_conv1 : {h_conv1.dtype}")


# 두번째 convolutional layer
W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 16, 16], stddev=0.1))
W_conv2 = tf.cast(W_conv2, tf.float16)
b_conv2 = tf.Variable(tf.constant(0.1, shape=[16]))
b_conv2 = tf.cast(b_conv2, tf.float16)
# b_conv2 = tf.Variable(tf.zeros([16]))
# h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
h_conv2 = tf.compat.v2.nn.leaky_relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

# 첫번째 Pooling layer
h_pool1 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(f"h_pool1 : {h_pool1.dtype}")

# 세번째 convolutional layer - 32개의 특징들(feature)을 64개의 특징들(feature)로 맵핑(maping)합니다.
W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 16, 32], stddev=0.1))
b_conv3 = tf.Variable(tf.constant(0.1, shape=[32]))
W_conv3 = tf.cast(W_conv3, tf.float16)
b_conv3 = tf.cast(b_conv3, tf.float16)
# b_conv3 = tf.Variable(tf.zeros([32]))
# h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
h_conv3 = tf.compat.v2.nn.leaky_relu(tf.nn.conv2d(h_pool1, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

# 두번째 pooling layer.
h_pool2 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Fully Connected Layer 1 (생략가능)
W_fc1 = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 32, 2048], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[2048]))
W_fc1 = tf.cast(W_fc1, tf.float16)
b_fc1 = tf.cast(b_fc1, tf.float16)
h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 32])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1 = tf.compat.v2.nn.leaky_relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
print("type check : ",W_fc1.dtype, b_fc1.dtype, h_fc1.dtype)

# Fully Connected Layer 2
W_fc2 = tf.Variable(tf.truncated_normal(shape=[2048, 1024], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[1024]))
W_fc2 = tf.cast(W_fc2, tf.float16)
b_fc2 = tf.cast(b_fc2, tf.float16)
# b_fc2 = tf.Variable(tf.zeros([1024]))
# h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
h_fc2 = tf.compat.v2.nn.leaky_relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
                # h_pool2_flat = tf.reshape(h_pool2, [-1, 2048])
                # h_fc2 = tf.compat.v2.nn.leaky_relu(tf.matmul(h_pool2_flat, W_fc2) + b_fc2)

# Fully Connected Layer 3
W_fc3 = tf.Variable(tf.truncated_normal(shape=[1024, 512], stddev=0.1))
b_fc3 = tf.Variable(tf.constant(0.1, shape=[512]))
W_fc3 = tf.cast(W_fc3, tf.float16)
b_fc3 = tf.cast(b_fc3, tf.float16)
# b_fc3 = tf.Variable(tf.zeros([512]))
# h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
h_fc3 = tf.compat.v2.nn.leaky_relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

# Fully Connected Layer 4
W_fc4 = tf.Variable(tf.truncated_normal(shape=[512, 10], stddev=0.1))
b_fc4 = tf.Variable(tf.constant(0.1, shape=[10]))
W_fc4 = tf.cast(W_fc4, tf.float16)
b_fc4 = tf.cast(b_fc4, tf.float16)
# b_fc4 = tf.Variable(tf.zeros([10]))
logits = tf.matmul(h_fc3, W_fc4) + b_fc4
y_pred = tf.nn.softmax(logits)
print(f"y_pred : {y_pred.dtype}")
# Cross Entropy를 비용함수(loss function)으로 정의하고, RMSPropOptimizer를 이용해서 비용 함수를 최소화합니다.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
# train_step = tf.train.RMSPropOptimizer(0.00005).minimize(loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)  # 수정수정

# 정확도를 계산하는 연산을 추가합니다.
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float16))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 수정 (형 16 32 64 결정하기)

tmp = 0  # 최댓값 찾아보기   1 : 2225

# 세션을 열어 실제 학습을 진행합니다.
with tf.Session() as sess:
    # 모든 변수들을 초기화한다.
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # 새로 학습을 할때는 아래 코드를 주석처리해야함 (주석)
    saver.restore(sess, r'C:\Users\SY\CSB\wild_boar\model/please\cifar10_float16.ckpt')
    # saver = tf.train.import_meta_graph(r'C:\Users\SY\CSB\wild_boar\model\please\cifar10_float16.ckpt.meta')
    # saver = tf.train.import_meta_graph(r'C:\Users\SY\CSB\wild_boar\model\please\cifar10_float16.ckpt.meta')
    # print(saver)
    # saver.restore(sess, tf.train.latest_checkpoint(r'C:\Users\SY\CSB\wild_boar\model\please'))
    #
    #

    # 10000 Step만큼 최적화를 수행합니다. (훈련)
    # for i in range(50000):
    #     batch = next_batch(32, x_train, y_train_one_hot.eval())
    #
    #     # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력합니다.
    #     if i % 1000 == 0:
    #         train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
    #         loss_print = loss.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
    #         print("반복(Epoch): %d, 트레이닝 데이터 정확도: %f, 손실 함수(loss): %f" % (i, train_accuracy, loss_print))
    #         saver.save(sess, r'C:\Users\SY\CSB\wild_boar\model\please\cifar10_float16.ckpt')
    #     # 20% 확률의 Dropout을 이용해서 학습을 진행합니다.
    #     sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.8})

    # 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력합니다.
    test_accuracy = 0.0
    for i in range(10):
        test_batch = next_batch(32, x_test, y_test_one_hot.eval())
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0})
        print("test" ,test_accuracy/(i+1))


    # 가중치를 레이어별로 txt 파일로 저장

    # W_conv1 출력
    # print(W_conv1.shape, type(W_conv1))
    # W_conv1 = sess.run(W_conv1).reshape(3, 16, 5, 5)
    # f = open("W_conv1.txt", 'w')
    # for i in range(3):
    #     for j in range(16):
    #         for k in range(5):
    #             for p in range(5):
    #                 if (W_conv1[i][j][k][p] < 0):
    #                     # print(str(np.round(W_conv1[i][j][k][p], 6)), end=", ")
    #                     # print(format(W_conv1[i][j][k][p], '.6f'), end=", ")
    #                     f.write(format(W_conv1[i][j][k][p], '.6f') + ", ")
    #                     tmp = max(tmp, W_conv1[i][j][k][p])
    #                 else:
    #                     # print("+" + format(W_conv1[i][j][k][p], '.6f'), end=", ")
    #                     f.write(format(W_conv1[i][j][k][p], '.6f') + ",")
    #                     tmp = max(tmp, W_conv1[i][j][k][p])
    #
    #             # print()
    #             f.write("\n")
    #         # print("\n\n")
    #         f.write("\n\n\n")
    #
    # f.close()
    # #
    # # W_conv2 출력
    # print(W_conv2.shape)
    # W_conv2 = sess.run(W_conv2).reshape(16, 16, 5, 5)
    # f = open("W_conv2.txt", 'w')
    # for i in range(16):
    #     for j in range(16):
    #         for k in range(5):
    #             for p in range(5):
    #                 if (W_conv2[i][j][k][p] < 0):
    #                     # print(format(W_conv2[i][j][k][p], '.6f'), end=", ")
    #                     f.write(format(W_conv2[i][j][k][p], '.6f') + ", ")
    #                 else:
    #                     # print("+" + format(W_conv2[i][j][k][p], '.6f'), end=", ")
    #                     f.write(format(W_conv2[i][j][k][p], '.6f') + ", ")
    #             # print()
    #             f.write("\n")
    #         # print("\n\n")
    #         f.write("\n\n\n")
    # f.close()
    #
    # # W_conv3 출력
    # # print(W_conv3.shape)
    # W_conv3 = sess.run(W_conv3).reshape(16, 32, 3, 3)
    # f = open("W_conv3.txt", 'w')
    # for i in range(16):
    #     for j in range(32):
    #         for k in range(3):
    #             for p in range(3):
    #                 if (W_conv3[i][j][k][p] < 0):
    #                     # print(format(W_conv3[i][j][k][p], '.6f'), end=", ")
    #                     f.write(format(W_conv3[i][j][k][p], '.6f') + ", ")
    #                     tmp = max(tmp, W_conv3[i][j][k][p])
    #                 else:
    #                     # print("+" + format(W_conv3[i][j][k][p], '.6f'), end=", ")
    #                     # f.write("+" + format(W_conv3[i][j][k][p], '.6f') + ", ")
    #                     f.write(format(W_conv3[i][j][k][p], '.6f') + ", ")
    #                     tmp = max(tmp, W_conv3[i][j][k][p])
    #
    #             # print()
    #             f.write("\n")
    #         # print("\n\n")
    #         f.write("\n\n\n")
    # f.close()

    # W_fc2 출력
    # print(W_fc2.shape)
    # W_fc2 = sess.run(W_fc2)
    # f = open("W_fc2.txt", 'w')
    # for i in range(2048):
    #     for j in range(1024):
    #         if (W_fc2[i][j] < 0):
    #             # print(format(W_fc2[i][j], '.6f'), end=",")
    #             f.write(format(W_fc2[i][j], '.6f') + ",")
    #             tmp = max(tmp, W_fc2[i][j])
    #         else:
    #             # print("+" + format(W_fc2[i][j], '.6f'), end=",")
    #             f.write(format(W_fc2[i][j], '.6f') + ",")
    #             tmp = max(tmp, W_fc2[i][j])
    #         if (j+1) % 8 == 0:
    #             # print()
    #             f.write("\n")
    #
    # f.close()

    # #W_fc3 출력
    print(W_fc3.shape)
    W_fc3 = sess.run(W_fc2)
    f = open("W_fc3.txt", 'w')
    for i in range(1024):
        for j in range(512):
            if (W_fc3[i][j] < 0):
                # print(format(W_fc3[i][j], '.6f'), end=",")
                f.write(format(W_fc3[i][j], '.6f') + ",")
                tmp = max(tmp, W_fc3[i][j])
            else:
                # print("+" + format(W_fc3[i][j], '.6f'), end=",")
                f.write(format(W_fc3[i][j], '.6f') + ",")
                tmp = max(tmp, W_fc3[i][j])
            if (j+1) % 8 == 0:
                # print()
                f.write("\n")

    f.close()

    #W_fc4 출력
    print(W_fc4.shape)
    W_fc4 = sess.run(W_fc4)
    f = open("W_fc4.txt", 'w')
    for i in range(1, 513):

        for j in range(1, 11):
            if (W_fc4[i - 1][j - 1] < 0):
                # print(format(W_fc4[i - 1][j - 1], '.6f'), end=",")
                f.write(format(W_fc4[i - 1][j - 1], '.6f') + ",")
                tmp = max(tmp, W_fc4[i - 1][j - 1])
            else:
                # print("+" + format(W_fc4[i - 1][j - 1], '.6f'), end=",")
                f.write(format(W_fc4[i - 1][j - 1], '.6f') + ",")
                tmp = max(tmp, W_fc4[i - 1][j - 1])
            if ((i - 1) * 10 + j) % 8 == 0:
                # print()
                f.write("\n")

    f.close()
