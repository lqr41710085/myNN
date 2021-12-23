import dataProcess as dp
import tensorflow as tf
from dnn import Model
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # mnist数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # tf.keras.datasets.cifar10.load_data()
    x_train, y_train = dp.dataProcess(x_train, y_train, oneHot_num=10)
    x_test, y_test = dp.dataProcess(x_test, y_test)

    # 训练模型
    m = Model(layer=[128, 64, 10], loss_func="crossentropy", regularization="L2", o_activation_func='softmax')
    m.fit(train_x=x_train, train_y=y_train, batch_size=64, epochs=15)

    # 用训练的模型进行预测
    y_predict = m.forward(x=x_test)
    results = []
    for i in range(y_predict.shape[0]):
        results.append(np.argmax(y_predict[i]))
    results = np.asarray(results).reshape(len(results), 1)
    np.set_printoptions(threshold=np.inf)
    print("head 10 of y_test:", y_test[:10])
    print("head 10 of results:", results[:10])
    print("准确率:", sum(y_test == results) / len(y_test))

