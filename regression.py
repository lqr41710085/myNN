from dnn import Model
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)


def data_produce(a, b, c, d, n=20000, tr_rate=0.8):
    x = np.random.random_sample([n])  # [0,1)
    x = 2*np.pi*x - np.pi  # [-pi, pi)
    y = a*np.cos(b*x)+c*np.sin(d*x)
    m = n * tr_rate
    x_train = x[:m]
    x_train = np.expand_dims(x_train, axis=1)
    x_test = x[m:]
    x_test = np.expand_dims(x_test, axis=1)
    y_train = y[:m]
    y_train = np.expand_dims(y_train, axis=1)
    y_test = y[m:]
    y_test = np.expand_dims(y_test, axis=1)
    print("x_train shape: ", x_train.shape)
    print("y_train shape: ", x_train.shape)
    print("x_test shape: ", x_test.shape)
    print("y_test shape: ", x_test.shape)
    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':

    n = 10000
    tr_rate = 0.7
    (x_train, y_train), (x_test, y_test) = data_produce(1, 2, 3, 4, n, tr_rate=tr_rate)
    # 训练模型
    m = Model(layer=[64, 32, 16, 1], loss_func="mse", regularization="L1", activation_func='sigmoid')
    m.fit(train_x=x_train, train_y=y_train, batch_size=32, epochs=200)
    # 用训练的模型进行预测
    y_predict = m.forward(x=x_test)
    sst = sum(np.power(y_test-np.mean(y_test), 2))
    sse = sum(np.power(y_test-y_predict, 2))
    print("R2 拟合优度:", 1-sse/sst)
    te = n * (1-tr_rate)
    x_test.resize((te,))
    y_test.resize((te,))
    y_predict.resize((te,))
    sorted_ind = np.argsort(x_test)
    x_test = x_test[sorted_ind]
    y_test = y_test[sorted_ind]
    y_predict = y_predict[sorted_ind]
    plt.plot(x_test, y_test)
    plt.plot(x_test, y_predict, c='red')
    plt.show()
