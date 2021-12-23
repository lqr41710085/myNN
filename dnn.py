import numpy as np

import time


class Model:

    def __init__(self, lr=0.08, layer=[3, 1], dropout_rate=None, activation_func="relu", decay_rate=0.2,
                 o_activation_func=None, loss_func="crossEntropy", regularization="L2", regularization_rate=0.0004,
                 optimizer="Adam", isInitializeWeightMatrix=True):

        self.layers_num = len(layer)
        self.dropout_rate = dropout_rate if dropout_rate else [0] * self.layers_num
        self.lr = lr
        self.decay_rate = decay_rate
        self.layerStructure = layer
        self.activation_func = activation_func
        self.o_activation_func = o_activation_func
        self.loss_func = loss_func
        self.regularization = regularization
        self.regularization_rate = regularization_rate
        self.optimizer = optimizer
        self.isInitializeWeightMatrix = isInitializeWeightMatrix

        self.w = []
        self.loss = []
        self.b = []
        self.z = []  # z = x * w.T
        self.a = []  # a = g(z)
        self.dg = []  # dg = da/dz
        self.dz = []  # dz = dL/dz = dL/da * da/dz
        self.dw = []  # dw = dL/dw = dL/dz * dz/dw
        self.db = []  # db = dL/db = dL/dz * dz/db = dL/dz
        self.m_w, self.m_b = [], []  # 一阶动量mt
        self.V_w, self.V_b = [], []  # 二阶动量Vt

        # 超参数
        self.alpha = 0.00000001
        self.beta = 0.9
        self.beta1 = 0.999
        nd = np.random.RandomState(2021)
        for n in range(len(layer)):
            if n + 1 < len(layer):
                self.w.append(nd.random([layer[n], layer[n + 1]]))
            self.b.append(nd.rand(1, layer[n]))

    def __activation_f(self, activation, z, dropout=0):
        a = z
        if activation == "relu":
            a[a < 0] = 0
            dg = a.copy()
            dg[dg > 0] = 1
        elif activation == "sigmoid":
            a = 1 / (1 + np.exp(-a))
            dg = a * (1 - a)
        elif activation == "tanh":
            m = np.exp(a)
            n = np.exp(-1 * a)
            a = (m - n) / (m + n)
            dg = 1 - a * a
        elif activation == "softmax":
            a = np.exp(a)
            b = np.sum(a, axis=1, keepdims=True)
            b[b == 0] = self.alpha
            a = a / b
            dg = a * (1 - a)
        else:
            dg = 1 * z.shape
        d = np.random.rand(z.shape[0], z.shape[1]) >= dropout
        a = a * d / (1 - dropout)
        dg = dg * d / (1 - dropout)
        return {"a": a, "dg": dg}

    def __loss_f(self, y_hat, y_true):
        y_hat[y_hat == 0] = self.alpha  # 避免出现log0
        y_hat[y_hat == 1] = (1 + self.alpha)
        if self.loss_func.lower() == "crossentropy":  # 搭配L2
            loss = -1 * np.sum(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat)) / y_hat.shape[0]
            dz_hat = y_hat - y_true
        elif self.loss_func.lower() == "mse":  # 搭配L1
            loss = np.sum(np.power(y_hat - y_true, 2)) / (2 * y_hat.shape[0])
            dz_hat = y_hat - y_true
        loss_w = 0
        if self.regularization == "L1":
            for w in self.w:
                loss_w += np.sum(np.abs(w))
        elif self.regularization == "L2":
            for w in self.w:
                loss_w += np.sum(np.power(w, 2))
        loss_w = loss_w * self.regularization_rate / y_hat.shape[0]
        loss = loss + loss_w
        return {"loss": loss, "dz_hat": dz_hat}

    def __initializeWeightMatrix_f(self):

        if self.isInitializeWeightMatrix:
            if self.activation_func == "relu":
                for w in self.w:
                    w *= np.sqrt(2 / w.shape[0])
            else:
                for w in self.w:
                    w *= np.sqrt(1 / w.shape[0])

    def __optimizer_f(self, learn_rate, index, optimizer='adam', global_step=None):

        if optimizer.lower() == "sgdm":
            self.m_w[index] = self.beta * self.m_w[index] + (1 - self.beta) * self.dw[index]
            self.m_b[index] = self.beta * self.m_b[index] + (1 - self.beta) * self.db[index]
            m_w, m_b, V_w, V_b = self.m_w[index], self.m_b[index], self.V_w[index], self.V_b[index]
        elif optimizer.lower() == "adagrad":
            self.m_w[index] = self.dw[index]
            self.m_b[index] = self.db[index]
            self.V_w[index] += np.square(self.dw[index])
            self.V_b[index] += np.square(self.db[index])
            m_w, m_b, V_w, V_b = self.m_w[index], self.m_b[index], self.V_w[index], self.V_b[index]
        elif optimizer.lower() == "adam":
            self.m_w[index] = self.beta * self.m_w[index] + (1 - self.beta) * self.dw[index]
            self.m_b[index] = self.beta * self.m_b[index] + (1 - self.beta) * self.db[index]
            self.V_w[index] = self.beta1 * self.V_w[index] + (1 - self.beta1) * np.square(self.dw[index])
            self.V_b[index] = self.beta1 * self.V_b[index] + (1 - self.beta1) * np.square(self.db[index])
            m_w, m_b, V_w, V_b = self.m_w[index], self.m_b[index], self.V_w[index], self.V_b[index]
            # 修正后的一阶和二阶动量
            m_w = m_w / ((1 - np.power(self.beta, int(global_step))) + self.alpha)
            m_b = m_b / ((1 - np.power(self.beta, int(global_step))) + self.alpha)
            V_w = V_w / ((1 - np.power(self.beta1, int(global_step))) + self.alpha)
            V_b = V_b / ((1 - np.power(self.beta1, int(global_step))) + self.alpha)
        # 下降梯度yita = lr * mt / sqrt(Vt)
        yita_w = learn_rate * (m_w / np.sqrt(V_w))
        yita_b = learn_rate * (m_b / np.sqrt(V_b))
        return yita_w, yita_b

    def __learning_rate_decay(self, epoch_num):

        return (1 / (1 + self.decay_rate * epoch_num)) * self.lr

    def forward(self, x, w=None, b=None):
        if not w:
            w = self.w
        if not b:
            b = self.b
        a = []  # a = g(z)
        z = np.dot(x, w[0]) + b[0]
        aa = self.__activation_f(activation=self.activation_func, z=z, dropout=self.dropout_rate[0])
        a.append(aa["a"])
        for l in range(1, self.layers_num):
            if l < self.layers_num - 1:
                activation = self.activation_func
                dropout = self.dropout_rate[l]
            # 输出层
            elif l == self.layers_num - 1:
                activation = self.o_activation_func
                dropout = 0
            z = np.dot(a[l - 1], w[l]) + b[l]
            aa = self.__activation_f(activation=activation, z=z, dropout=dropout)
            a.append(aa["a"])
        return a[-1]

    def fit(self, train_x, train_y, batch_size=None, epochs=10):

        sample_num = train_x.shape[0]
        nd = np.random.RandomState(14)
        self.w.insert(0, nd.rand(train_x.shape[1], self.layerStructure[0]) * 0.01)  # 添加第一层网络的训练参数w
        self.__initializeWeightMatrix_f()  # 初始化权重矩阵
        self.m_w, self.m_b = [np.zeros(m_w.shape) for m_w in self.w], [np.zeros(m_b.shape) for m_b in self.b]
        self.V_w, self.V_b = [np.ones(v_w.shape) for v_w in self.w], [np.ones(v_b.shape) for v_b in self.b]
        global_step = 0
        print("TrainSample : {} ".format(train_x.shape[0]))

        for epoch in range(epochs):
            time_s = time.time()
            batch_count = 0
            front = 0
            rear = 0
            batch_size = batch_size if batch_size is not None else sample_num
            while True:
                if rear == sample_num:
                    break
                front = batch_count * batch_size
                rear = front + batch_size
                rear = rear if rear <= sample_num else sample_num  # 防止溢出
                mini_batch_size = rear - front
                batch_count += 1
                global_step += 1

                x_mini_batch = train_x[front:rear]
                y_mini_batch = train_y[front:rear]

                self.z = []  # z = x * w.T + b
                self.a = []  # a = g(z)
                self.dg = []  # dg = da/dz
                self.dz = []  # dz = dL/dz = dL/da * da/dz
                self.dw = []  # dw = dL/dw = dL/dz * dz/dw
                self.db = []  # db = dL/db = dL/dz * dz/db = dL/dz

                # ###前向传播###
                z = np.dot(x_mini_batch, self.w[0]) + self.b[0]
                aa = self.__activation_f(activation=self.activation_func, z=z, dropout=self.dropout_rate[0])
                self.z.append(z)
                self.a.append(aa["a"])
                self.dg.append(aa["dg"])
                for l in range(1, self.layers_num):
                    if l < self.layers_num - 1:
                        activation = self.activation_func
                        dropout = self.dropout_rate[l]
                    # 输出层
                    elif l == self.layers_num - 1:
                        activation = self.o_activation_func
                        dropout = 0
                    z = np.dot(self.a[l - 1], self.w[l]) + self.b[l]
                    aa = self.__activation_f(activation=activation, z=z, dropout=dropout)
                    self.z.append(z)
                    self.a.append(aa["a"])
                    self.dg.append(aa["dg"])

                # ###反向传播###
                loss = self.__loss_f(y_hat=self.a[-1], y_true=y_mini_batch)
                self.dz.insert(0, loss["dz_hat"])
                self.dw.insert(0, np.dot(self.a[-2].T, self.dz[0]) / mini_batch_size)
                self.db.insert(0, np.sum(self.dz[0], axis=0, keepdims=True) / mini_batch_size)
                for b in reversed(range(0, self.layers_num - 1)):
                    self.dz.insert(0, np.dot(self.dz[0], self.w[b + 1].T) * self.dg[b])
                    if b > 0:
                        self.dw.insert(0, np.dot(self.a[b - 1].T, self.dz[0]) / mini_batch_size)
                    elif b == 0:
                        self.dw.insert(0, np.dot(x_mini_batch.T, self.dz[0]) / mini_batch_size)
                    self.db.insert(0, np.sum(self.dz[0], axis=0, keepdims=True) / mini_batch_size)

                # ###参数更新###
                for i in range(len(self.dw)):
                    yita_w, yita_b = self.__optimizer_f(learn_rate=self.__learning_rate_decay(epoch),
                                                        optimizer=self.optimizer, index=i, global_step=global_step)
                    self.w[i] -= yita_w
                    self.b[i] -= yita_b

            time_u = time.time() - time_s

            self.loss.append(loss["loss"])

            print("epoch_num: {:0>3d}   train_loss: {:0<.5f}    use_time: {:0<.5f}min".format(epoch, self.loss[-1],
                                                                                              time_u / 60))
