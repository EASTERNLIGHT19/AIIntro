import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR

FILES = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images": "t10k-images-idx3-ubyte",
    "test_labels": "t10k-labels-idx1-ubyte"
}


def load_images(filename):
    path = os.path.join(DATA_DIR, filename)

    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)

    data = data.reshape(num, 1, rows, cols).astype(np.float32)
    data /= 255.0

    return data


def load_labels(filename):
    path = os.path.join(DATA_DIR, filename)

    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels


def load_fashion_mnist():
    x_train = load_images(FILES["train_images"])
    t_train = load_labels(FILES["train_labels"])
    x_test = load_images(FILES["test_images"])
    t_test = load_labels(FILES["test_labels"])

    return (x_train, t_train), (x_test, t_test)


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape

    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h

        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3)
    col = col.reshape(N * out_h * out_w, -1)

    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape

    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w)
    col = col.transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))

    for y in range(filter_h):
        y_max = y + stride * out_h

        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


def softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        y = np.exp(x)

        return y / np.sum(y, axis=1, keepdims=True)

    x = x - np.max(x)

    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]

    if t.size == y.size:
        t = t.argmax(axis=1)

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0

        return dout


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape

        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1)
        out = out.transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape

        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)

        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape

        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        self.arg_max = np.argmax(col, axis=1)

        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C)
        out = out.transpose(0, 3, 1, 2)

        self.x = x

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))

        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()

        dcol = dmax.reshape(dmax.shape[0], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx


class SimpleConvNet:
    def __init__(
            self,
            input_dim=(1, 28, 28),
            conv_param_1={"filter_num": 32, "filter_size": 3, "pad": 1, "stride": 1},
            conv_param_2={"filter_num": 64, "filter_size": 3, "pad": 1, "stride": 1},
            hidden_size=256,
            output_size=10,
            weight_init_std=0.01
    ):
        filter_num_1 = conv_param_1["filter_num"]
        filter_size_1 = conv_param_1["filter_size"]
        filter_pad_1 = conv_param_1["pad"]
        filter_stride_1 = conv_param_1["stride"]

        filter_num_2 = conv_param_2["filter_num"]
        filter_size_2 = conv_param_2["filter_size"]
        filter_pad_2 = conv_param_2["pad"]
        filter_stride_2 = conv_param_2["stride"]

        input_size = input_dim[1]

        conv_output_size_1 = int((input_size - filter_size_1 + 2 * filter_pad_1) / filter_stride_1 + 1)
        conv_output_size_2 = int((conv_output_size_1 - filter_size_2 + 2 * filter_pad_2) / filter_stride_2 + 1)

        pool_output_size = int(filter_num_2 * (conv_output_size_2 / 2) * (conv_output_size_2 / 2))

        self.params = {}

        self.params["W1"] = weight_init_std * np.random.randn(
            filter_num_1,
            input_dim[0],
            filter_size_1,
            filter_size_1
        )
        self.params["b1"] = np.zeros(filter_num_1)

        self.params["W2"] = weight_init_std * np.random.randn(
            filter_num_2,
            filter_num_1,
            filter_size_2,
            filter_size_2
        )
        self.params["b2"] = np.zeros(filter_num_2)

        self.params["W3"] = weight_init_std * np.random.randn(
            pool_output_size,
            hidden_size
        )
        self.params["b3"] = np.zeros(hidden_size)

        self.params["W4"] = weight_init_std * np.random.randn(
            hidden_size,
            output_size
        )
        self.params["b4"] = np.zeros(output_size)

        self.layers = OrderedDict()

        self.layers["Conv1"] = Convolution(
            self.params["W1"],
            self.params["b1"],
            conv_param_1["stride"],
            conv_param_1["pad"]
        )
        self.layers["Relu1"] = Relu()

        self.layers["Conv2"] = Convolution(
            self.params["W2"],
            self.params["b2"],
            conv_param_2["stride"],
            conv_param_2["pad"]
        )
        self.layers["Relu2"] = Relu()

        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers["Affine1"] = Affine(self.params["W3"], self.params["b3"])
        self.layers["Relu3"] = Relu()

        self.layers["Affine2"] = Affine(self.params["W4"], self.params["b4"])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)

        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        acc = 0.0

        for i in range(0, x.shape[0], batch_size):
            tx = x[i:i + batch_size]
            tt = t[i:i + batch_size]

            y = self.predict(tx)
            y = np.argmax(y, axis=1)

            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}

        grads["W1"] = self.layers["Conv1"].dW
        grads["b1"] = self.layers["Conv1"].db

        grads["W2"] = self.layers["Conv2"].dW
        grads["b2"] = self.layers["Conv2"].db

        grads["W3"] = self.layers["Affine1"].dW
        grads["b3"] = self.layers["Affine1"].db

        grads["W4"] = self.layers["Affine2"].dW
        grads["b4"] = self.layers["Affine2"].db

        return grads


(x_train, t_train), (x_test, t_test) = load_fashion_mnist()

network = SimpleConvNet()

iters_num = 15000
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = train_size // batch_size

print("훈련 데이터:", x_train.shape, t_train.shape)
print("테스트 데이터:", x_test.shape, t_test.shape)
print("학습 시작")

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)

    epoch = i // iter_per_epoch

    if epoch >= 15:
        current_lr = 0.005
    elif epoch >= 10:
        current_lr = 0.01
    else:
        current_lr = 0.03

    for key in network.params.keys():
        network.params[key] -= current_lr * grads[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:", epoch, "loss:", loss, "train acc:", train_acc, "test acc:", test_acc)

final_train_acc = network.accuracy(x_train, t_train)
final_test_acc = network.accuracy(x_test, t_test)

print("최종 훈련 정확도:", final_train_acc * 100, "%")
print("최종 테스트 정확도:", final_test_acc * 100, "%")

x = np.arange(len(train_acc_list))

train_acc_percent = np.array(train_acc_list) * 100
test_acc_percent = np.array(test_acc_list) * 100

plt.figure()
plt.plot(x, train_acc_percent, label="train acc")
plt.plot(x, test_acc_percent, label="test acc", linestyle="--")
plt.xlabel("epochs")
plt.ylabel("accuracy (%)")
plt.ylim(0, 100)
plt.legend()
plt.title("Fashion MNIST CNN Accuracy")
plt.savefig("fashion_mnist_cnn_accuracy.png")
plt.show()