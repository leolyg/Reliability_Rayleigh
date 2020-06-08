import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.initializers import Zeros
from keras.callbacks import TensorBoard
from keras import optimizers
from keras import backend as K
import tensorflow as tf


class AutoEncoder:
    def __init__(self, encoding_dim=32, input_shape=100, output_shape=100):
        self.encoding_dim = encoding_dim
        # self.build_encoder(input_shape,output_shape)

    def build_net(self, input_shape, output_shape):
        encoding_dim = self.encoding_dim

        input_vec = Input(shape=(input_shape,))
        encoded = Dense(16, kernel_initializer='random_uniform', bias_initializer='zeros')(input_vec)
        encoded = Dense(16, kernel_initializer='random_uniform', bias_initializer='zeros')(encoded)
        encoded_output = Dense(encoding_dim, kernel_initializer='random_uniform', bias_initializer='zeros')(encoded)
        decoded = Dense(16, kernel_initializer='random_uniform', bias_initializer='zeros')(encoded_output)
        decoded = Dense(16, kernel_initializer='random_uniform', bias_initializer='zeros')(decoded)
        decoded = Dense(output_shape, kernel_initializer='random_uniform', bias_initializer='zeros')(decoded)

        autoencoder_model = Model(inputs=input_vec, outputs=decoded)
        encoder_model = Model(inputs=input_vec, outputs=encoded_output)
        decoded_input = Input(shape=(encoding_dim,))
        decoder_layer1 = autoencoder_model.layers[-3]
        decoder_layer2 = autoencoder_model.layers[-2]
        decoder_layer3 = autoencoder_model.layers[-1]
        decoded_output = decoder_layer3(decoder_layer2(decoder_layer1(decoded_input)))
        decoder_model = Model(decoded_input, decoded_output)

        autoencoder_model.compile(optimizer='Adam', loss='mse')


        self.autoencoder_model = autoencoder_model
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

        tb = TensorBoard(log_dir='./logs',  # log 目录
                         histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                         batch_size=32,  # 用多大量的数据计算直方图
                         write_graph=True,  # 是否存储网络结构图
                         write_grads=False,  # 是否可视化梯度直方图
                         write_images=False,  # 是否可视化参数
                         embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None)
        self.callbacks = [tb]

    # 训练编码器
    def train(self, x_train, y_train):
        # 通常情况x_train  = y_train
        # self.autoencoder.fit(x_train, y_train, epochs=20, batch_size=32,validation_split=0.2,callbacks=self.callbacks)
        self.autoencoder_model.fit(x_train, y_train, epochs=40, batch_size=32, validation_split=0.2)

    def predict(self, x_test):
        return np.rint(self.autoencoder_model.predict(x_test))

    # 编码
    def encoding(self, x_test):
        return self.encoder_model.predict(x_test)

    # #解码
    def decoding(self, encoded_data):
        return self.decoder_model.predict(encoded_data)

    def save_model(self):
        self.autoencoder_model.save("autoencoder.h5")
        self.encoder_model.save("encoder.h5")
        self.decoder_model.save("decoder.h5")
    def save_power_encoding_model(self):
        self.autoencoder_model.save("autoencoder_power.h5")
        self.encoder_model.save("encoder_power.h5")
        self.decoder_model.save("decoder_power.h5")

    def load_model(self):
        autoencoder_model = load_model("DQN/save/autoencoder.h5")
        encoder_model = load_model("DQN/save/encoder.h5")
        decoder_model = load_model("DQN/save/decoder.h5")
        self.autoencoder_model = autoencoder_model
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

    def load_power_model(self):
        autoencoder_model = load_model("plot/autoencoder_power.h5")
        encoder_model = load_model("plot/encoder_power.h5")
        decoder_model = load_model("plot/decoder_power.h5")
        self.autoencoder_model = autoencoder_model
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model


def customized_accuracy(y_true, y_pred):
    # 四舍五入运算
    y_pred = K.round(y_pred)
    not_equal = K.sum(K.clip((y_true - y_pred), 0, 1))
    total = K.sum(K.clip((y_true), 1, 1))
    return 1 - not_equal / total


def multi_training():
    for i in [10, 20, 30, 40, 50]:
        print("input shape ", i)
        sample_size = 100000
        autoencoder = AutoEncoder(encoding_dim=64, input_shape=i, output_shape=i)
        x_train = np.random.randint(-1, 5, size=sample_size * i).reshape(sample_size, i)
        autoencoder.train(x_train, x_train)
        K.clear_session()
        del autoencoder
        del x_train


def training():
    input_shape = 50
    output_shape = 50
    autoencoder = AutoEncoder(encoding_dim=16, input_shape=input_shape, output_shape=output_shape)
    autoencoder.build_net(input_shape, output_shape)
    # 训练相关代码
    x_train = np.random.randint(-1, 5, size=40000000).reshape(800000, 50)
    autoencoder.train(x_train, x_train)
    autoencoder.save_model()

def train_power_autoencoder():
    input_shape = 12
    output_shape = 12
    autoencoder = AutoEncoder(encoding_dim=6, input_shape=input_shape, output_shape=output_shape)
    autoencoder.build_net(input_shape, output_shape)
    # 训练相关代码
    x_train = np.random.randint(1, 16, size=120000).reshape(10000, 12)
    print(x_train.shape)
    autoencoder.train(x_train, x_train)
    autoencoder.save_power_encoding_model()
    input = np.array([[15,15,15,15,15,15,15,15,15,15,15,15]])
    output = autoencoder.predict(input)
    print(output)


if __name__ == "__main__":
    training()
    #train_power_autoencoder()
    # input = np.array([[15, 12, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]])
    # autoencoder = AutoEncoder(encoding_dim=6, input_shape=12, output_shape=12)
    # autoencoder.load_power_model()
    # print(autoencoder.encoder_model.predict(input))
