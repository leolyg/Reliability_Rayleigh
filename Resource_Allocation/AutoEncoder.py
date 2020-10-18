import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.initializers import Zeros
from keras.callbacks import TensorBoard
from keras import optimizers
from keras import backend as K
import tensorflow as tf
from Resource_Allocation.Parameter import Parameter



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
    # 训练编码器
    def train(self, x_train, y_train):
        # 通常情况x_train  = y_train
        # self.autoencoder.fit(x_train, y_train, epochs=20, batch_size=32,validation_split=0.2,callbacks=self.callbacks)
        self.autoencoder_model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

    def predict(self, x_test):
        return np.rint(self.autoencoder_model.predict(x_test))

    # 编码
    def encoding(self, x_test):
        return self.encoder_model.predict(x_test)

    # #解码
    def decoding(self, encoded_data):
        return self.decoder_model.predict(encoded_data)

    def save_model(self,suffix):
        self.autoencoder_model.save("autoencoder/autoencoder"+suffix+".h5")
        self.encoder_model.save("autoencoder/encoder"+suffix+".h5")
        self.decoder_model.save("autoencoder/decoder"+suffix+".h5")

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

def training(parameter,encoding_dim):
    #input_shape = parameter.bs_num*parameter.subcarrier_num
    input_shape = parameter.step_threshold
    output_shape = input_shape
    encoding_dim = encoding_dim
    autoencoder = AutoEncoder(encoding_dim=encoding_dim, input_shape=input_shape, output_shape=output_shape)
    autoencoder.build_net(input_shape, output_shape)
    # 训练相关代码
    sample_size =1000000
    x_train = np.random.randint(-1, parameter.ue_num, size=sample_size*input_shape).reshape(sample_size, input_shape)
    autoencoder.train(x_train, x_train)
    suffix = ''+str(parameter.ue_num)+'-'+str(parameter.step_threshold)+'-'+str(encoding_dim)
    autoencoder.save_model(suffix)

def load_autoencoder_model():
    autoencoder_model = load_model("autoencoder/autoencoder.h5")
    return autoencoder_model

def load_encode_model(parameter,encoding_dim):
    suffix = '' + str(parameter.ue_num) + '-' + str(parameter.step_threshold) + '-' + str(encoding_dim)
    encode_model = load_model("autoencoder/encoder"+suffix+".h5")
    return encode_model

def load_decode_model(parameter,encoding_dim):
    suffix = '' + str(parameter.ue_num) + '-' + str(parameter.step_threshold) + '-' + str(encoding_dim)
    decode_model = load_model("autoencoder/decoder"+suffix+".h5")
    return decode_model

if __name__ == "__main__":
    parameter = Parameter(
        bs_num=5,
        subcarrier_num=12,
        ue_num=6,
        step_threshold=8,
        power_level=8,  # 功率级别
    )
    training(parameter, 16)
    training(parameter,32)
    # model = load_autoencoder_model()
    # decode_model = load_decode_model()
    # encode_model = load_encode_model()
    # state = np.array([[-1,-1,-1,-1,-1,-1],
    #                   [1,2,3,4,1,2]])
    #
    # print(decode_model.predict(encode_model.predict(state)))

    K.clear_session()


