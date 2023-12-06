# -*- coding: utf-8 -*-
__author__ = 'Mike'
'''
1、VAE变分自编码器和GAN生成对抗网络。VAE和GAN采用不同的策略。
VAE非常适合学习具有良好结构的潜在空间，空间中的特定方向表示数据中有意义的变化轴。
GAN可以生成非常逼真的图像，但他的潜在空间可能没有良好得结构，连续性也不强。
2、图像编辑的概念向量，比如在人脸图像的潜在空间中，就可能存在一个微笑向量smile vector
3、变分自编码器VAE，VAE在自编码器上添加了一些统计魔法，迫使其学习连续、高度结构化的潜在空间。然后生成图像。
    VAE并没有将输入图像压缩在潜在空间中的固定编码，而是将图像转换为统计分布的参数，即均值和方差。这本质上意味着
    我们假设输入图像是由统计过程生成的，在编码和解码的过程中应该考虑随机性。VAE使用均值和方差这两个参数从分布中随机采样一个元素，
    并将这个元素解码为原始输入。这一过程的随机性提高了其稳健性，并迫使潜在空间任何位置都对应有意义的表示，
    即在潜在空间中采样的每个点都能解码为有效输出
VAE伪代码：
    z_mean, z_log_var = encoder(input_img)
    z = z_mean + exp(0.5 * z_log_var) * epsilon #利用随机的小epsilon来抽取一个潜在点
    reconstructed_img = decodder(z) #将z解码为图像
    model = Model(input_img, reconstructed_img) #将自编码器模型实例化，它将输入图像映射为他的重构结果
'''
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Sampler(layers.Layer):
    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size)) #从正态分布中随机抽取一个向量批量
        #采用VAE采样公式
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self,encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampler()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss") #让KL散度来保证数据偏差不大

    @property
    #指标属性中列出各项指标，可以让模型在每轮过后或者在多次调用fit()、evaluate之间重置这些指标
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler(z_mean, z_log_var)
            reconstruction = decoder(z)
            #对重构损失在空间维度（轴1和轴2）上求和，并在批量维度上计算均值
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction),
                axis=(1, 2)
            ))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean)) - tf.exp(z_log_var)
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }


latent_dim = 2

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
#利用z_mean和z_log_var来生成一个潜在空间点,假设二者是生成input_img的统计分布参数
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")

#VAE解码器网络，将潜在空间点映射为图像
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x) #编码器Flatten层的逆操作
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

#训练VAE
(x_train, _),(x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32")/255

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
callbacks = [keras.callbacks.ModelCheckpoint("vae.keras", save_best_only=True)]
vae.fit(mnist_digits, epochs=30, batch_size=128)

n = 30
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
#在二维网格上对点进行线性采样
grid_x = np.linspace(-1, 1, n)
grid_y = np.linspace(-1, 1, n)[::-1]

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = vae.decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[
            i * digit_size : (i + 1) * digit_size,
            j * digit_size : (j + 1) * digit_size
        ]=digit

plt.figure(figsize=(15, 15))
start_range = digit_size // 2
end_range = n * digit_size + start_range
pixel_range = np.arange(start_range, end_range, digit_size)
sample_range_x = np.round(grid_x, 1)
sample_range_y = np.round(grid_y, 1)
plt.xticks(pixel_range, sample_range_x)
plt.yticks(pixel_range, sample_range_y)
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.axis("off")
plt.imshow(figure, cmap="Greys_r")