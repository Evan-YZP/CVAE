# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data

FILE_PATH = os.path.dirname(__file__)


class CVAE(object):
    def __init__(self, input_shapes, filters, kernel_sizes, n_features=None, latent_size=96):
        self.input_shapes = input_shapes
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.latent_size = latent_size
        self.n_features = n_features
        self.latent_kernal = int(np.sqrt(self.n_features/self.filters[-1]))
        self.input_x = tf.placeholder(tf.float32, shape=[None, self.input_shapes[0], self.input_shapes[1], 1], name='Input_x')
        self.latent_z = tf.placeholder(tf.float32, shape=[None, self.latent_size], name='latent_z')
        self.weights = self.init_weights()

    def init_weights(self, ):
        all_weights = dict()
        n_layers = len(self.filters)
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=None, dtype=tf.float32)
        b_init = tf.constant_initializer(0)
        de_kernel_size = self.kernel_sizes[::-1]  # 解码器卷积核
        de_n_hidden = self.filters[:: -1]  # 解码器隐层神经元个数
        with tf.variable_scope("Init_Weight"):
            for iter_i in range(n_layers):
                enc_name_wi = 'enc_w' + str(iter_i)
                enc_name_bi = 'enc_b' + str(iter_i)
                if iter_i == 0:
                    all_weights[enc_name_wi] = tf.get_variable(enc_name_wi,
                                                            shape=[self.kernel_sizes[iter_i],
                                                                    self.kernel_sizes[iter_i], 1,
                                                                    self.filters[iter_i]],
                                                                    initializer=w_init, dtype=tf.float32)
                    
                else:
                    all_weights[enc_name_wi] = tf.get_variable(enc_name_wi,
                                                            shape=[self.kernel_sizes[iter_i],
                                                                    self.kernel_sizes[iter_i], self.filters[iter_i-1],
                                                                    self.filters[iter_i]], 
                                                                    initializer=w_init, dtype=tf.float32)
                all_weights[enc_name_bi] = tf.get_variable(enc_name_bi, shape=[self.filters[iter_i]],
                                                                    initializer=b_init, dtype=tf.float32)

            for iter_i in range(n_layers):
                dec_name_wi = 'dec_w' + str(iter_i)
                dec_name_bi = 'dec_b' + str(iter_i)
                if iter_i < n_layers - 1:
                    all_weights[dec_name_wi] = tf.get_variable(dec_name_wi,
                                                            shape=[de_kernel_size[iter_i], de_kernel_size[iter_i],
                                                                    de_n_hidden[iter_i+1], de_n_hidden[iter_i]],
                                                                    initializer=w_init, dtype=tf.float32)
                    all_weights[dec_name_bi] = tf.get_variable(dec_name_bi, shape=[de_n_hidden[iter_i+1]],
                                                                            initializer=b_init, dtype=tf.float32)
                else:
                    all_weights[dec_name_wi] = tf.get_variable(dec_name_wi,
                                                            shape=[de_kernel_size[iter_i], de_kernel_size[iter_i],
                                                                    1, de_n_hidden[iter_i]],
                                                                    initializer=w_init, dtype=tf.float32)
                    all_weights[dec_name_bi] = tf.get_variable(dec_name_bi, shape=[1], initializer=b_init, dtype=tf.float32)
        return all_weights

    def encoder(self, x, weights):
        layer_i = x
        shapes = list()
        n_layers = len(self.filters)
        with tf.name_scope("Encoder"):
            print('encoder')
            for iter_i in range(n_layers):
                shapes.append(layer_i.get_shape().as_list())
                print(layer_i.get_shape().as_list())
                layer_i = tf.nn.bias_add(tf.nn.conv2d(layer_i, weights['enc_w'+str(iter_i)], strides=[1, 2, 2, 1],
                                                      padding='SAME'), weights['enc_b'+str(iter_i)])
                layer_i = tf.nn.relu(layer_i)
            print(layer_i.get_shape().as_list())
            layer_x = tf.layers.flatten(layer_i, name='Flatten')
            z_mean = tf.contrib.layers.fully_connected(layer_x, self.latent_size, activation_fn=None)
            z_log_var = tf.contrib.layers.fully_connected(layer_x, self.latent_size, activation_fn=None)
        return z_mean, z_log_var, shapes

    def decoder(self, z, weights, shapes=None, train_able=True, reuse=False):
        n_layers = len(self.filters)      
        print('decoder') 
        shapes = shapes[::-1]
        with tf.variable_scope('scope', reuse=reuse):
            # z = tf.layers.dense(z, self.n_features, activation=None, trainable=train_able, name='de_full_connect')
            z = CVAE.fully_connected(z, self.n_features, 'de_full_connect')
            # z = tf.contrib.layers.fully_connected(z, self.n_features, activation_fn=None, trainable=train_able)
            z = tf.reshape(z, [-1, self.latent_kernal, self.latent_kernal, self.filters[-1]])
            layer_i = z            
            for iter_i in range(n_layers):
                print(layer_i.get_shape().as_list())
                shape_de = shapes[iter_i]            
                layer_i = tf.add(tf.nn.conv2d_transpose(layer_i, weights['dec_w'+str(iter_i)],
                                                        tf.stack([tf.shape(layer_i)[0], shape_de[1],
                                                                  shape_de[2], shape_de[3]]), strides=[1, 2, 2, 1],
                                                        padding='SAME'), weights['dec_b'+str(iter_i)])
                if iter_i == n_layers - 1:
                    layer_i = tf.nn.sigmoid(layer_i)
                else:
                    layer_i = tf.nn.relu(layer_i)
            
            print(layer_i.get_shape().as_list())
        return layer_i

    def build_model(self, ):
        z_mean, z_log_var, shapes = self.encoder(self.input_x, self.weights)
        with open('de_shapes.pkl', 'wb') as f:
            pkl.dump(shapes, f)
        sample_z = CVAE.reparameterization(z_mean, z_log_var)
        print(sample_z.get_shape().as_list())
        const_x = self.decoder(sample_z, self.weights, shapes)

        # 定义损失
        with tf.name_scope('Loss'):
            cross_entropy_loss = self.input_x * tf.log(tf.clip_by_value(const_x, 1e-6, 1-1e-6)) + (1-self.input_x) * tf.log(tf.clip_by_value(1-const_x, 1e-6, 1-1e-6))
            recon_loss = -tf.reduce_sum(cross_entropy_loss, axis=[1, 2, 3], name='recon_loss')

            kl = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = -tf.reduce_sum(kl, axis=-1, name='KL_loss')

            model_loss = tf.reduce_mean(recon_loss+kl_loss)
        
        with tf.name_scope('opt'):
            opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(model_loss)
        return model_loss, opt, const_x, sample_z

    def fit_model(self, x, epoches):
        vae_loss, vae_opt, const_x, sample_z = self.build_model()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epoches):
                _, loss, recon = sess.run([const_x, vae_opt, vae_loss], feed_dict={self.input_x: x})
                if epoch % 19 == 0:
                    recon, _, loss, sam_z = sess.run([const_x, vae_opt, vae_loss, sample_z], feed_dict={self.input_x: x})
                    print('第 {} 次迭代, 损失为{}'.format(epoch, loss))
                    self.save_model(FILE_PATH, sess)
            recon_x = sess.run(const_x, feed_dict={self.input_x: x})
            reshape_x = recon_x.reshape([-1, 28, 28, 1])
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.imshow(reshape_x[i, :, :, 0], cmap='gray')
                plt.axis('off')
            plt.show()

    def reconstruct_x(self, nums_samples):
        with open('de_shapes.pkl', 'rb') as f:
            shapes = pkl.load(f)
        z_samples = np.random.randn(nums_samples, self.latent_size)
        const_x = self.decoder(self.latent_z, self.weights, shapes, train_able=False, reuse=True)
        with tf.Session() as sess:
            self.load_model(FILE_PATH, sess)
            recon_x = sess.run(const_x, feed_dict={self.latent_z: z_samples})
        reshape_x = recon_x.reshape([-1, 28, 28, 1])
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.imshow(reshape_x[i, :, :, 0], cmap='gray')
            plt.axis('off')
        plt.show()
    
    def save_model(self, path_, sess):
        m_saver = tf.train.Saver()
        model_path = os.path.join(path_, 'cvae_model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            print('创建模型文件目录')
        m_saver.save(sess, os.path.join(model_path, 'cvae.ckpt'))
        
    def load_model(self, path_, sess):
        m_saver = tf.train.Saver()
        model_path = os.path.join(path_, 'cvae_model')
        print(model_path)
        try:
            m_saver.restore(sess, os.path.join(model_path, 'cvae.ckpt'))
        except:
            raise Exception('{} 不存在'.format(model_path))

    @staticmethod
    def reparameterization(z_mean, z_log_var):
        eps = tf.random_normal(shape=tf.shape(z_mean), mean=0.0, stddev=1.0, dtype=tf.float32)
        return z_mean + eps * tf.exp(z_log_var/2)

    @staticmethod
    def fully_connected(x, output_dim, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
            w = tf.get_variable("weights", [x.shape[1], output_dim], initializer=tf.random_normal_initializer())
            b = tf.get_variable("biases", [output_dim], initializer=tf.constant_initializer(0.0))
            return tf.matmul(x, w) + b

    
def load_datas():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
    # 标准化图片到区间 [0., 1.] 内
    train_images /= 255.
    test_images /= 255.
    
    return train_images, test_images

if __name__=='__main__':
    tf.reset_default_graph()
    import math
    train_x, test_x = load_datas()
    train_input = train_x[: 400]
    filters = [64, 32, 32]
    input_shapes = [28, 28]
    kernel_sizes= [3, 3, 3]
    n_features = filters[-1] * (math.ceil(input_shapes[0] / (2 ** len(kernel_sizes))) ** 2)
    model_vae = CVAE(input_shapes=input_shapes, filters=filters, kernel_sizes=kernel_sizes, n_features=n_features)
    # model_vae.fit_model(train_input, 5000)
    model_vae.reconstruct_x(16)

    