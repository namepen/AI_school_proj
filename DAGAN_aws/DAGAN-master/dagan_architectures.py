import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, layer_norm
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from tensorflow.python.ops.nn_ops import leaky_relu
from utils.network_summary import count_parameters
slim = tf.contrib.slim
import numpy as np


def remove_duplicates(input_features):
    """
    Remove duplicate entries from layer list.
    :param input_features: A list of layers
    :return: Returns a list of unique feature tensors (i.e. no duplication).
    """
    feature_name_set = set()
    non_duplicate_feature_set = []
    for feature in input_features:
        if feature.name not in feature_name_set:
            non_duplicate_feature_set.append(feature)
        feature_name_set.add(feature.name)
    return non_duplicate_feature_set


class UResNetGenerator:
    def __init__(self, layer_sizes, layer_padding, batch_size, num_channels=1,
                 inner_layers=0, name="g"):
        """
        Initialize a UResNet generator.
        :param layer_sizes: A list with the filter sizes for each MultiLayer e.g. [64, 64, 128, 128]
        :param layer_padding: A list with the padding type for each layer e.g. ["SAME", "SAME", "SAME", "SAME"]
        :param batch_size: An integer indicating the batch size
        :param num_channels: An integer indicating the number of input channels
        :param inner_layers: An integer indicating the number of inner layers per MultiLayer
        """
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_sizes = layer_sizes
        self.layer_padding = layer_padding
        self.inner_layers = inner_layers
        self.conv_layer_num = 0
        self.build = True
        self.name = name
        self.encoder_layers = []
        self.decoder_layers = []
        self.num_filters = 8
        self.batch_norm_params = {'decay' : 0.99, 'scale' : True, 'center' : True,
                                 'is_training' : False, 'renorm' : True}
        
    def upscale(self, x, h_size, w_size):
        """
        Upscales an image using nearest neighbour
        :param x: Input image
        :param h_size: Image height size
        :param w_size: Image width size
        :return: Upscaled image
        """
        [b, h, w, c] = [int(dim) for dim in x.get_shape()]

        return tf.image.resize_nearest_neighbor(x, (h_size, w_size))
    
    def encoder_block(self, x1, num): 
        #with tf.variable_scope(scope +'/encoder_block'):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            num_outputs = 64, padding = 'SAME',
                            kernel_size = [3,3], stride = (1,1),
                            activation_fn = tf.nn.leaky_relu,
                            normalizer_fn=slim.batch_norm, normalizer_params= self.batch_norm_params):

            conv1_1 = slim.conv2d(x1)
            output1_1 = tf.concat([conv1_1, x1], axis=3)

            conv1_2 = slim.conv2d(output1_1)
            output1_2 = tf.concat([conv1_2, output1_1], axis=3)

            conv1_3 = slim.conv2d(output1_2)
            output1_3 = tf.concat([conv1_3, output1_2], axis=3)

            conv1_4 = slim.conv2d(output1_3, stride=(2,2))
            output = slim.dropout(conv1_4, keep_prob=0.5)
            self.encoder_layers.append(output)

            if num == 3:
                pass
            else :
                input_projection = slim.conv2d(conv1_3, num_outputs=conv1_3.get_shape()[3], stride=(2,2),
                                               activation_fn= None, normalizer_fn= None)
                output = tf.concat([output, input_projection], axis=3)

        return output
    
    def decoder_block(self, x1): 
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            num_outputs = 64, padding = 'SAME',
                            kernel_size = [3,3], stride = (1,1),
                            activation_fn = tf.nn.leaky_relu,
                            normalizer_fn=slim.batch_norm, normalizer_params= self.batch_norm_params):

            conv1_1 = slim.conv2d(x1)
            output1_1 = tf.concat([conv1_1, x1], axis=3)
            

            conv1_2 = slim.conv2d(output1_1)
            output1_2 = tf.concat([conv1_2, output1_1], axis=3)
            

            conv1_3 = slim.conv2d(output1_2)
            output1_3 = tf.concat([conv1_3, output1_2], axis=3)
            
            conv1_4 = slim.conv2d_transpose(output1_3, stride=(2,2))
            self.decoder_layers.append(conv1_4)
            output = slim.dropout(conv1_4, keep_prob=0.5)

            input_projection = slim.conv2d_transpose(conv1_3, num_outputs=conv1_3.get_shape()[3], stride=(2,2),
                                           activation_fn= None, normalizer_fn= None)
            output = tf.concat([output, input_projection], axis=3)

        return output
    
    def z_noise_concat(self, inputs, z_inputs, h, w):
        print(z_inputs)
        z_dense = tf.layers.dense(z_inputs, h*w*self.num_filters)
        z_noise = tf.reshape(z_dense, [self.batch_size, h, w, self.num_filters])
        z_noise_concat = tf.concat([inputs, z_noise], axis= 3)
        
        self.num_filters = np.int(self.num_filters / 2)
        
        return z_noise_concat
    
        
    def __call__(self, z_inputs, conditional_input, training=False, dropout_rate=0.0):
        """
        Apply network on data.
        :param z_inputs: Random noise to inject [batch_size, z_dim]
        :param conditional_input: A batch of images to use as conditionals [batch_size, height, width, channels]
        :param training: Training placeholder or boolean
        :param dropout_rate: Dropout rate placeholder or float
        :return: Returns x_g (generated images), encoder_layers(encoder features), decoder_layers(decoder features)
        """
        z_layer=[]
        h, w, n = 2, 2, 8
        for i in range(4):
            z_dence = tf.layers.dense(z_inputs, h*w*n)
            z_noise = tf.reshape(z_dence, [16,h,w,n])
            z_layer.append(z_noise)
            h = h*2
            w = w*2
            n = np.int(n/2)
        
        with tf.variable_scope(self.name, reuse=self.reuse):
            # reshape from inputs
            conditional_input = tf.convert_to_tensor(conditional_input)
            
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                       num_outputs = 64, padding = 'SAME',
                        kernel_size = [3,3], stride = (1,1), activation_fn = None):
                with tf.variable_scope(self.name + 'first_en_conv'):
                    conv = slim.conv2d(conditional_input, stride=(2,2))
                    self.encoder_layers.append(conv)
                    
                    input_projection = slim.conv2d(conditional_input, num_outputs=conditional_input.get_shape()[3], stride=(2,2))
                    output1 = tf.concat([conv, input_projection], axis=3)

                en1 = self.encoder_block(output1, 1) #[B,7, 7, 64]
                en2 = self.encoder_block(en1, 2) #[B,4,4,64]
                en3 = self.encoder_block(en2, 3) #[b,2,2,64]
                #end encoder
                
                with tf.variable_scope(self.name + '/First_de_conv'):
                    self.decoder_layers.append(en3)
                    #input_noise = self.z_noise_concat(en3, z_inputs, 2, 2)                #[b,2,2,72]
                    input_noise = tf.concat([en3, z_layer[0]], axis=3)
                    
                with tf.variable_scope(self.name + '/First_de_block'):
                    de_conv1 = self.decoder_block(input_noise)                            #[b, 4, 4, 64]
                    de_conv1 = tf.concat([de_conv1, self.encoder_layers[2]], axis=3)
                    #de_conv1_noise = self.z_noise_concat(de_conv1, z_inputs, 4, 4)
                    de_conv1_noise = tf.concat([de_conv1, z_layer[1]], axis=3)
                    
                with tf.variable_scope(self.name + '/Second_de_block'):
                    de_conv2 = self.decoder_block(de_conv1_noise)                         #[b, 8, 8, 64]
                    #de_conv2_noise = self.z_noise_concat(de_conv2, z_inputs, 8, 8)
                    de_conv2_noise = tf.concat([de_conv2, z_layer[2]], axis=3)
                    de_conv2 = self.upscale(de_conv2_noise, 7, 7)
                    de_conv2 = tf.concat([de_conv2, self.encoder_layers[1]], axis=3)
                    
                with tf.variable_scope(self.name + '/Third_de_block'):
                    de_conv3 = self.decoder_block(de_conv2)                               #[b, 14, 14 ,64]
                    de_conv3 = tf.concat([de_conv3, self.encoder_layers[0]], axis=3)
                    
                with tf.variable_scope(self.name + '/Forth_de_block'):
                    de_conv4 = self.decoder_block(de_conv3)                               #[b, 28, 28 ,64]
                    de_conv4 = tf.concat([de_conv4, conditional_input], axis=3)
                    
                with tf.variable_scope(self.name + '/Last_de_block'):
                    de_conv5_1 = slim.conv2d(de_conv4)
                    de_conv5_1 = tf.concat([de_conv5_1, de_conv4], axis=3)
                    
                    de_conv5_2 = slim.conv2d(de_conv5_1)
                    de_conv5_2 = tf.concat([de_conv5_2, de_conv5_1], axis=3)
                    
                with tf.variable_scope(self.name + '/P_process'):
                    de_conv = slim.conv2d(de_conv5_2)
                    de_conv = slim.conv2d(de_conv, num_outputs = 3)
                
                with tf.variable_scope('g_tanh'):
                    gan_decoder = tf.tanh(de_conv, name='outputs')
                
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        if self.build:
            count_parameters(self.variables, 'generator_parameter_num')
        self.build = False
        
        return gan_decoder, self.encoder_layers, self.decoder_layers        
                          


class Discriminator:
    def __init__(self, batch_size, layer_sizes, inner_layers, use_wide_connections=False, name="d"):
        """
        Initialize a discriminator network.
        :param batch_size: Batch size for discriminator.
        :param layer_sizes: A list with the feature maps for each MultiLayer.
        :param inner_layers: An integer indicating the number of inner layers.
        """
        self.reuse = True
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.inner_layers = inner_layers
        self.conv_layer_num = 0
        self.use_wide_connections = use_wide_connections
        self.build = True
        self.name = name
        self.is_training = False
        self.batch_norm_params = {'decay' : 0.99, 'scale' : True, 'center' : True,
                                 'is_training' : self.is_training, 'renorm' : True}
        self.current_layers = []
        
    def upscale(self, x, scale):
        """
            Upscales an image using nearest neighbour
            :param x: Input image
            :param h_size: Image height size
            :param w_size: Image width size
            :return: Upscaled image
        """
        [b, h, w, c] = [int(dim) for dim in x.get_shape()]

        return tf.image.resize_nearest_neighbor(x, (h * scale, w * scale))
        
    def encoder_block(self, x1, num): #last output, previous_output 
        #with tf.variable_scope(np.str(num)+'encoder_block'):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            num_outputs = 64, padding = 'SAME',
                            kernel_size = [3,3], stride = (1,1),
                            activation_fn = tf.nn.leaky_relu,
                            normalizer_fn=slim.batch_norm, normalizer_params= self.batch_norm_params):

            conv1_1 = slim.conv2d(x1)
            self.current_layers.append(conv1_1)
            output1_1 = tf.concat([conv1_1, x1], axis=3)

            conv1_2 = slim.conv2d(output1_1)
            self.current_layers.append(conv1_2)
            output1_2 = tf.concat([conv1_2, output1_1], axis=3)

            conv1_3 = slim.conv2d(output1_2)
            self.current_layers.append(conv1_3)
            output1_3 = tf.concat([conv1_3, output1_2], axis=3)

            conv1_4 = slim.conv2d(output1_3)
            self.current_layers.append(conv1_4)
            output1_4 = tf.concat([conv1_4, output1_3], axis=3)

            conv1_5 = slim.conv2d(output1_4, stride=(2,2))
            self.current_layers.append(conv1_5)
            output = slim.dropout(conv1_5, keep_prob=0.5)
            self.current_layers.append(output)

            if num == 3:
                pass
            else :
                input_projection = slim.conv2d(conv1_4, num_outputs=conv1_4.get_shape()[3], stride=(2,2),
                                               activation_fn= None, normalizer_fn= None)
                output = tf.concat([output, input_projection], axis=3)

        return output
    
    def __call__(self, conditional_input, generated_input, training=False, dropout_rate=0.0):
        """
        :param conditional_input: A batch of conditional inputs (x_i) of size [batch_size, height, width, channel]
        :param generated_input: A batch of generated inputs (x_g) of size [batch_size, height, width, channel]
        :param training: Placeholder for training or a boolean indicating training or validation
        :param dropout_rate: A float placeholder for dropout rate or a float indicating the dropout rate
        :param name: Network name
        :return:
        """
        conditional_input = tf.convert_to_tensor(conditional_input)
        generated_input = tf.convert_to_tensor(generated_input)
        
        
        with tf.variable_scope(self.name, reuse=self.reuse):
            concat_images = tf.concat([conditional_input, generated_input], axis=3)
            outputs = concat_images
            self.current_layers.append(outputs)
               
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        num_outputs = 64, padding = 'SAME',
                        kernel_size = [3,3], stride = (1,1),
                        activation_fn = tf.nn.leaky_relu,
                        normalizer_fn=slim.batch_norm, normalizer_params= self.batch_norm_params):
            
        #with tf.variable_scope('first_conv', reuse=self.reuse): 
            conv = slim.conv2d(outputs, stride=(2,2))
            self.current_layers.append(conv)

            input_projection = slim.conv2d(outputs, num_outputs=outputs.get_shape()[3], kernel_size = [3,3], stride=(2,2),
                                   activation_fn= None, normalizer_fn= None)
            conv1 = tf.concat([conv, input_projection], axis=3)

            en1 = self.encoder_block(conv1, 1)
            en2 = self.encoder_block(en1, 2)
            en3 = self.encoder_block(en2, 3)

        #with tf.variable_scope('decoder_block', reuse=self.reuse):
            feature_level_flatten = tf.reduce_mean(en3, axis=[1, 2])
            location_level_flatten = tf.layers.flatten(en3)

            feature_level_dense = tf.layers.dense(feature_level_flatten, units=1024, activation=tf.nn.leaky_relu)
            combo_level_flatten = tf.concat([feature_level_dense, location_level_flatten], axis=1)

        #with tf.variable_scope('discriminator_out_block', reuse=self.reuse):
            outputs = tf.layers.dense(combo_level_flatten, 1)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
        if self.build:
            print("discr layers", self.conv_layer_num)
            count_parameters(self.variables, name="discriminator_parameter_num")
        self.build = False
        
        return outputs, self.current_layers
