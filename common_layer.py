import tensorflow as tf

class FeedForwardNetwork(tf.keras.models.Model):
    '''Transformer用のPosition-wise Feed forward Neural Network
    '''
    def __init__(self, hidden_dim, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim   = hidden_dim
        self.dropout_rate = dropout_rate

        self.filter_dense_layer = tf.keras.layers.Dense(hidden_dim * 4, use_bias=True,
                                                        activation=tf.nn.relu, name='filter_layer')
        self.output_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=True, name='output_layer')
        self.dropout_layer = tf.keras.Dropout(dropout_rate)

    def call(self, input, training):
        '''FeedForwardNetworkを適用

        Args:
            input (tf.Tensor): shape = [batch_size, lengthm hidden_dim]
            training (bool):

        Returns:
            shape = [batch_size, length, hidden_dim]
        '''
        tensor = self.filter_dense_layer(input)
        tensor = self.dropout_layer(tensor, training=training)
        return self.output_dense_layer(tensor)

class ResidualNormalizationWrapper(tf.keras.models.Model):
    '''与えられたレイヤー(もしくはモデル)に対して、下記のノーマライゼーションを行う

    - Layer Normalization  
    - Dropout
    - Residual Connection
    '''
    def __init__(self, layer, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = layer
        self.layer_normalization = LayerNormalization()
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input: tf.Tensor, training: bool, *args, **kwargs):
        tensor = self.layer_normalization(input)
        tensor = self.layer(tensor, training=training, *args, **kwargs)
        tensor = self.dropout_layer(tensor, training=training)
        return input + tensor

class LayerNormalization(tf.keras.layers.Layer):
    '''レイヤーノーマライゼーション
    
    レイヤーの出力が平均bias、標準偏差scaleになるように調整する
    '''
    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        self.scale = self.add_weight('layer_norm_scale', shape=[hidden_dim],
                                    initializer=tf.ones_initializer())
        self.bias = self.add_weight('layer_norm_bias', shape=[hidden_dim],
                                    initializer=tf.zeros_initializer())
        super().build(input_shape)

    def call(self, x, epsilon: float=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_meana(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)

        return norm_x * self.scale + self.bias