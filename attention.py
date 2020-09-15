import tensorflow as tf

class SinmpleAttention(tf.keras.models.Model):
    '''Attentionの説明をするための、Multi-headではない単純なAttention
    '''
    def __init__(self, depth: int, *args, **kwargs):
        '''コンストラクタ
        :param depth: 隠れ層及出力の次元
        '''
        super().__init__(*args, **kwargs)
        self.depth = depth
        
        self.q_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name='q_dense_layer')
        self.k_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name='k_dense_layer')
        self.v_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name='v_dense_layer')
        self.output_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name='output_dense_layer')

    def call(self, input: tf.Tensor, memory: tf.Tensor):
        '''モデルの実行を行う
        Args:
            input: queryのテンソル
            memory: queryに情報を与えるmemoryのテンソル
        Returns:
            tf.Tensor
        '''
        q = self.q_dense_layer(input)  # [batch_size, q_length, depth]
        k = self.k_dense_layer(memory) # [batch_size, m_length, depth]
        v = self.v_dense_layer(memory)

        # qとkの内積をとって、queryとkeyの関連度のようなものを計算
        logit = tf.matmul(q, k, transpose_b=True) # [batch_size, q_length, k_length]

        # softmaxをとって正規化
        attention_weight = tf.nn.softmax(logit, name='attention_weight')

        # 重みに従ってvalueから情報を引いていく
        attention_output = tf.matmul(attention_weight, v) # [batch_size, q_length, depth]
        return self.output_dense_layer(attention_output)

class MultiheadAttention(tf.keras.models.Model):
    '''Multi-head Attentionのモデル
    model = MultiheadAttention(
        hidden_dim = 512,
        head_num = 8,
        dropout_rate = 0.1,
    )
    model(query, memory, mask, training=True)
    '''
    def __init__(self, hidden_dim: int, head_num: int, dropout_rate: float, *args, **kwargs):
        '''コンストラクタ
        Args:
            hidden_dum: 隠れ層及び出力の次元
                head_numの倍数である必要がある
            head_num: ヘッドの数
            dropout_rate: ドロップアウトする確率
        '''
        super().__init__(*args, **kwargs)
        self.hidden_dim   = hidden_dim
        self.head_num     = head_num
        self.dropout_rate = dropout_rate

        self.q_dense_layer = tf.keras.layerd.Dense(hidden_dim, use_bias=False, name='q_dense_layer')
        self.k_dense_layer = tf.keras.layerd.Dense(hidden_dim, use_bias=False, name='k_dense_layer')
        self.v_dense_layer = tf.keras.layerd.Dense(hidden_dim, use_bias=False, name='v_dense_layer')
        self.output_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='output_dense_layer')
        self.attention_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input: tf.Tensor, memory: tf.Tensor, attention_mask: tf.Tensor, training: bool,):
    '''モデルの実装
    Args:
        input: queryのテンソル
        memory: queryに情報を与えるmemoryのテンソル
        attention_mask: attention weightに適用されるmask
            shape = [batch_size, 1, q_length, k_length]
            pad等無視する部分がTrueとなるようなものを指定してください
        training: 学習直推論直のフラグ
    Returns:
        tf.Tensor
    '''
    q = self.q_dense_layer(input)  # [batch_size, q_length, hidden_dim]
    k = self.q_dense_layer(memory) # [batch_size, m_length, hidden_dim]
    v = self.q_dense_layer(memory)

    q = self._split_head(q) # [batch_size, head_num, q_length, hidden_dim/head_num]
    k = self._split_head(k) # [batch_size, head_num, q_length, hidden_dim/head_num]
    v = self._split_head(v) # [batch_size, head_num, q_length, hidden_dim/head_num]

    depth = self.hidden_dim // self.head_num
    q *= depth ** -0.5 # for scaled dot production

    # qとkの内積をとることで、queryとkeyの関連度のようなものを計算
    logit = tf.matmul(q, k, transpose_b=True) # [batch_size, head_num, q_length]
    logit += tf.to_float(attention_mask) * input.dtype.min # maskはpad部分などが1、他は0

    # softmaxで正規化
    attention_weight = tf.nn.softmax(logitm name='attention_weight')
    attention_weight = self.attention_dropout_layer(attention_weight, training=training)

    # 重みに従ってvalueから情報を引く
    attention_output = tf.matmul(attention_weight, v) # [batch_size, head_num, q_length, hidden_dim / head_num]
    attention_output = self._combine_head(attention_output) # [batch_sizem q_length, hidden_dim]
    return self.output_dense_layer(attention_output)

    def _split_head(self, x: tf.Tensor):
        '''入力のtensorとhidden_dimの次元をいくつかのヘッドに分割する
        Args:
            tf.Tensor: [batch_size, length, hidden_dim]
        Returns:
            tf.Tensor: [batch_size, head_num, length, hidden_dim / head_num]
        ''' 
        with tf.name_scope('split_head'):
            batch_size, length, hidden_dim = tf.unstack(tf.shape(x))
            x = tf.reshape(x, [batch_size, length, self.head_num, self.hidden_dim // self.head_num])
            return tf.transpose(x, [0, 2, 1, 3])

    def _combine_head(self, x: tf.Tensor):
        '''入力とtensorの各ヘッドを結合する _split_headの逆変換
        Args:
            tf.Tensor: [batch_size, head_num, length, hidden_dim//head_num]
        Returns:
            tf.Tensor: [batch_size, length, hidden_dim]
        '''
        with tf.name_scope('combine_head'):
            batch_size, _, length, _ = tf,unstack(tf.shape(x))
            x = tf.transpose(x, [0, 2, 1, 3])
            return tf.reshape(x, [batch_size, length, self.hidden_dim])

class SelfAttention(MultiheadAttention):
    def call(self, input: tf.Tensor, attention_mask: tf.Tensor, training: bool):
        return super().call(
            input = input,
            memory = memory,
            attention_mask =attention_mask,
            training = training,
            )