import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalAveragePooling1D, Conv2D, MaxPooling2D
from keras.models import Model
import tensorflow as tf

class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=8, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)
        self.projection_dim = self.embed_dim // self.num_heads

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        attention, _ = self.attention(query, key, value)
        attention = tf.reshape(attention, (batch_size, seq_len, self.embed_dim))
        output = self.combine_heads(attention)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

from keras.layers import LayerNormalization, Dropout, Layer, Dense
from keras.models import Sequential

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ffn_dim, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = Sequential([
            Dense(ffn_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate=0.01)
        self.dropout2 = Dropout(rate=0.01)

    def call(self, inputs, training):
        attention = self.att(inputs)
        attention = self.dropout1(attention, training=training)
        out1 = self.layernorm1(inputs + attention)
        out2 = self.ffn(out1)
        out2 = self.dropout2(out2, training=training)
        return self.layernorm2(out1 + out2)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

from keras.layers import Layer, Dense

class PatchEmbedding(Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.projection = Dense(embed_dim)

    def call(self, patches):
        return self.projection(patches)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "embed_dim": self.embed_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class VisionImageTransformer(Model):
    def __init__(self, num_patches, num_heads, embed_dim, ffn_dim, num_layers, num_classes, **kwargs):
        super(VisionImageTransformer, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.patch_embed = PatchEmbedding(num_patches, embed_dim)
        self.transformer_layers = [
            TransformerBlock(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ]

        self.flatten = Flatten()
        self.conv2D_layer1 = Conv2D(16, (3, 3), strides=(1, 1), activation='relu')
        self.mpool_layer1 = MaxPooling2D(pool_size=(2, 2))
        self.conv2D_layer2 = Conv2D(16, (3, 3), strides=(1, 1), activation='relu')
        self.mpool_layer2 = MaxPooling2D(pool_size=(2, 2))
        self.dense = Dense(1, activation='sigmoid')
        self.pool = GlobalAveragePooling1D()
        self.dropout = Dropout(rate=0.5)

    def call(self, inputs, training=False):
        inp_preprocessed = self.conv2D_layer1(inputs)
        inp_preprocessed = self.mpool_layer1(inp_preprocessed)
        inp_preprocessed = self.conv2D_layer2(inp_preprocessed)
        inp_preprocessed = self.mpool_layer2(inp_preprocessed)

        patches = tf.image.extract_patches(
            images=inp_preprocessed,
            sizes=[1, 16, 16, 1],
            strides=[1, 16, 16, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'
        )
        batch_size = tf.shape(patches)[0]
        patches = tf.reshape(patches, [batch_size, -1, tf.shape(patches)[-1]])

        x = self.patch_embed(patches)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, training=training)

        x = self.dropout(x, training=training)
        x = self.pool(x)
        return self.dense(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "num_heads": self.num_heads,
            "embed_dim": self.embed_dim,
            "ffn_dim": self.ffn_dim,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
