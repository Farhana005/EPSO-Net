import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv3D, Conv3DTranspose, MaxPooling3D, Dropout,
    GlobalAveragePooling3D, Add, Multiply, Dense, Reshape, Layer,
    Activation, LayerNormalization, AveragePooling3D, UpSampling3D, LeakyReLU
)
from tensorflow.keras.models import Model
from keras.utils import register_keras_serializable

# --- Custom GroupNormalization (drop-in replacement, no tf-addons) ---
class GroupNormalization(Layer):
    def __init__(self, groups=1, axis=-1, epsilon=1e-5, **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None or dim % self.groups != 0:
            raise ValueError(f"Channel dimension {dim} not divisible by groups {self.groups}")
        self.gamma = self.add_weight(name='gamma', shape=(dim,), initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=(dim,), initializer='zeros', trainable=True)
        super(GroupNormalization, self).build(input_shape)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        # reshape to [N, D, H, W, groups, C_per_group]
        reshaped = tf.reshape(
            inputs,
            [-1,
             input_shape[1],
             input_shape[2],
             input_shape[3],
             self.groups,
             input_shape[4] // self.groups]
        )
        mean, variance = tf.nn.moments(reshaped, axes=[1, 2, 3, 5], keepdims=True)
        normalized = (reshaped - mean) / tf.sqrt(variance + self.epsilon)
        normalized = tf.reshape(normalized, tf.shape(inputs))
        return self.gamma * normalized + self.beta

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"groups": self.groups, "axis": self.axis, "epsilon": self.epsilon})
        return cfg

# --- Depthwise Separable Conv3D ---
@register_keras_serializable()
class DepthwiseSeparableConv3D(Layer):
    def __init__(self, filters, kernel_size, strides=(1,1,1), padding='same',
                 depth_multiplier=1, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.depth_multiplier = depth_multiplier
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        channel_dim = input_shape[-1]
        self.depthwise_conv = Conv3D(
            filters=channel_dim * self.depth_multiplier,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            groups=channel_dim,
            use_bias=False
        )
        self.pointwise_conv = Conv3D(
            filters=self.filters,
            kernel_size=(1,1,1),
            strides=(1,1,1),
            padding='same',
            use_bias=True
        )

    def call(self, inputs):
        x = self.depthwise_conv(inputs)
        x = self.pointwise_conv(x)
        return self.activation(x) if self.activation is not None else x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "depth_multiplier": self.depth_multiplier,
            "activation": tf.keras.activations.serialize(self.activation),
        })
        return cfg


# assume GroupNormalization and DepthwiseSeparableConv3D are defined elsewhere


def UTSA_module(x, filters, num_layers, activation, dropout_rate, pooling_type):
    skip_connections = []
    if len(x.shape) == 4:
        x = tf.expand_dims(x, axis=1)
    for _ in range(num_layers):
        residual = Conv3D(x.shape[-1], 1, padding='same')(x)
        residual = activation()(residual) if isinstance(activation, type) else activation(residual)
        residual = GroupNormalization(groups=1)(residual)
        x = Add()([x, residual])
        skip_connections.append(x)
        x = pooling_type(pool_size=(2,2,2), strides=2, padding='same')(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        filters *= 2
    return x, skip_connections

def Astra_module(x, filters, num_layers, dilation_rate, activation, dropout_rate, pooling_type):
    skip_connections = []
    for _ in range(num_layers):
        shortcut = x
        ch = x.shape[-1]
        dilated = Conv3D(ch, 3, dilation_rate=dilation_rate, padding='same')(x)
        dilated = Conv3D(ch, 1, padding='same', activation='relu')(dilated)
        se = GlobalAveragePooling3D()(shortcut)
        se = Dense(ch//16, activation='relu')(se)
        se = Dense(ch, activation='sigmoid')(se)
        se = Reshape((1,1,1,ch))(se)
        fused = Multiply()([dilated, se])
        out = Add()([fused, shortcut])
        out = LayerNormalization()(out)
        x = activation()(out) if isinstance(activation, type) else activation(out)
        skip_connections.append(x)
        x = pooling_type(pool_size=(2,2,2), strides=2, padding='same')(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        filters *= 2
    return x, skip_connections

def Revo_module(x, utsa_skips, astra_skips, filters, num_layers,
                activation, dropout_rate, aggregation_method):
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import (
        Conv3D, Conv3DTranspose, AveragePooling3D, UpSampling3D,
        Add, Multiply, Activation, Dropout
    )

    for i in reversed(range(num_layers)):
        # 1) upsample decoder state
        x = Conv3DTranspose(filters, 3, strides=2, padding='same')(x)
        x = activation()(x) if isinstance(activation, type) else activation(x)

        # 2) grab skips
        skip_u = utsa_skips[i]
        skip_a = astra_skips[i]
        target = K.int_shape(x)[1:4]

        # 3) align skip_u to target
        while K.int_shape(skip_u)[1:4] != target:
            src = K.int_shape(skip_u)[1:4]
            if all(s >= t for s, t in zip(src, target)):
                skip_u = AveragePooling3D(pool_size=2, padding='same')(skip_u)
            else:
                skip_u = UpSampling3D(size=2)(skip_u)

        # 4) align skip_a to target
        while K.int_shape(skip_a)[1:4] != target:
            src = K.int_shape(skip_a)[1:4]
            if all(s >= t for s, t in zip(src, target)):
                skip_a = AveragePooling3D(pool_size=2, padding='same')(skip_a)
            else:
                skip_a = UpSampling3D(size=2)(skip_a)

        # 5) project + attention
        skip_u_proj = Conv3D(filters//2, 1, padding='same')(skip_u)
        skip_a_proj = Conv3D(filters//2, 1, padding='same')(skip_a)
        dec_proj    = Conv3D(filters//2, 1, padding='same')(x)

        combined    = Add()([skip_u_proj, skip_a_proj])
        fusion_pre  = Add()([combined, dec_proj])
        act         = Activation('relu')(fusion_pre)
        attn        = Activation('sigmoid')(
                          Conv3D(1, 1, padding='same')(act))

        # 6) gate & fuse
        gated_u    = Multiply()([skip_u, attn])
        gated_a    = Multiply()([skip_a, attn])
        gated_all  = Add()([gated_u, gated_a])
        x = aggregation_method()([x, gated_all])

        # 7) post‐fusion conv + act + drop
        x = Conv3D(filters, 3, padding='same')(x)
        x = activation()(x) if isinstance(activation, type) else activation(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    return x


def best_model(input_shape, num_layers, dilation_rate, filters, kernel_size, activation,
               dropout_rate, n_classes, conv_type, norm_type, pooling_type, aggregation_method):

    inputs = Input(shape=input_shape)
    
    # --- Encoder Stage: Initial Projection ---
    x = conv_type(filters, kernel_size, strides=2, padding='same')(inputs)
    x = activation()(x) if isinstance(activation, type) else activation(x)
    x = GroupNormalization(groups=1)(x)

    # --- Encoder Stage 1: UTSA ---
    x, skips1 = UTSA_module(
        x, filters, num_layers, activation, dropout_rate, pooling_type
    )

    # Transition conv after UTSA
    x = DepthwiseSeparableConv3D(filters, 3, padding='same')(x)
    x = activation()(x) if isinstance(activation, type) else activation(x)

    # --- Encoder Stage 2: Astra ---
    x, skips2 = Astra_module(
        x, filters, num_layers, dilation_rate, activation, dropout_rate, pooling_type
    )

    # --- Bottleneck ---
    # Transition before bottleneck
    x = DepthwiseSeparableConv3D(filters, 1, padding='same')(x)
    x = activation()(x) if isinstance(activation, type) else activation(x)
    # Main bottleneck conv
    x = DepthwiseSeparableConv3D(filters, 3, padding='same')(x)
    x = activation()(x) if isinstance(activation, type) else activation(x)
    x = Dropout(dropout_rate)(x)

    # Squeeze-and-Excitation block in bottleneck
    se = GlobalAveragePooling3D()(x)
    se = Dense(filters // 4, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, 1, filters))(se)
    x = Multiply()([x, se])

    # --- Decoder Stage: Revo ---
    x = Revo_module(
        x, skips1, skips2,
        filters, num_layers,
        activation, dropout_rate,
        aggregation_method
    )

    # Final upsampling layers
    for i in range(5):
        x = Conv3DTranspose(filters, 2, strides=2, padding='same')(x)
        x = activation()(x) if isinstance(activation, type) else activation(x)

    # Last upsampling marked for GradCAM
    x = DepthwiseSeparableConv3D(filters, 1, padding='same', name='conv_last')(x)
    x = activation()(x) if isinstance(activation, type) else activation(x)

    # --- Output Stage ---
    x = norm_type()(x)
    outputs = conv_type(n_classes, (1, 1, 1), activation='softmax')(x)

    return Model(inputs, outputs, name='best_model')

