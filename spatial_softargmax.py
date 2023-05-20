import tensorflow as tf
from keras.layers import Layer
from keras.utils import tf_utils


class Spatial_SoftArgmax(Layer):

    def __init__(self):
        super(Spatial_SoftArgmax, self).__init__()
    
    def build(self, input_shape):
        self.temperature = self.add_weight(name="temperature",
                                           shape=(),
                                           dtype=tf.float32,
                                           initializer=tf.keras.initializers.ones(),
                                           trainable=True)
        super(Spatial_SoftArgmax, self).build(input_shape) 
    
    def call(self, x):
        _, height, width, channels = x.shape
        
        # add temperature coefficient
        x = x/self.temperature

        # Flatten the feature map
        flattened_map = tf.reshape(x, (-1, height * width, channels))

        # Apply softmax along the spatial dimensions
        softmax_map = tf.nn.softmax(flattened_map, axis=1)

        # make grid
        x_coords, y_coords = tf.meshgrid(
            tf.linspace(0., tf.cast(height - 1, tf.float32), num=height),
            tf.linspace(0., tf.cast(width - 1, tf.float32), num=width),
            indexing='ij')

        x_coords = tf.reshape(x_coords, [1, height * width, 1])
        y_coords = tf.reshape(y_coords, [1, height * width, 1])

        # Calculate the x-coordinate and y-coordinate separately
        x = tf.reduce_sum(softmax_map * x_coords, axis=1)
        y = tf.reduce_sum(softmax_map * y_coords, axis=1)

        # Normalize the coordinates to [0, 1]
        x /= tf.cast(width - 1, tf.float32)
        y /= tf.cast(height - 1, tf.float32)

        # Stack the x_y coordinates
        coordinates = tf.stack([x, y], axis=-1)
        coordinates = tf.reshape(coordinates, [-1, channels * 2])

        return coordinates

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3] * 2
