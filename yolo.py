import tensorflow as tf
import numpy as np
import time
import cv2

Cell = 7
Boxes = 2
Class = 20

image = tf.placeholder(tf.float32, [None, 448, 448, 3])

temp_conv = tf.layers.conv2d(image, 64, 7, 2, 'same')
temp_conv = tf.layers.max_pooling2d(temp_conv, 2, 2, 'same')

temp_conv = tf.layers.conv2d(temp_conv, 192, 3, 1, 'same')
temp_conv = tf.layers.max_pooling2d(temp_conv, 2, 2, 'same')

temp_conv = tf.layers.conv2d(temp_conv, 128, 1, 1, 'same')
temp_conv = tf.layers.conv2d(temp_conv, 256, 3, 1, 'same')

temp_conv = tf.layers.conv2d(temp_conv, 256, 1, 1, 'same')
temp_conv = tf.layers.conv2d(temp_conv, 512, 3, 1, 'same')
temp_conv = tf.layers.max_pooling2d(temp_conv, 2, 2, 'same')

for i in range(4):
    temp_conv = tf.layers.conv2d(temp_conv, 256, 1, 1, 'same')
    temp_conv = tf.layers.conv2d(temp_conv, 512, 3, 1, 'same')

temp_conv = tf.layers.conv2d(temp_conv, 512, 1, 1, 'same')
temp_conv = tf.layers.conv2d(temp_conv, 1024, 3, 1, 'same')
temp_conv = tf.layers.max_pooling2d(temp_conv, 2, 2, 'same')

for i in range(2):
    temp_conv = tf.layers.conv2d(temp_conv, 512, 1, 1, 'same')
    temp_conv = tf.layers.conv2d(temp_conv, 1024, 3, 1, 'same')

temp_conv = tf.layers.conv2d(temp_conv, 1024, 3, 1, 'same')
temp_conv = tf.layers.conv2d(temp_conv, 1024, 3, 2, 'same')

temp_conv = tf.layers.conv2d(temp_conv, 1024, 3, 1, 'same')
temp_conv = tf.layers.conv2d(temp_conv, 1024, 3, 1, 'same')

temp_conv_flat = tf.reshape(temp_conv, [-1, Cell * Cell * 1024])


fc_1 = tf.nn.leaky_relu(tf.layers.dense(temp_conv_flat, 4096), alpha=0.1)
fc_1 = tf.nn.dropout(fc_1, keep_prob=0.5)
fc_2 = tf.layers.dense(fc_1, Cell * Cell * (Class + 5 * Boxes))
predicts = tf.reshape(fc_2, [-1, Cell, Cell, Class + 5 * Boxes])

config = tf.ConfigProto(log_device_placement=True)
# config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

img = cv2.imread('./test/person.jpg')
s = time.time()
h_img, w_img, _ = img.shape
img_resized = cv2.resize(img, (448, 448))
img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
img_resized_np = np.asarray( img_RGB )
inputs = np.zeros((1,448,448,3),dtype='float32')
inputs[0] = (img_resized_np/255.0)*2.0-1.0

session.run(init_op)
result = session.run(predicts, {image: inputs})
print(result)