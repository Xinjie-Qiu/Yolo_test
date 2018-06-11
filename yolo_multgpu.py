import tensorflow as tf
import sys
import numpy as np
import datetime

datadir = '/data/pascal/val.tfrecords'

w_img = 448
h_img = 448
d_img = 3

batch_size = 1
num_preprocess_threads = 8
min_queue_examples = 2000

max_objects = 20
num_gpus = 1

cell_size = 7
num_class = 20
num_box = 2

coord_scale = 5
object_scale = 1
noobject_scale = 0.5
class_scale = 1

learning_rate = 0.0001
momentum = 0.9
max_iterators = 3000


def getfeature(dir):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([dir], capacity=4 * batch_size)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/object/xmin': tf.VarLenFeature(tf.float32),
            'image/object/ymin': tf.VarLenFeature(tf.float32),
            'image/object/xmax': tf.VarLenFeature(tf.float32),
            'image/object/ymax': tf.VarLenFeature(tf.float32),
            'image/object/classes': tf.VarLenFeature(tf.float32),
            'image/object/object_number': tf.FixedLenFeature([], tf.int64),
            'image/raw': tf.FixedLenFeature([], tf.string)
        }
    )
    xmin = features['image/object/xmin']
    ymin = features['image/object/ymin']
    xmax = features['image/object/xmax']
    ymax = features['image/object/ymax']
    classes = features['image/object/classes']
    object_number = features['image/object/object_number']
    image = tf.decode_raw(features['image/raw'], tf.float32)
    image = tf.reshape(image, [h_img, w_img, d_img])

    xmin = tf.sparse_tensor_to_dense(tf.sparse_reset_shape(xmin, [20]))
    ymin = tf.sparse_tensor_to_dense(tf.sparse_reset_shape(ymin, [20]))
    xmax = tf.sparse_tensor_to_dense(tf.sparse_reset_shape(xmax, [20]))
    ymax = tf.sparse_tensor_to_dense(tf.sparse_reset_shape(ymax, [20]))
    classes = tf.sparse_tensor_to_dense(tf.sparse_reset_shape(classes, [20]))
    label = tf.transpose([xmin, ymin, xmax, ymax, classes], [1, 0])

    # return tf.train.shuffle_batch([image, label, object_number], batch_size, batch_size * 3 + 200, 200,
    #                               num_preprocess_threads, shapes=[[448, 448, 3], [20, 5], []])
    return tf.train.shuffle_batch([image, label, object_number], batch_size, 4 * batch_size, 1 * batch_size, num_preprocess_threads,
                          shapes=[[448, 448, 3], [20, 5], []])


def build_network(image):
    temp_conv = tf.nn.leaky_relu(tf.layers.conv2d(image, 64, 7, 2, 'same'), alpha=0.1)
    temp_conv = tf.layers.max_pooling2d(temp_conv, 2, 2, 'same')

    temp_conv = tf.nn.leaky_relu(tf.layers.conv2d(temp_conv, 192, 3, 1, 'same'), alpha=0.1)
    temp_conv = tf.layers.max_pooling2d(temp_conv, 2, 2, 'same')

    temp_conv = tf.nn.leaky_relu(tf.layers.conv2d(temp_conv, 128, 1, 1, 'same'), alpha=0.1)
    temp_conv = tf.nn.leaky_relu(tf.layers.conv2d(temp_conv, 256, 3, 1, 'same'), alpha=0.1)

    temp_conv = tf.nn.leaky_relu(tf.layers.conv2d(temp_conv, 256, 1, 1, 'same'), alpha=0.1)
    temp_conv = tf.nn.leaky_relu(tf.layers.conv2d(temp_conv, 512, 3, 1, 'same'), alpha=0.1)
    temp_conv = tf.layers.max_pooling2d(temp_conv, 2, 2, 'same')

    for i in range(4):
        temp_conv = tf.nn.leaky_relu(tf.layers.conv2d(temp_conv, 256, 1, 1, 'same'), alpha=0.1)
        temp_conv = tf.nn.leaky_relu(tf.layers.conv2d(temp_conv, 512, 3, 1, 'same'), alpha=0.1)

    temp_conv = tf.nn.leaky_relu(tf.layers.conv2d(temp_conv, 512, 1, 1, 'same'), alpha=0.1)
    temp_conv = tf.nn.leaky_relu(tf.layers.conv2d(temp_conv, 1024, 3, 1, 'same'), alpha=0.1)
    temp_conv = tf.layers.max_pooling2d(temp_conv, 2, 2, 'same')

    for i in range(2):
        temp_conv = tf.nn.leaky_relu(tf.layers.conv2d(temp_conv, 512, 1, 1, 'same'), alpha=0.1)
        temp_conv = tf.nn.leaky_relu(tf.layers.conv2d(temp_conv, 1024, 3, 1, 'same'), alpha=0.1)

    temp_conv = tf.nn.leaky_relu(tf.layers.conv2d(temp_conv, 1024, 3, 1, 'same'), alpha=0.1)
    temp_conv = tf.nn.leaky_relu(tf.layers.conv2d(temp_conv, 1024, 3, 2, 'same'), alpha=0.1)

    temp_conv = tf.nn.leaky_relu(tf.layers.conv2d(temp_conv, 1024, 3, 1, 'same'), alpha=0.1)
    temp_conv = tf.nn.leaky_relu(tf.layers.conv2d(temp_conv, 1024, 3, 1, 'same'), alpha=0.1)

    temp_conv_flat = tf.reshape(temp_conv, [-1, cell_size * cell_size * 1024])

    fc_1 = tf.nn.leaky_relu(tf.layers.dense(temp_conv_flat, 4096), alpha=0.1)
    fc_1 = tf.nn.dropout(fc_1, keep_prob=0.5)
    fc_2 = tf.layers.dense(fc_1, cell_size * cell_size * (5 * num_box + num_class))
    fc_2 = tf.reshape(fc_2, [batch_size, cell_size, cell_size, (num_box * 5 + num_class)])
    return fc_2


def cond1(num, predict, labels, object_number, loss):
    return num < object_number


def body1(num, predict, labels, object_number, loss):
    label = labels[num, :]

    x_min = label[0] / (w_img / cell_size)
    y_min = label[1] / (h_img / cell_size)
    x_max = label[2] / (w_img / cell_size)
    y_max = label[3] / (h_img / cell_size)

    # this cell have object or not
    temp = [tf.ceil(y_max) - tf.floor(y_min), tf.ceil(x_max) - tf.floor(x_min)]
    label_class = tf.ones(tf.cast(temp, tf.int32))
    temp = [[tf.floor(y_min), cell_size - tf.ceil(y_max)],
            [tf.floor(x_min), cell_size - tf.ceil(x_max)]]
    label_class = tf.pad(label_class, tf.cast(temp, tf.int32), 'CONSTANT')
    label_class = tf.reshape(label_class, [cell_size, cell_size, 1])

    # this cell is object's center or not
    y_center = ((label[3] + label[1]) / 2.0) / (h_img / cell_size) + 1e-6
    x_center = ((label[2] + label[0]) / 2.0) / (w_img / cell_size) + 1e-6
    h = (label[3] - label[1]) / (w_img / cell_size)
    w = (label[2] - label[0]) / (h_img / cell_size)


    label_object = tf.ones([1, 1])
    temp = [[tf.floor(y_center), cell_size - tf.ceil(y_center)],
            [tf.floor(x_center), cell_size - tf.ceil(x_center)]]
    label_object = tf.pad(label_object, tf.cast(temp, tf.int32), 'CONSTANT')
    label_object = tf.reshape(label_object, [cell_size, cell_size, 1])

    # class_label
    C = tf.one_hot(tf.cast(label[4], tf.int32), num_class)

    class_loss = tf.nn.l2_loss((predict[:, :, 5 * num_box:] - C) * label_class) * class_scale

    predict_boxes = predict[:, :, :4 * num_box]
    predict_boxes = tf.reshape(predict_boxes, [cell_size, cell_size, num_box, 4])
    base_boxes = np.zeros([cell_size, cell_size, num_box, 4])
    for i in range(cell_size):
        for j in range(cell_size):
            base_boxes[i, j, :, 0] += j
            base_boxes[i, j, :, 1] += i
    predict_boxes = (base_boxes + predict_boxes)

    iou_truth = iou(predict_boxes, [x_min, y_min, x_max, y_max])
    max_iou = tf.reduce_max(iou_truth, 2, keep_dims=True)
    I = tf.cast(iou_truth >= max_iou, tf.float32) * label_object
    no_I = tf.ones_like(I) - I

    object_loss = tf.nn.l2_loss(I * (predict[:, :, 4 * num_box:4 * num_box + num_box] - iou_truth)) * object_scale
    noobject_loss = tf.nn.l2_loss(no_I * predict[:, :, 4 * num_box:4 * num_box + num_box]) * noobject_scale
    # object_loss = tf.constant(0, tf.float32)
    # noobject_loss = tf.constant(0, tf.float32)
    coord_loss = (
                         tf.nn.l2_loss((predict_boxes[:, :, :, 0] - x_center) * I)
                         + tf.nn.l2_loss((predict_boxes[:, :, :, 1] - y_center) * I)
                         + tf.nn.l2_loss((predict_boxes[:, :, :, 2] - w) * I)
                         + tf.nn.l2_loss((predict_boxes[:, :, :, 3] - h) * I)
                 ) * coord_scale
    # coord_loss = tf.constant(0, tf.float32)
    return num + 1, predict, labels, object_number, [loss[0] + class_loss, loss[1] + object_loss,
                                                     loss[2] + noobject_loss, loss[3] + coord_loss]


def iou(predicts, label):
    """calculate ious
            Args:
              predicts: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
              label: 1-D tensor [4] ===> (x_min, y_min, x_max, y_max)
            Return:
              iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            """
    predicts = tf.stack(
        [predicts[:, :, :, 0] - predicts[:, :, :, 2] / 2, predicts[:, :, :, 1] - predicts[:, :, :, 3] / 2,
         predicts[:, :, :, 0] + predicts[:, :, :, 2] / 2, predicts[:, :, :, 1] + predicts[:, :, :, 3] / 2])
    predicts = tf.transpose(predicts, [1, 2, 3, 0])

    lu = tf.maximum(predicts[:, :, :, 0:2], label[0:2])
    rd = tf.minimum(predicts[:, :, :, 2:], label[2:])

    intersection = rd - lu

    squrt = intersection[:, :, :, 0] * intersection[:, :, :, 1]
    mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)

    inter_squrt = squrt * mask

    squrt_1 = (predicts[:, :, :, 2] - predicts[:, :, :, 0]) * (predicts[:, :, :, 3] - predicts[:, :, :, 1])
    squrt_2 = (label[2] - label[0]) * (label[3] - label[1])

    return inter_squrt / (squrt_1 + squrt_2 - inter_squrt + 1e-6)


def loss(predicts, labels, objects_number):
    class_loss = tf.constant(0, tf.float32)
    object_loss = tf.constant(0, tf.float32)
    noobject_loss = tf.constant(0, tf.float32)
    coord_loss = tf.constant(0, tf.float32)
    loss = [0, 0, 0, 0]

    for j in range(batch_size):
        predict = predicts[j, :, :, :]
        label = labels[j, :, :]
        object_number = objects_number[j]
        object_number = tf.cast(object_number, tf.int32)

        tuple_result = tf.while_loop(cond1, body1, [tf.constant(0), predict, label, object_number,
                                                    [class_loss, object_loss, noobject_loss, coord_loss]])
        for k in range(4):
            loss[k] += tuple_result[4][k]
    tf.summary.scalar('class_loss', loss[0] / batch_size)
    tf.summary.scalar('object_loss', loss[1] / batch_size)
    tf.summary.scalar('noobject_loss', loss[2] / batch_size)
    tf.summary.scalar('coord_loss', loss[3] / batch_size)
    losses = tf.add_n(loss) / batch_size

    return losses


def main(argvs):
    with tf.device('/cpu'):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        image, labels, objects_number = getfeature(datadir)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        opt = tf.train.MomentumOptimizer(learning_rate, momentum)
        tower_gradient = []
        with tf.device('/gpu:1'):
            predicts = build_network(image)
            total_loss = loss(predicts, labels, objects_number)
            gradient = opt.compute_gradients(total_loss)
            tower_gradient.append(gradient)
        apply_op = opt.apply_gradients(tower_gradient[0])
        saver = tf.train.Saver()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        # saver.restore(sess, './model_test/mymodel.ckpt')

        tf.summary.scalar('loss', total_loss)
        merged = tf.summary.merge_all()
        logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

        writer = tf.summary.FileWriter(logdir, sess.graph)

        print('complete')

        for i in range(max_iterators):
            sess.run(apply_op)
            if i % 100 == 0:
                summmary, losses = sess.run([merged, total_loss])
                writer.add_summary(summmary, global_step=i)
                print('step {}, loss = {}'.format(i, losses))

        saver.save(sess, './model_test/mymodel.ckpt')
        print('yea')
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main(sys.argv)
