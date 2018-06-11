import tensorflow as tf
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt

datadir = '/data/pascal/train.tfrecords'

num_preprocess_threads = 8

batch_size = 16
cell_size = 7
num_class = 20
num_box = 2

w_img = 448
h_img = 448
d_img = 3

threshold = 0.2
iou_threshold = 0.5

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


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


def iou(predicts, label):
    """calculate ious
            Args:
              predicts: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
              label: 1-D tensor [4] ===> (x_min, y_min, x_max, y_max)
            Return:
              iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            """
    predicts = np.stack(
        [predicts[:, :, :, 0] - predicts[:, :, :, 2] / 2, predicts[:, :, :, 1] - predicts[:, :, :, 3] / 2,
         predicts[:, :, :, 0] + predicts[:, :, :, 2] / 2, predicts[:, :, :, 1] + predicts[:, :, :, 3] / 2], axis=3)

    lu = np.maximum(predicts[:, :, :, 0:2], label[0:2])
    rd = np.minimum(predicts[:, :, :, 2:], label[2:])

    intersection = rd - lu

    squrt = intersection[:, :, :, 0] * intersection[:, :, :, 1]
    mask = np.float32(intersection[:, :, :, 0] > 0) * np.float32(intersection[:, :, :, 1] > 0)

    inter_squrt = squrt * mask

    squrt_1 = (predicts[:, :, :, 2] - predicts[:, :, :, 0]) * (predicts[:, :, :, 3] - predicts[:, :, :, 1])
    squrt_2 = (label[2] - label[0]) * (label[3] - label[1])

    return inter_squrt / (squrt_1 + squrt_2 - inter_squrt + 1e-6)


def build_network(image, keep_prob):
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
    fc_1 = tf.nn.dropout(fc_1, keep_prob)
    fc_2 = tf.layers.dense(fc_1, cell_size * cell_size * (5 * num_box + num_class))
    fc_2 = tf.reshape(fc_2, [batch_size, cell_size, cell_size, (num_box * 5 + num_class)])
    return fc_2


def show_result(img, results):
    img_cp = (img.copy() + 1) / 2
    for i in range(len(results)):
        x = int(results[i][1])
        y = int(results[i][2])
        w = int(results[i][3]) // 2
        h = int(results[i][4]) // 2
    cv2.rectangle(img_cp, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
    # cv2.rectangle(img_cp, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
    cv2.putText(img_cp, results[i][0] + ' : %.2f' % results[i][5], (x - w + 5, y - h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imshow('YOLO detection', img_cp)
    cv2.waitKey(10000)
    print('yea')


def interpret_output(output, label):
    probs = np.zeros([cell_size, cell_size, num_box, num_class])
    class_probs = np.reshape(output[:, :, 5 * num_box:], [cell_size, cell_size, num_class])
    scales = np.reshape(output[:, :, 4 * num_box:4 * num_box + num_box], [cell_size, cell_size, num_box])

    predict_boxes = output[:, :, :4 * num_box]
    predict_boxes = np.reshape(predict_boxes, [cell_size, cell_size, num_box, 4])
    base_boxes = np.zeros([cell_size, cell_size, num_box, 4], np.float32)
    for i in range(cell_size):
        for j in range(cell_size):
            base_boxes[i, j, :, 0] += j
            base_boxes[i, j, :, 1] += i
    predict_boxes = (base_boxes + predict_boxes)

    label = label[0]
    x_min = label[0] / (w_img / cell_size)
    y_min = label[1] / (h_img / cell_size)
    x_max = label[2] / (w_img / cell_size)
    y_max = label[3] / (h_img / cell_size)

    y_center = ((label[3] + label[1]) / 2.0) / (h_img / cell_size) + 1e-6
    x_center = ((label[2] + label[0]) / 2.0) / (w_img / cell_size) + 1e-6
    h = (label[3] - label[1]) / (w_img / cell_size)
    w = (label[2] - label[0]) / (h_img / cell_size)


    print([x_min, y_min, x_max, y_max])
    iou_truth = iou(predict_boxes, [x_min, y_min, x_max, y_max])

    predict_boxes[:, :, :, 0:4] = predict_boxes[:, :, :, 0:4] / cell_size
    predict_boxes[:, :, :, 0] *= w_img
    predict_boxes[:, :, :, 1] *= h_img
    predict_boxes[:, :, :, 2] *= w_img
    predict_boxes[:, :, :, 3] *= h_img

    for i in range(2):
        for j in range(20):
            probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

    filter_mat_probs = np.array(probs >= threshold, dtype='bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)
    boxes_filtered = predict_boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[
        filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

    argsort = np.array(np.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort]
    probs_filtered = probs_filtered[argsort]
    classes_num_filtered = classes_num_filtered[argsort]

    for i in range(len(boxes_filtered)):
        if probs_filtered[i] == 0: continue
        for j in range(i + 1, len(boxes_filtered)):
            if iou(boxes_filtered[i], boxes_filtered[j]) > iou_threshold:
                probs_filtered[j] = 0.0

    filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]
    print('yea')

    result = []
    for i in range(len(boxes_filtered)):
        result.append([classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[i][1],
                       boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

    return result


def main(argvs):
    with tf.device('/cpu'):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        image, labels, objects_number = getfeature(datadir)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        keep_prob = tf.constant(1, tf.float32)
        with tf.device('/gpu:1'):
            predicts = build_network(image, keep_prob)

        saver = tf.train.Saver()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        saver.restore(sess, './model/mymodel.ckpt')

        net_output, label, images = sess.run([predicts, labels, image])
        result = interpret_output(net_output[0], label[0])
        print(result)
        show_result(images[0], result)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main(sys.argv)
