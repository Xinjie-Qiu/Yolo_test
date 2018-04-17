import tensorflow as tf
import numpy as np
import time
import cv2
import sys
import random
from queue import Queue
from threading import Thread


class Yolo:
    fromfile = 'test/person.jpg'
    tofile_img = 'test/output.jpg'
    tofile_txt = 'test/output.txt'
    imshow = True
    filewrite_img = False
    filewrite_txt = False
    disp_console = True
    weights_file = 'weights/YOLO_tiny.ckpt'
    alpha = 0.1
    threshold = 0.2
    iou_threshold = 0.5
    num_class = 20
    num_box = 2
    grid_size = 7
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    data_path = 'data/pascal_voc.txt'
    batch_size = 16

    thread_num = 5
    max_objects = 20

    image_size = 448
    cell_size = 7

    w_img = 448
    h_img = 448

    object_scale = 1
    noobject_scale = 0.5
    class_scale = 1
    coord_scale = 5

    learning_rate = 1e-6
    max_iterators = 100

    def __init__(self, argvs=[]):
        # record and image_label queue
        self.record_queue = Queue(maxsize=10000)
        self.image_label_queue = Queue(maxsize=512)
        self.loaddata()
        # self.argv_parser(argvs)
        self.training()
        self.testing()

    def argv_parser(self, argvs):
        for i in range(1, len(argvs), 2):
            if argvs[i] == '-fromfile': self.fromfile = argvs[i + 1]
            if argvs[i] == '-tofile_img': self.tofile_img = argvs[i + 1]; self.filewrite_img = True
            if argvs[i] == '-tofile_txt': self.tofile_txt = argvs[i + 1]; self.filewrite_txt = True
            if argvs[i] == '-imshow':
                if argvs[i + 1] == '1':
                    self.imshow = True
                else:
                    self.imshow = False
            if argvs[i] == '-disp_console':
                if argvs[i + 1] == '1':
                    self.disp_console = True
                else:
                    self.disp_console = False

    def build_networks(self, image):

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

        temp_conv_flat = tf.reshape(temp_conv, [-1, self.grid_size * self.grid_size * 1024])

        fc_1 = tf.nn.leaky_relu(tf.layers.dense(temp_conv_flat, 4096), alpha=0.1)
        fc_1 = tf.nn.dropout(fc_1, keep_prob=0.5)
        fc_2 = tf.layers.dense(fc_1, self.grid_size * self.grid_size * (self.num_class + 5 * self.num_box))
        return fc_2

    def detect_from_cvmat(self, img):
        s = time.time()
        self.h_img, self.w_img, _ = img.shape
        img_resized = cv2.resize(img, (448, 448))
        img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_resized_np = np.asarray(img_RGB)
        inputs = np.zeros((1, 448, 448, 3), dtype='float32')
        inputs[0] = (img_resized_np / 255.0) * 2.0 - 1.0
        in_dict = {self.x: inputs}
        net_output = self.sess.run(self.fc_2, feed_dict=in_dict)
        self.result = self.interpret_output(net_output[0])
        self.show_results(img, self.result)
        strtime = str(time.time() - s)
        if self.disp_console: print('Elapsed time : ' + strtime + ' secs' + '\n')

    def detect_from_file(self, filename):
        if self.disp_console: print('Detect from ' + filename)
        img = cv2.imread(filename)
        # img = misc.imread(filename)
        self.detect_from_cvmat(img)

    def detect_from_crop_sample(self):
        self.w_img = 640
        self.h_img = 420
        f = np.array(open('person_crop.txt', 'r').readlines(), dtype='float32')
        inputs = np.zeros((1, 448, 448, 3), dtype='float32')
        for c in range(3):
            for y in range(448):
                for x in range(448):
                    inputs[0, y, x, c] = f[c * 448 * 448 + y * 448 + x]

        in_dict = {self.x: inputs}
        net_output = self.sess.run(self.fc_19, feed_dict=in_dict)
        self.boxes, self.probs = self.interpret_output(net_output[0])
        img = cv2.imread('person.jpg')
        self.show_results(self.boxes, img)

    def interpret_output(self, output):
        probs = np.zeros((7, 7, 2, 20))
        class_probs = np.reshape(output[0:980], (7, 7, 20))
        scales = np.reshape(output[980:1078], (7, 7, 2))
        boxes = np.reshape(output[1078:], (7, 7, 2, 4))
        offset = np.transpose(np.reshape(np.array([np.arange(7)] * 14), (2, 7, 7)), (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, 0:2] = boxes[:, :, :, 0:2] / 7.0
        boxes[:, :, :, 2] = np.multiply(boxes[:, :, :, 2], boxes[:, :, :, 2])
        boxes[:, :, :, 3] = np.multiply(boxes[:, :, :, 3], boxes[:, :, :, 3])

        boxes[:, :, :, 0] *= self.w_img
        boxes[:, :, :, 1] *= self.h_img
        boxes[:, :, :, 2] *= self.w_img
        boxes[:, :, :, 3] *= self.h_img

        for i in range(2):
            for j in range(20):
                probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
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
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[i][1],
                           boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

        return result

    def show_results(self, img, results):
        img_cp = img.copy()
        if self.filewrite_txt:
            ftxt = open(self.tofile_txt, 'w')
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3]) // 2
            h = int(results[i][4]) // 2
            if self.disp_console: print(
                '    class : ' + results[i][0] + ' , [x,y,w,h]=[' + str(x) + ',' + str(y) + ',' + str(
                    int(results[i][3])) + ',' + str(int(results[i][4])) + '], Confidence = ' + str(results[i][5]))
            if self.filewrite_img or self.imshow:
                cv2.rectangle(img_cp, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(img_cp, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
                cv2.putText(img_cp, results[i][0] + ' : %.2f' % results[i][5], (x - w + 5, y - h - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            if self.filewrite_txt:
                ftxt.write(results[i][0] + ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h) + ',' + str(
                    results[i][5]) + '\n')
        if self.filewrite_img:
            if self.disp_console: print('    image file writed : ' + self.tofile_img)
            cv2.imwrite(self.tofile_img, img_cp)
        if self.imshow:
            cv2.imshow('YOLO_tiny detection', img_cp)
            cv2.waitKey(1)
        if self.filewrite_txt:
            if self.disp_console: print('    txt file writed : ' + self.tofile_txt)
            ftxt.close()

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2],
                                                                         box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3],
                                                                         box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def loaddata(self):
        input_file = open(self.data_path, 'r')

        self.record_list = []
        for line in input_file:
            line = line.strip()
            ss = line.split(' ')
            ss[1:] = [float(num) for num in ss[1:]]
            self.record_list.append(ss)

        self.record_point = 0
        self.record_number = len(self.record_list)

        self.num_batch_per_epoch = int(self.record_number / self.batch_size)
        t_record_producer = Thread(target=self.record_producer)
        t_record_producer.daemon = True
        t_record_producer.start()

        for i in range(self.thread_num):
            t = Thread(target=self.record_customer)
            t.daemon = True
            t.start()

    def record_producer(self):
        """record_queue's processor
        """
        while True:
            if self.record_point % self.record_number == 0:
                random.shuffle(self.record_list)
                self.record_point = 0
            self.record_queue.put(self.record_list[self.record_point])
            self.record_point += 1

    def record_process(self, record):
        """record process
        Args: record
        Returns:
          image: 3-D ndarray
          labels: 2-D list [self.max_objects, 5] (xcenter, ycenter, w, h, class_num)
          object_num:  total object number  int
        """
        image = cv2.imread(record[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h = image.shape[0]
        w = image.shape[1]

        width_rate = self.w_img * 1.0 / w
        height_rate = self.h_img * 1.0 / h

        image = cv2.resize(image, (self.h_img, self.w_img))

        labels = [[0, 0, 0, 0, 0]] * self.max_objects
        i = 1
        object_num = 0
        while i < len(record):
            xmin = record[i]
            ymin = record[i + 1]
            xmax = record[i + 2]
            ymax = record[i + 3]
            class_num = record[i + 4]

            xcenter = (xmin + xmax) * 1.0 / 2 * width_rate
            ycenter = (ymin + ymax) * 1.0 / 2 * height_rate

            box_w = (xmax - xmin) * width_rate
            box_h = (ymax - ymin) * height_rate

            labels[object_num] = [xcenter, ycenter, box_w, box_h, class_num]
            object_num += 1
            i += 5
            if object_num >= self.max_objects:
                break
        return [image, labels, object_num]

    def record_customer(self):
        """record queue's customer
        """
        while True:
            item = self.record_queue.get()
            out = self.record_process(item)
            self.image_label_queue.put(out)

    def batch(self):
        """get batch
        Returns:
          images: 4-D ndarray [batch_size, height, width, 3]
          labels: 3-D ndarray [batch_size, max_objects, 5]
          objects_num: 1-D ndarray [batch_size]
        """
        images = []
        labels = []
        objects_num = []
        for i in range(self.batch_size):
            image, label, object_num = self.image_label_queue.get()
            images.append(image)
            labels.append(label)
            objects_num.append(object_num)
        images = np.asarray(images, dtype=np.float32)
        images = images / 255 * 2 - 1
        labels = np.asarray(labels, dtype=np.float32)
        objects_num = np.asarray(objects_num, dtype=np.int32)
        return images, labels, objects_num

    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    def training(self):  # TODO add training function!
        self.images = tf.placeholder(tf.float32, (self.batch_size, self.h_img, self.w_img, 3))
        self.labels = tf.placeholder(tf.float32, (self.batch_size, self.max_objects, 5))
        self.objects_num = tf.placeholder(tf.int32, (self.batch_size))
        self.predicts =  self.build_networks(self.images)
        self.total_loss, self.nilboy = self.loss(tf.reshape(self.predicts, [-1, self.grid_size, self.grid_size, self.num_class + 5 * self.num_box])
                                                 , self.labels, self.objects_num)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)
        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)
        # tf.summary.scalar('loss', self.total_loss)
        for step in range(self.max_iterators):
            np_images, np_labels, np_objects_num = self.batch()
            _, loss_value, nilboy = self.sess.run([self.train_op, self.total_loss, self.nilboy],
                                             {self.images: np_images, self.labels: np_labels,
                                                        self.objects_num: np_objects_num})
        return None

    def testing(self):
        np_image, _, _ = self.batch()
        result = self.sess.run(self.predicts, {self.images: np_image})
        self.result = self.interpret_output(result[0])
        self.show_results(np_image[0], self.result)



    def iou(self, boxes1, boxes2):
        """calculate ious
        Args:
          boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 1-D tensor [4] ===> (x_center, y_center, w, h)
        Return:
          iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                          boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])
        boxes2 = tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                          boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])

        # calculate the left up point
        lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
        rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

        # intersection
        intersection = rd - lu

        inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]

        mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)

        inter_square = mask * inter_square

        # calculate the boxs1 square and boxs2 square
        square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
        square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

        return inter_square / (square1 + square2 - inter_square + 1e-6)

    def cond1(self, num, object_num, loss, predict, label, nilboy):
        """
        if num < object_num
        """
        return num < object_num

    def body1(self, num, object_num, loss, predict, labels, nilboy):
        """
        calculate loss
        Args:
          predict: 3-D tensor [cell_size, cell_size, 5 * boxes_per_cell]
          labels : [max_objects, 5]  (x_center, y_center, w, h, class)
        """
        label = labels[num:num + 1, :]
        label = tf.reshape(label, [-1])

        # calculate objects  tensor [CELL_SIZE, CELL_SIZE]
        min_x = (label[0] - label[2] / 2) / (self.image_size / self.cell_size)
        max_x = (label[0] + label[2] / 2) / (self.image_size / self.cell_size)

        min_y = (label[1] - label[3] / 2) / (self.image_size / self.cell_size)
        max_y = (label[1] + label[3] / 2) / (self.image_size / self.cell_size)

        min_x = tf.floor(min_x)
        min_y = tf.floor(min_y)

        max_x = tf.ceil(max_x)
        max_y = tf.ceil(max_y)

        temp = tf.cast(tf.stack([max_y - min_y, max_x - min_x]), dtype=tf.int32)
        objects = tf.ones(temp, tf.float32)

        temp = tf.cast(tf.stack([min_y, self.cell_size - max_y, min_x, self.cell_size - max_x]), tf.int32)
        temp = tf.reshape(temp, (2, 2))
        objects = tf.pad(objects, temp, "CONSTANT")

        # calculate objects  tensor [CELL_SIZE, CELL_SIZE]
        # calculate responsible tensor [CELL_SIZE, CELL_SIZE]
        center_x = label[0] / (self.image_size / self.cell_size)
        center_x = tf.floor(center_x)

        center_y = label[1] / (self.image_size / self.cell_size)
        center_y = tf.floor(center_y)

        response = tf.ones([1, 1], tf.float32)

        temp = tf.cast(tf.stack([center_y, self.cell_size - center_y - 1, center_x, self.cell_size - center_x - 1]),
                       tf.int32)
        temp = tf.reshape(temp, (2, 2))
        response = tf.pad(response, temp, "CONSTANT")
        # objects = response

        # calculate iou_predict_truth [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        predict_boxes = predict[:, :, self.num_class + self.num_box:]

        predict_boxes = tf.reshape(predict_boxes, [self.cell_size, self.cell_size, self.num_box, 4])

        predict_boxes = predict_boxes * [self.image_size / self.cell_size, self.image_size / self.cell_size,
                                         self.image_size, self.image_size]

        base_boxes = np.zeros([self.cell_size, self.cell_size, 4])

        for y in range(self.cell_size):
            for x in range(self.cell_size):
                # nilboy
                base_boxes[y, x, :] = [self.image_size / self.cell_size * x, self.image_size / self.cell_size * y, 0, 0]
        base_boxes = np.tile(np.resize(base_boxes, [self.cell_size, self.cell_size, 1, 4]),
                             [1, 1, self.num_box, 1])

        predict_boxes = base_boxes + predict_boxes

        iou_predict_truth = self.iou(predict_boxes, label[0:4])
        # calculate C [cell_size, cell_size, boxes_per_cell]
        C = iou_predict_truth * tf.reshape(response, [self.cell_size, self.cell_size, 1])

        # calculate I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        I = iou_predict_truth * tf.reshape(response, (self.cell_size, self.cell_size, 1))

        max_I = tf.reduce_max(I, 2, keep_dims=True)

        I = tf.cast((I >= max_I), tf.float32) * tf.reshape(response, (self.cell_size, self.cell_size, 1))

        # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        no_I = tf.ones_like(I, dtype=tf.float32) - I

        p_C = predict[:, :, self.num_class:self.num_class + self.num_box]

        # calculate truth x,y,sqrt_w,sqrt_h 0-D
        x = label[0]
        y = label[1]

        sqrt_w = tf.sqrt(tf.abs(label[2]))
        sqrt_h = tf.sqrt(tf.abs(label[3]))
        # sqrt_w = tf.abs(label[2])
        # sqrt_h = tf.abs(label[3])

        # calculate predict p_x, p_y, p_sqrt_w, p_sqrt_h 3-D [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        p_x = predict_boxes[:, :, :, 0]
        p_y = predict_boxes[:, :, :, 1]

        # p_sqrt_w = tf.sqrt(tf.abs(predict_boxes[:, :, :, 2])) * ((tf.cast(predict_boxes[:, :, :, 2] > 0, tf.float32) * 2) - 1)
        # p_sqrt_h = tf.sqrt(tf.abs(predict_boxes[:, :, :, 3])) * ((tf.cast(predict_boxes[:, :, :, 3] > 0, tf.float32) * 2) - 1)
        # p_sqrt_w = tf.sqrt(tf.maximum(0.0, predict_boxes[:, :, :, 2]))
        # p_sqrt_h = tf.sqrt(tf.maximum(0.0, predict_boxes[:, :, :, 3]))
        # p_sqrt_w = predict_boxes[:, :, :, 2]
        # p_sqrt_h = predict_boxes[:, :, :, 3]
        p_sqrt_w = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
        p_sqrt_h = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))
        # calculate truth p 1-D tensor [NUM_CLASSES]
        P = tf.one_hot(tf.cast(label[4], tf.int32), self.num_class, dtype=tf.float32)

        # calculate predict p_P 3-D tensor [CELL_SIZE, CELL_SIZE, NUM_CLASSES]
        p_P = predict[:, :, 0:self.num_class]

        # class_loss
        class_loss = tf.nn.l2_loss(
            tf.reshape(objects, (self.cell_size, self.cell_size, 1)) * (p_P - P)) * self.class_scale
        # class_loss = tf.nn.l2_loss(tf.reshape(response, (self.cell_size, self.cell_size, 1)) * (p_P - P)) * self.class_scale

        # object_loss
        object_loss = tf.nn.l2_loss(I * (p_C - C)) * self.object_scale
        # object_loss = tf.nn.l2_loss(I * (p_C - (C + 1.0)/2.0)) * self.object_scale

        # noobject_loss
        # noobject_loss = tf.nn.l2_loss(no_I * (p_C - C)) * self.noobject_scale
        noobject_loss = tf.nn.l2_loss(no_I * (p_C)) * self.noobject_scale

        # coord_loss
        coord_loss = (tf.nn.l2_loss(I * (p_x - x) / (self.image_size / self.cell_size)) +
                      tf.nn.l2_loss(I * (p_y - y) / (self.image_size / self.cell_size)) +
                      tf.nn.l2_loss(I * (p_sqrt_w - sqrt_w)) / self.image_size +
                      tf.nn.l2_loss(I * (p_sqrt_h - sqrt_h)) / self.image_size) * self.coord_scale

        nilboy = I

        return num + 1, object_num, [loss[0] + class_loss, loss[1] + object_loss, loss[2] + noobject_loss,
                                     loss[3] + coord_loss], predict, labels, nilboy

    def loss(self, predicts, labels, objects_num):
        """Add Loss to all the trainable variables

        Args:
          predicts: 4-D tensor [batch_size, cell_size, cell_size, self.num_classes + 5 * boxes_per_cell]
          ===> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
          labels  : 3-D tensor of [batch_size, max_objects, 5]
          objects_num: 1-D tensor [batch_size]
        """
        class_loss = tf.constant(0, tf.float32)
        object_loss = tf.constant(0, tf.float32)
        noobject_loss = tf.constant(0, tf.float32)
        coord_loss = tf.constant(0, tf.float32)
        loss = [0, 0, 0, 0]
        for i in range(self.batch_size):
            predict = predicts[i, :, :, :]
            label = labels[i, :, :]
            object_num = objects_num[i]
            nilboy = tf.ones([7, 7, 2])
            tuple_results = tf.while_loop(self.cond1, self.body1, [tf.constant(0), object_num,
                                                                   [class_loss, object_loss, noobject_loss, coord_loss],
                                                                   predict, label, nilboy])
            for j in range(4):
                loss[j] = loss[j] + tuple_results[2][j]
            nilboy = tuple_results[5]

        tf.add_to_collection('losses', (loss[0] + loss[1] + loss[2] + loss[3]) / self.batch_size)

        # tf.scalar_summary('class_loss', loss[0] / self.batch_size)
        # tf.scalar_summary('object_loss', loss[1] / self.batch_size)
        # tf.scalar_summary('noobject_loss', loss[2] / self.batch_size)
        # tf.scalar_summary('coord_loss', loss[3] / self.batch_size)
        # tf.scalar_summary('weight_loss', tf.add_n(tf.get_collection('losses')) - (
        #         loss[0] + loss[1] + loss[2] + loss[3]) / self.batch_size)

        return tf.add_n(tf.get_collection('losses'), name='total_loss'), nilboy

def main(argvs):
    yolo = Yolo(argvs)
    cv2.waitKey(1000)


if __name__ == '__main__':
    main(sys.argv)
