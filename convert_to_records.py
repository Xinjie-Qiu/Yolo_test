import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import random
import cv2
import numpy as np

h_img = 448
w_img = 448
d_img = 3

classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]


classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
    'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
    'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
    'sofa': 17, 'train': 18, 'tvmonitor': 19}

max_objects = 20

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(xml_file):
    with open(xml_file, 'r') as f:
        xml_tree = ET.parse(f)
    root = xml_tree.getroot()
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size[0].text)
    height = int(size[1].text)
    depth = int(size[2].text)
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    object_number = 0
    for objects in root.findall('object'):
        xmin.append(float(objects[4][0].text) * w_img / width)
        ymin.append(float(objects[4][1].text) * h_img / height)
        xmax.append(float(objects[4][2].text) * w_img / width)
        ymax.append(float(objects[4][3].text) * h_img / height)
        classes.append(classes_num[objects[0].text])
        object_number += 1
        if object_number >= max_objects:
            break
    image_path = os.path.join('/data/pascal/VOCdevkit/VOC2007/JPEGImages', '{}'.format(filename))
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (h_img, w_img))
    image = np.asarray(image, dtype=np.float32)
    image = image / 255 * 2 - 1
    # for i in range(len(xmin)):
    #     cv2.rectangle(image, (int(xmin[i]), int(ymin[i])), (int(xmax[i]), int(ymax[i])), (255, 0, 0), 2)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    example = tf.train.Example(features = tf.train.Features(feature={
        # 'filename': _bytes_feature(filename.encode('utf8')),
        'image/object/xmin': _float_feature(xmin),
        'image/object/ymin': _float_feature(ymin),
        'image/object/xmax': _float_feature(xmax),
        'image/object/ymax': _float_feature(ymax),
        'image/object/classes': _float_feature(classes),
        'image/object/object_number': _int64_feature(object_number),
        'image/raw': _bytes_feature(image.tostring())
    }))
    return example


def main():
    writer_train = tf.python_io.TFRecordWriter('/data/pascal/train.tfrecords')
    writer_test = tf.python_io.TFRecordWriter('/data/pascal/test.tfrecords')
    filename_list = tf.train.match_filenames_once("/data/pascal/VOCdevkit/VOC2007/Annotations/*.xml")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    list = sess.run(filename_list)
    random.shuffle(list)
    i = 1
    tst = 0
    trn = 0
    for xml_file in list:
        try:
            example = create_example(xml_file)
            if i % 10 == 0:
                writer_test.write(example.SerializeToString())
                tst += 1
            else:
                writer_train.write(example.SerializeToString())
                trn += 1
            i += 1
        except Exception:
            pass
    writer_test.close()
    writer_train.close()
    print('Successfully convert tfrecord')
    print('train dataset: # ')
    print(trn)
    print('test dataset: # ')
    print(tst)


if __name__ == '__main__':
    main()