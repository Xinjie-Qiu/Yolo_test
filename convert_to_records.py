import os
import io
import hashlib
import xml.etree.ElementTree as ET
import tensorflow as tf
import random
import cv2

from PIL import Image


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
    for objects in root.findall('object'):
        xmin.append(int(objects[4][0].text))
        ymin.append(int(objects[4][1].text))
        xmax.append(int(objects[4][2].text))
        ymax.append(int(objects[4][3].text))
        classes.append(objects[0].text.encode('utf8'))
    image_path = os.path.join('/data/pascal/VOCdevkit/VOC2007/JPEGImages', '{}'.format(filename))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    example = tf.train.Example(features = tf.train.Features(feature={
        'filename': _bytes_feature(filename.encode('utf8')),
        'image/width': _int64_feature(width),
        'image/height': _int64_feature(height),
        'image/depth': _int64_feature(depth),
        'image/object/xmin': _int64_feature(xmin),
        'image/object/ymin': _int64_feature(ymin),
        'image/object/xmax': _int64_feature(xmax),
        'image/object/ymax': _int64_feature(ymax),
        'image/object/classes': _bytes_feature(classes),
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