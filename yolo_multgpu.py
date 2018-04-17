import tensorflow as tf
import sys


def main(argvs):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(['/data/pascal/train.tfrecords'])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'filename': tf.FixedLenFeature([], tf.string),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/depth': tf.FixedLenFeature([], tf.int64),
            'image/object/xmin': tf.VarLenFeature(tf.int64),
            'image/object/ymin': tf.VarLenFeature(tf.int64),
            'image/object/xmax': tf.VarLenFeature(tf.int64),
            'image/object/ymax': tf.VarLenFeature(tf.int64),
            'image/object/classes': tf.VarLenFeature(tf.string),
            'image/raw': tf.FixedLenFeature([], tf.string)
        }
    )
    filename = tf.cast(features['filename'], tf.string)
    width = features['image/width']
    height = features['image/height']
    depth = features['image/depth']
    xmin = features['image/object/xmin']
    ymin = features['image/object/ymin']
    xmax = features['image/object/xmax']
    ymax = features['image/object/ymax']
    classes = features['image/object/classes']
    image = tf.decode_raw(features['image/raw'], tf.float32)



    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)



    print('complete')



if __name__ == '__main__':
    main(sys.argv)