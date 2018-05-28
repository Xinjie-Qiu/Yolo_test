import tensorflow as tf
import yolo_multgpu as yolo
import sys
import matplotlib.pyplot as plt

datadir = '/data/pascal/test.tfrecords'


def main(argvs):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    image, labels, objects_number = yolo.getfeature(datadir)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    n_image = sess.run(image)
    plt.imshow(n_image[0])

    print('yea')
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main(sys.argv)
