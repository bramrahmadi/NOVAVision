from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import argparse
import math
import facenet
import sys
import pickle
import shutil

def main(args):

    with tf.Graph().as_default():

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            np.random.seed(seed=args.seed)

            dataset = facenet.get_dataset(args.input_dir)

            if len(dataset)>0 :

                paths, labels = facenet.get_image_paths_and_labels(dataset)

                print('Number of images : %d' % len(paths))

                print('Loading feature extraction model..')
                facenet.load_model(args.model)

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embeddings_size = embeddings.get_shape()[1]

                print('Calculating features...')
                num_images = len(paths)
                num_batches_per_epoch = int(math.ceil(1.0*num_images / args.batch_size))
                emb_array = np.zeros((num_images, embeddings_size))
                for i in range(num_batches_per_epoch):
                    start_index = i*args.batch_size
                    end_index = min((i+1)*args.batch_size, num_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, args.image_size)
                    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                    emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

                classifier_filename = os.path.expanduser(args.class_file)

                with open(classifier_filename, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model : %s' % classifier_filename)

                predictions = model.predict_proba(emb_array)
                best_class_index = np.argmax(predictions, axis=1)
                best_prob = predictions[np.arange(len(best_class_index)), best_class_index]

                for i in range(len(best_class_index)):
                    if best_prob[i]>args.treshold:
                        class_n = class_names[best_class_index[i]]
                        print(class_n)
                        move_file(args, class_n, paths[i])
                    else:
                        print('unrecognized')
                        move_file(args, 'unrecognized', paths[i])

def move_file(args, class_name, old_path):
    class_dir = os.path.join(os.path.expanduser(args.input_dir), class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    filen = os.path.split(old_path)[1]
    new_path = class_dir+'/'+filen
    shutil.move(old_path, new_path)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='~/Documents/TA/CODE/facenet/data/TA2/mtcnn3/')
    parser.add_argument('--model', type=str, default='~/Documents/TA/CODE/facenet/models/20180408-102900.pb')
    parser.add_argument('--class_file', type=str, default='~/Documents/TA/CODE/facenet/models/class_model_MTCNN2.pkl')

    parser.add_argument('--image_size', type=int, default=160)

    parser.add_argument('--seed', type=int, default=666)

    parser.add_argument('--batch_size', type=int, default=90)

    parser.add_argument('--gpu_memory_fraction', type=float, default=0.7)

    parser.add_argument('--treshold', type=float, default=0.8)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


                        
