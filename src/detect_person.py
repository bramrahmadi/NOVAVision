from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC


def main(args):

    with tf.Graph().as_default():
        
        with tf.Session() as sess:

            np.random.seed(seed=args.seed)

            
            
            # names = list()
            # probabilities = list()
            idx = list()

            # face_prediction = os.path.expanduser(args.prediction_file)

            print('Loading feature extraction model..')
            facenet.load_model(args.model)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            while True:
                
                path_exp = os.path.expanduser(args.path_to_faces)
                image_paths = facenet.get_image_paths(path_exp)

                if len(image_paths) != 0 :
                    print('Calculating features for images')
                    image_num = len(image_paths)
                    emb_array = np.zeros((image_num, embedding_size))
                    images = facenet.load_data(image_paths, False, False, args.image_size)
                    feed_dict = { images_placeholder:images, phase_train_placeholder:False}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                    
                    classifier_model = os.path.expanduser(args.classifier_filename)

                    print('Recognizing Faces')
                    with open(classifier_model, 'rb') as infile:
                        (model, class_names) = pickle.load(infile)
                    
                    print('Loaded classifier model from file "%s"' % classifier_model)

                    predictions = model.predict_proba(emb_array)
                    best_indices = np.argmax(predictions, axis=1)
                    best_prob = predictions[np.arange(len(best_indices)), best_indices]

                    for i in range(len(best_indices)):
                        print('%4d  %s: %.3f' % (i, class_names[best_indices[i]], best_prob[i]))
                        ids = class_names[best_indices[i]]+' : '+repr(round(best_prob[i],2))+' \n'
                        idx.clear()
                        idx.append(ids)

                    with open(face_prediction, 'w') as outfile:
                        outfile.writelines(idx)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='~/Documents/TA/CODE/facenet/models/20180408-102900.pb')
    parser.add_argument('--path_to_faces', type=str, default='~/Documents/TA/CODE/facenet/data/detection')
    parser.add_argument('--classifier_filename', type=str, default='~/Documents/TA/CODE/facenet/models/class_model.pkl')
    parser.add_argument('--prediction_file', type=str, default='~/Documents/TA/CODE/facenet/data/face_prediction.txt')

    parser.add_argument('--image_size', type=str, default=160)

    parser.add_argument('--seed', type=int, default=666)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

                    


