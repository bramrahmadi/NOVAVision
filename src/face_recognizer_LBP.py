from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import pickle
import cv2 as cv
import time


def main(args):

    with tf.Graph().as_default():
        
        with tf.Session() as sess:

            np.random.seed(seed=args.seed)
            
            totalacc = 0.0
            total = 0.0
            k = 0
            l = 0
            
            #load face detector
            faceDetector = cv.CascadeClassifier()
            CascadeFile = os.path.expanduser(args.cascade_file)
            faceDetector.load(CascadeFile)

            # names = list()
            # probabilities = list()
            # idx = list()

            # face_prediction = os.path.expanduser(args.prediction_file)
            classifier_model = os.path.expanduser(args.classifier_filename)

            # print('Recognizing Faces')
            with open(classifier_model, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            print('Loading feature extraction model..')
            facenet.load_model(args.model)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image_size = args.image_size
            embedding_size = embeddings.get_shape()[1]
            
            # cap = cv.VideoCapture(os.path.expanduser('~/Documents/TA/CODE/facenet/data/my_video-1.avi'))
            cap = cv.VideoCapture(1)
            cap.set(3, 800)
            cap.set(4, 600)
            cap.set(5, 30)

            fcount = 0
            start = time.time()
            while True:
                
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv.flip(frame, 1)
                fcount += 1
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv.destroyAllWindows()
                    break
                
                start_time = time.time()
                faces = faceDetector.detectMultiScale(gray)
                end = time.time() - start_time
                total += end
                k += 1
                print('--- %s Second ---' % end)

                faceSum = len(faces)

                if (faceSum>0) :
                    images = np.zeros((faceSum, image_size, image_size, 3))
                    faceCount = 0
                    for (x,y,w,h) in faces :
                        face = frame[y:y+h, x:x+w]
                        # emb_array = np.zeros((image_num, embedding_size))
                        # emb_array = np.zeros((1, embedding_size))
                        resized = cv.resize(face, (image_size, image_size), interpolation = cv.INTER_LINEAR)
                        # img = facenet.crop(resized, False, image_size)
                        # img = facenet.flip(img, False)
                        img = facenet.prewhiten(resized)
                        images[faceCount,:,:,:] = img
                        faceCount += 1

                    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                
                    # print('Loaded classifier model from file "%s"' % classifier_model)

                    predictions = model.predict_proba(emb_array)
                    best_indices = np.argmax(predictions, axis=1)
                    best_prob = predictions[np.arange(len(best_indices)), best_indices]

                    for i in range(len(best_indices)):
                        x,y,w,h = faces[i]
                        if best_prob[i]>args.treshold:
                            person = class_names[best_indices[i]]
                            totalacc += best_prob[i]
                            l += 1
                        else:
                            person = 'Unrecognized'
                        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        ids = person+' : '+repr(round(best_prob[i], 2))
                        cv.putText(frame, ids, (x, y-7), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
                        
                cv.imshow('face_recognizer', frame)
            
            end = time.time()

            timecount = end - start
            fps = fcount/timecount

            print('Time taken : {0} seconds'.format(round(end, 2)))
            print('Average fps : {0} fps'.format(round(fps, 2)))
            
            print('Average inference time : %f Second' % (total/k))
            print('Average prob : %f ' % (round(totalacc/l, 2)))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cascade_file', type=str, default='~/Documents/TA/CODE/facenet/models/lbpcascade_frontalface.xml')
    parser.add_argument('--model', type=str, default='~/Documents/TA/CODE/facenet/models/20170512-110547.pb')
    parser.add_argument('--path_to_faces', type=str, default='~/Documents/TA/CODE/facenet/data/detection')
    parser.add_argument('--classifier_filename', type=str, default='~/Documents/TA/CODE/facenet/models/model_split.pkl')
    parser.add_argument('--prediction_file', type=str, default='~/Documents/TA/CODE/facenet/data/face_prediction.txt')

    parser.add_argument('--image_size', type=int, default=160)

    parser.add_argument('--seed', type=int, default=666)

    parser.add_argument('--treshold', type=int, default=0.5)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))