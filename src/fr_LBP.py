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

    # load classifier
    face_recognizer = cv.face.LBPHFaceRecognizer_create()

    face_recognizer.read(classifier_model)

    # load class names
    class_names_exp = os.path.expanduser(args.class_name)

    with open(class_names_exp, 'rb') as infile:
        class_names = pickle.load(infile)

    print('Loaded classifier model from file "%s"' % class_names_exp)
    
    # cap = cv.VideoCapture(os.path.expanduser('~/Documents/TA/CODE/facenet/data/my_video-1.avi'))
    cap = cv.VideoCapture(0)
    cap.set(3, 432)
    cap.set(4, 240)
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

        image_size = args.image_size

        if (faceSum>0) :
            for (x,y,w,h) in faces :
                face = gray[y:y+h, x:x+w]

                label, confidence = face_recognizer.predict(face)

                if confidence>args.treshold:
                    person = class_names[label]
                    totalacc += confidence
                    l += 1
                else:
                    person = 'Unrecognized'

                cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                ids = person+' : '+repr(round(confidence, 2))
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
    
    parser.add_argument('--class_name', type=str, default='~/Documents/TA/CODE/facenet/models/class_names.pkl')
    parser.add_argument('--cascade_file', type=str, default='~/Documents/TA/CODE/facenet/models/haarcascade_frontalface_alt.xml')
    parser.add_argument('--model', type=str, default='~/Documents/TA/CODE/facenet/models/20180408-102900.pb')
    parser.add_argument('--path_to_faces', type=str, default='~/Documents/TA/CODE/facenet/data/detection')
    parser.add_argument('--classifier_filename', type=str, default='~/Documents/TA/CODE/facenet/models/model_LBP.xml')
    parser.add_argument('--prediction_file', type=str, default='~/Documents/TA/CODE/facenet/data/face_prediction.txt')

    parser.add_argument('--image_size', type=int, default=160)

    parser.add_argument('--seed', type=int, default=666)

    parser.add_argument('--treshold', type=int, default=0.5)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

                    


