from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import dlib
import numpy as np
import argparse
import facenet
import os
import sys
import pickle
import cv2 as cv
import time
import unify
import errno


def main(args):

    args = parse_arguments(args)

    with tf.Graph().as_default():
        
        with tf.Session() as sess:

            np.random.seed(seed=args.seed)
            
            totalacc = 0.0
            total = 0.0
            infercount = 0
            l = 0
            
            #load face detector
            faceDetector = cv.CascadeClassifier()
            CascadeFile = os.path.expanduser(args.cascade_file)
            faceDetector.load(CascadeFile)

            # names = list()
            # probabilities = list()
            # idx = list()
            
            # load face track
            tracker = dlib.correlation_tracker()

            #initiate face track
            # track = 0

            # face_prediction = os.path.expanduser(args.prediction_file)

            # print('Recognizing Faces')
            P1 = os.path.expanduser(args.classifier1)
            P2 = os.path.expanduser(args.classifier2)
            with open(P1, 'rb') as infile:
                (model1, class_names1) = pickle.load(infile)

            with open(P2, 'rb') as infile:
                (model2, class_names2) = pickle.load(infile)

            print('Loading feature extraction model..')
            facenet.load_model(args.model)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image_size = args.image_size
            embedding_size = embeddings.get_shape()[1]
            
            # cap = cv.VideoCapture(os.path.expanduser('~/Documents/TA/CODE/facenet/data/my_video-1.avi'))
            impath = os.path.expanduser(args.image_path)
            groundpath = os.path.expanduser(args.ground_path)

            seq_paths, seq_names = unify.get_filenames(groundpath)
            
            fcount = 0
            reinit = 0
            brk = 0
            start = time.time()

            for seq in range(len(seq_paths)):
                
                if brk: #check break
                    break

                sequence = seq_names[seq]
                if sequence[0:2] == "P1":
                    (model, class_names) = (model1, class_names1)
                elif sequence[0:2] == "P2":
                    (model, class_names) = (model2, class_names2)
                seqfile = open(seq_paths[seq], "r")
                filename = os.path.join(impath,args.flags+"/"+sequence+str(args.threshold)+".txt")
                
                if not os.path.exists(os.path.dirname(filename)):
                    try:
                        os.makedirs(os.path.dirname(filename))
                    except OSError as exc: # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                
                predictfile = open(filename, "w+")
                runeval = open(os.path.join(impath,args.flags+"/eval"+str(args.threshold)+".txt"), "w+")
                track = 0


                for line in seqfile:

                    if cv.waitKey(1) & 0xFF == ord('q'):
                        cv.destroyAllWindows()
                        brk = 1
                        break

                    fields = line.split(",")
                    framenum = fields[0]
                    framefile = impath+sequence+"/"+framenum+".jpg"
                    # print(framefile)
                    frame = cv.imread(framefile)
                    fcount+=1
                    newline = '0'

                    if track == 0 or fcount == 1:

                        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                        start_infer = time.time()
                        faces = faceDetector.detectMultiScale(gray)
                        end_infer = time.time() - start_infer
                        total += end_infer
                        infercount += 1
                        # print('--- %s Second ---' % end_infer)
                        print('DETECTING FACES')

                        #initiate bounding box for tracked face
                        maxArea = 0
                        xt = 0
                        yt = 0
                        wt = 0
                        ht = 0

                        faceSum = len(faces)

                        if (faceSum>0) :        
                            # images = np.zeros((faceSum, image_size, image_size, 3))
                            images = np.zeros((1, image_size, image_size, 3))
                            # faceCount = 0
                            for (x,y,w,h) in faces :
                                if w*h > maxArea:
                                    xt = int(x)
                                    yt = int(y)
                                    wt = int(w)
                                    ht = int(h)
                                    maxArea = w*h

                            if maxArea > 0:
                                face = frame[yt:yt+ht, xt:xt+wt, :]
                                resized = cv.resize(face, (image_size, image_size), interpolation = cv.INTER_LINEAR)
                                img = facenet.prewhiten(resized)
                                images[0,:,:,:] = img
                                
                                # calculate embeddings
                                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                                #predict class
                                predictions = model.predict_proba(emb_array)
                                best_indices = np.argmax(predictions, axis=1)
                                best_prob = predictions[np.arange(len(best_indices)), best_indices]

                                for i in range(len(best_indices)):
                                # x,y,w,h = faces[i]
                                    if best_prob[i]>float(args.threshold):
                                        person = class_names[best_indices[i]]
                                        totalacc += best_prob[i]
                                        l += 1
                                    else:
                                        person = '0000'

                                    ids = person

                                tracker.start_track(frame, dlib.rectangle(xt, yt, xt+wt, yt+ht))

                                track = 1
                        
                        else:
                            newline = '0'
                            # faceCount += 1
                        
                            # print('Loaded classifier model from file "%s"' % classifier_model)
                    if track == 1:

                        print('TRACKING')

                        trackQuality = tracker.update(frame)

                        if trackQuality > 8.75:
                            pos = tracker.get_position()

                            pos_x = int(pos.left())
                            pos_y = int(pos.top())
                            pos_w = int(pos.right())
                            pos_h = int(pos.bottom())

                            cv.rectangle(frame, (pos_x, pos_y), (pos_w, pos_h), (255, 0, 0), 2)
                            cv.putText(frame, ids, (pos_x, pos_y-15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
                            newline = framenum+','+str(pos_y)+','+str(pos_x)+','+str(pos_w-pos_x)+','+str(pos_h-pos_y)+','+ids

                        else:
                            newline = '0'
                            track = 0
                            reinit+=1
                    
                    if newline == '0':
                        (pos_h, pos_x, pos_y, pos_w) = (0,0,0,0)
                        ids = '0000'
                        newline = framenum+','+str(pos_y)+','+str(pos_x)+','+str(pos_w-pos_x)+','+str(pos_h-pos_y)+','+ids
                    
                    predictfile.write(newline+"\r\n")
                    cv.imshow('face_recognizer', frame)
                
                predictfile.close()
                seqfile.close()
            
            end = time.time()

            timecount = end - start
            fps = fcount/timecount
            
            runeval.write('Frame count : {0}\r\n'.format(fcount))
            runeval.write('Time taken : {0} seconds\r\n'.format(round(timecount, 2)))
            runeval.write('Average fps : {0} fps\r\n'.format(round(fps, 2)))
            
            runeval.write('Average inference time : %f Second\r\n' % (total/infercount))
            # print('Average prob : %f ' % (round(totalacc/l, 2)))
            runeval.write('Tracker Re-Init : %d times\r\n' % reinit)
            
            runeval.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    # parser.add_argument('--cascade_file', type=str, default='~/Documents/TA/CODE/facenet/models/lbpcascade_frontalface.xml')
    # parser.add_argument('--model', type=str, default='~/Documents/TA/CODE/facenet/models/20180408-102900.pb')
    parser.add_argument('--cascade_file', type=str, default='~/Documents/TA/CODE/facenet/models/haarcascade_frontalface_alt.xml')
    parser.add_argument('--model', type=str, default='~/Documents/TA/CODE/facenet/models/20170512-110547.pb')
    parser.add_argument('--image_path', type=str, default='~/Documents/TA/CODE/ChokePoint/video/')
    parser.add_argument('--classifier1', type=str, default='~/Documents/TA/CODE/ChokePoint/face/P1/G2/P1L_S4_C1.pkl')
    parser.add_argument('--classifier2', type=str, default='~/Documents/TA/CODE/ChokePoint/face/P2/G2/P2E_S3_C1.pkl')
    parser.add_argument('--ground_path', type=str, default='~/Documents/TA/CODE/ChokePoint/gt/G1/')

    parser.add_argument('--image_size', type=int, default=160)

    parser.add_argument('--seed', type=int, default=666)

    parser.add_argument('--flags', type=str, default='H')

    parser.add_argument('--threshold', type=str, default=0)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(sys.argv[1:])
