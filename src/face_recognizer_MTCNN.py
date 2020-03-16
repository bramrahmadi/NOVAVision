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
import align.detect_face
import time


def main(args):

    with tf.Graph().as_default():
        
        with tf.Session() as sess:

            np.random.seed(seed=args.seed)

            totalacc = 0.0
            total = 0.0
            infercount = 0
            l = 0

            # saver = tf.train.import_meta_graph(os.path.expanduser(args.mtcnn_meta)
	        # saver.restore(sess, os.path.expanduser(args.mtcnn_ckpt))

            #load face detector
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

            # names = list()
            # probabilities = list()
            # idx = list()

            # face_prediction = os.path.expanduser(args.prediction_file)
            classifier_model = os.path.expanduser(args.classifier_filename) # model for prediction

            with open(classifier_model, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            print('Loading feature extraction model..')
            facenet.load_model(args.model)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0") # get tensors
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image_size = args.image_size
            embedding_size = embeddings.get_shape()[1]

            # cap = cv.VideoCapture(os.path.expanduser('~/Documents/TA/CODE/facenet/data/amel.mp4'))
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
                fcount += 1
                frame = cv.flip(frame, 1)
                gray = cv.cvtColor(frame, 0)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv.destroyAllWindows()
                    break
                start_infer = time.time()
                state, faces, bbox = detect(gray, pnet, rnet, onet, args)
                end_infer = time.time() - start_infer
                total += end_infer
                infercount += 1
                print('--- %s Second ---' % end_infer)

                if state:
                    images = np.zeros((len(faces), image_size, image_size, 3))
                    for i, face in enumerate(faces) :
                        # img = facenet.crop(face, False, image_size)
                        # img = facenet.flip(img, False)
                        img = facenet.prewhiten(face)
                        images[i,:,:,:] = img
                        # bb = bbox[i]
                        # emb_array = np.zeros((image_num, embedding_size))
                        # emb_array = np.zeros((1, embedding_size))
                        # resized = cv.resize(face,(image_size,image_size),interpolation = cv.INTER_LINEAR)
                        # images = facenet.load_img(face, image_size)
                    
                    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                
                        # print('Recognizing Faces')
                        # print('Loaded classifier model from file "%s"' % classifier_model)

                    predictions = model.predict_proba(emb_array)
                    best_indices = np.argmax(predictions, axis=1)
                    best_prob = predictions[np.arange(len(best_indices)), best_indices]
                    
                    for j in range(len(best_indices)):
                        bb = bbox[j]
                        if best_prob[j]>args.treshold:
                            person = class_names[best_indices[j]]
                            totalacc += best_prob[j]
                            l += 1
                        else:
                            person = 'Unrecognized'

                        cv.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 2)
                        ids = person+' : '+repr(round(best_prob[j], 2))
                        # w = int(bb[2]-bb[0])//2
                        # h = int(bb[3]-bb[1])//2
                        cv.putText(frame, ids, (bb[0], bb[1]-7), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
                cv.imshow('face_recognizer', frame)
            
            end = time.time()

            timecount = end - start
            fps = fcount/timecount

            print('Time taken : {0} seconds'.format(round(end, 2)))
            print('Average fps : {0} fps'.format(round(fps, 2)))
            print('Average inference time : %f second' % (total/infercount))
            print('Average prob : %f ' % (round(totalacc/l, 2)))

def detect(img, pnet, rnet, onet, args):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    if img.ndim<2:
        print('Unable to align')
    if img.ndim == 2:
        img = facenet.to_rgb(img)
    img = img[:,:,0:3]

    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces == 0 :
        return False, img, [0,0,0,0]
    else:
        det = bounding_boxes[:,0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces>1:
            if args.detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                det_arr.append(det[index,:])
        else:
            det_arr.append(np.squeeze(det))

        if len(det_arr)>0:
            detfaces = []
            boundbox = []
            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-args.margin/2, 0)
                bb[1] = np.maximum(det[1]-args.margin/2, 0)
                bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                scaled = cv.resize(cropped, (args.image_size, args.image_size), interpolation=cv.INTER_LINEAR)
                detfaces.append(scaled)
                boundbox.append(bb)
            return True,detfaces,boundbox

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='~/Documents/TA/CODE/facenet/models/20170512-110547.pb')
    parser.add_argument('--path_to_faces', type=str, default='~/Documents/TA/CODE/facenet/data/detection')
    parser.add_argument('--classifier_filename', type=str, default='~/Documents/TA/CODE/facenet/models/model_split.pkl')
    parser.add_argument('--prediction_file', type=str, default='~/Documents/TA/CODE/facenet/data/face_prediction.txt')
    parser.add_argument('--detect_multiple_faces', type=bool, default=True)
    parser.add_argument('--image_size', type=int, default=160)
    parser.add_argument('--mtcnn_meta', type=str, default='~/Documents/TA/CODE/facenet/models/model-20180408-102900.meta')
    parser.add_argument('--mtcnn_ckpt', type=str, default='~/Documents/TA/CODE/facenet/models/model-20180408-102900.ckpt-90.data-00000-of-00001')

    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--seed', type=int, default=666)

    parser.add_argument('--treshold', type=int, default=0.5)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

                    


