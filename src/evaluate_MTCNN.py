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
import align.detect_face
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
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

            #load face tracker
            tracker = dlib.correlation_tracker()

            # face_prediction = os.path.expanduser(args.prediction_file)
            
            # model for prediction
            P1 = os.path.expanduser(args.classifier1)
            P2 = os.path.expanduser(args.classifier2)
            with open(P1, 'rb') as infile:
                (model1, class_names1) = pickle.load(infile)
            with open(P2, 'rb') as infile:
                (model2, class_names2) = pickle.load(infile)

            print('Loading feature extraction model..')
            facenet.load_model(args.model)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0") # get tensors
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image_size = args.image_size
            embedding_size = embeddings.get_shape()[1]

            # cap = cv.VideoCapture(os.path.expanduser('~/Documents/TA/CODE/facenet/data/amel.mp4'))
            impath = os.path.expanduser(args.image_path)
            groundpath = os.path.expanduser(args.ground_path)

            seq_paths, seq_names = unify.get_filenames(groundpath)

            fcount = 0
            reinit = 0
            brk = 0
            start = time.time()
            
            for seq in range(len(seq_paths)):
                
                if brk:
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
                        
                        gray = cv.cvtColor(frame, 0)
                        start_infer = time.time()
                        state, faces, bbox = detect(gray, pnet, rnet, onet, args)
                        end_infer = time.time() - start_infer
                        total += end_infer
                        infercount += 1
                        # print('--- %s Second ---' % end_infer)
                        print('DETECTING FACES')

                        #initiate bounding box for tracking
                        maxArea = 0
                        xt = 0
                        yt = 0
                        wt = 0
                        ht = 0

                        
                        if state:
                            # images = np.zeros((len(faces), image_size, image_size, 3))
                            images = np.zeros((1, image_size, image_size, 3))
                            for i, face in enumerate(faces) :
                                bb = bbox[i]
                                if bb[2]*bb[3] > maxArea:
                                    xt = bb[0]
                                    yt = bb[1]
                                    wt = bb[2]
                                    ht = bb[3]
                                    maxArea = bb[2]*bb[3]
                                    faceTrack = face
                                
                            if maxArea > 0:
                                #preprocess faces for CNN input
                                img = facenet.prewhiten(faceTrack)
                                images[0,:,:,:] = img

                                #calculate embedding for face
                                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                                emb_array = sess.run(embeddings, feed_dict=feed_dict)
                        
                                # print('Recognizing Faces')
                                # print('Loaded classifier model from file "%s"' % classifier_model)

                                #predict face class
                                predictions = model.predict_proba(emb_array)
                                best_indices = np.argmax(predictions, axis=1)
                                best_prob = predictions[np.arange(len(best_indices)), best_indices]
                                
                                for j in range(len(best_indices)):
                                    if best_prob[j]>float(args.threshold):
                                        person = class_names[best_indices[j]]
                                        totalacc += best_prob[j]
                                        l += 1
                                    else:
                                        person = '0000'
                                    ids = person
                                
                                tracker.start_track(frame, dlib.rectangle(xt, yt, wt, ht))

                                track = 1
                        else:
                            newline = '0'
                            
                    if track == 1:

                        print('TRACKING')

                        trackQuality = tracker.update(frame)

                        if trackQuality > 8.75 :
                            pos = tracker.get_position()

                            pos_x = int(pos.left())
                            pos_y = int(pos.top())
                            pos_w = int(pos.right())
                            pos_h = int(pos.bottom())

                            cv.rectangle(frame, (pos_x, pos_y), (pos_w, pos_h), (255, 0, 0), 2)
                            cv.putText(frame, ids, (pos_x, pos_y-7), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
                            newline = framenum+','+str(pos_y)+','+str(pos_x)+','+str(pos_w-pos_x)+','+str(pos_h-pos_y)+','+ids

                        else:
                            newline = '0'
                            track = 0
                            reinit+=1

                    cv.imshow('face_recognizer', frame)
                    
                    if newline == '0':
                        (pos_h, pos_x, pos_y, pos_w) = (0,0,0,0)
                        ids = '0000'
                        newline = framenum+','+str(pos_y)+','+str(pos_x)+','+str(pos_w-pos_x)+','+str(pos_h-pos_y)+','+ids
                    
                    predictfile.write(newline+"\r\n")

                
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
    
    # parser.add_argument('--model', type=str, default='~/Documents/TA/CODE/facenet/models/20180408-102900.pb')
    parser.add_argument('--model', type=str, default='~/Documents/TA/CODE/facenet/models/20170512-110547.pb')
    parser.add_argument('--image_path', type=str, default='~/Documents/TA/CODE/ChokePoint/video/')
    parser.add_argument('--classifier1', type=str, default='~/Documents/TA/CODE/ChokePoint/face/P1/G2/P1L_S4_C1.pkl')
    parser.add_argument('--classifier2', type=str, default='~/Documents/TA/CODE/ChokePoint/face/P2/G2/P2E_S3_C1.pkl')
    parser.add_argument('--ground_path', type=str, default='~/Documents/TA/CODE/ChokePoint/gt/G1/')
    parser.add_argument('--detect_multiple_faces', type=bool, default=True)
    parser.add_argument('--image_size', type=int, default=160)
    # parser.add_argument('--mtcnn_meta', type=str, default='~/Documents/TA/CODE/facenet/models/model-20180408-102900.meta')
    # parser.add_argument('--mtcnn_ckpt', type=str, default='~/Documents/TA/CODE/facenet/models/model-20180408-102900.ckpt-90.data-00000-of-00001')
    parser.add_argument('--mtcnn_meta', type=str, default='~/Documents/TA/CODE/facenet/models/model-20170512-110547.meta')
    parser.add_argument('--mtcnn_ckpt', type=str, default='~/Documents/TA/CODE/facenet/models/model-20170512-110547.ckpt-250000.data-00000-of-00001')

    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--seed', type=int, default=666)

    parser.add_argument('--flags', type=str, default='M')

    parser.add_argument('--threshold', type=str, default=0)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(sys.argv[1:])