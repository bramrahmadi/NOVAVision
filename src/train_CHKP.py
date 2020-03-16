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
from sklearn.linear_model import SGDClassifier
import classifier
import time

def main(args):

    seqs = get_sequences(args.data_dir)
    path = os.path.expanduser(args.data_dir)
    # count=0
    if args.mode=="VALIDATE":
        start = time.time()
        f = open(os.path.join(path,"val1.txt"), "w+")
        i = 0
        for seq1 in seqs:
            modelname = args.data_dir+seq1+".pkl"
            totacc = 0
            count = 0
            for seq2 in seqs:
                if seq1 != seq2:
                    seqname = args.data_dir+seq2+"/"
                    acc = classifier.main(["CLASSIFY", seqname, args.model, modelname])
                    totacc+=acc
                    count+=1
                    i+=1
                    print("STEP %d" % i)
            f.write("%s, %.3f\r\n" % (modelname, totacc/count))
            # print('Average accuracy: %.3f' % acc/count)
        f.close()
        end = time.time()
        totaltime = end - start
        print("Running Time : %f seconds, %.2f minutes" % (totaltime, totaltime/60))

    else:
        
        for seq in seqs:
            modelname = args.data_dir+seq+".pkl"
            seqname = args.data_dir+seq+"/"
            classifier.main([args.mode, seqname, args.model, modelname])
            # os.system("src/classifier.py "+args.mode+" "+seqname+" "+args.model+" "+modelname+"--use_split_dataset")

def get_sequences(sequence_path):
    path_exp = os.path.expanduser(sequence_path)
    sequences = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    sequences.sort()
    return sequences

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY', 'VALIDATE'], default='CLASSIFY')
    parser.add_argument('--data_dir', type=str, default='~/Documents/TA/CODE/ChokePoint/face/P2/G2/')
    # parser.add_argument('--model', type=str, default='~/Documents/TA/CODE/facenet/models/20180408-102900.pb')
    parser.add_argument('--model', type=str, default='~/Documents/TA/CODE/facenet/models/20170512-110547.pb')
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
