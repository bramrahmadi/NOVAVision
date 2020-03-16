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
import evaluate_HAAR
import evaluate_LBP
import evaluate_MTCNN

def main():

    G1P1 = "~/Documents/TA/CODE/ChokePoint/face/P1/G1/P1E_S1_C1.pkl"
    G1P2 = "~/Documents/TA/CODE/ChokePoint/face/P2/G1/P2L_S1_C1.pkl"

    G2P1 = "~/Documents/TA/CODE/ChokePoint/face/P1/G2/P1E_S4_C1.pkl"
    G2P2 = "~/Documents/TA/CODE/ChokePoint/face/P2/G2/P2L_S3_C3.pkl"

    ground1 = '~/Documents/TA/CODE/ChokePoint/gt/G1/'
    ground2 = '~/Documents/TA/CODE/ChokePoint/gt/G2/'

    flagsH1 = "H1"
    flagsH2 = "H2"
    flagsM1 = "M1"
    flagsM2 = "M2"
    flagsL1 = "L1"
    flagsL2 = "L2"

    thresholds = ["0.1","0.3","0.5","0.7","0.9"]


    for t in thresholds:
        evaluate_HAAR.main(["--classifier1", G1P1,"--classifier2", G1P2,"--ground_path", ground2,"--flags", flagsH1,"--threshold",t])
        evaluate_HAAR.main(["--classifier1", G2P1,"--classifier2", G2P2,"--ground_path", ground1,"--flags", flagsH2,"--threshold",t])
        evaluate_MTCNN.main(["--classifier1", G1P1,"--classifier2", G1P2,"--ground_path", ground2,"--flags", flagsM1,"--threshold",t])
        evaluate_MTCNN.main(["--classifier1", G2P1,"--classifier2", G2P2,"--ground_path", ground1,"--flags", flagsM2,"--threshold",t])
        evaluate_LBP.main(["--classifier1", G1P1,"--classifier2", G1P2,"--ground_path", ground2,"--flags", flagsL1,"--threshold",t])
        evaluate_LBP.main(["--classifier1", G2P1,"--classifier2", G2P2,"--ground_path", ground1,"--flags", flagsL2,"--threshold",t])

if __name__ == "__main__":
    main()