from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2 as cv
import time

def main():

    cap = cv.VideoCapture(1)
    cap.set(3,432)
    cap.set(4,240)
    cap.set(5,30)

    k = 0
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame,1)
        k += 1
        if cv.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            break
        cv.imshow('test',frame)
    
    end = time.time()

    timecount = start - end
    fps = k/timecount

    print('FPS = {0} fps'.format(round(fps, 2)))

if __name__ == '__main__':
    main()