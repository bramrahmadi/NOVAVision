from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from operator import add

def main():
    prd_dir = "~/Documents/TA/CODE/ChokePoint/prediction/"
    predictpath = get_folders(prd_dir)

    prob = ["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"]
    for p in prob:
        metH = [0,0,0,0,0,0,0,0,0]
        metM = [0,0,0,0,0,0,0,0,0]
        metL = [0,0,0,0,0,0,0,0,0]
        for path in predictpath:

            root = os.path.dirname(path)
            met = open(os.path.join(path,"met"+p+".txt"), 'r')
            if path[-2] == "H":
                lineH = read_data(met)
                metH = list(map(add,metH,lineH))
            elif path[-2] == "M":
                lineM = read_data(met)
                metM = list(map(add,metM,lineM))
            elif path[-2] == "L":
                lineL = read_data(met)
                metL = list(map(add,metL,lineL))
            met.close()
        
        resH = open(os.path.join(root,"resH"+p+".txt"), 'w+')
        linesH = create_lines(metH)
        resH.writelines(linesH)
        resM = open(os.path.join(root,"resM"+p+".txt"), 'w+')
        linesM = create_lines(metM)
        resM.writelines(linesM)
        resL = open(os.path.join(root,"resL"+p+".txt"), 'w+')
        linesL = create_lines(metL)
        resL.writelines(linesL)

        resH.close()
        resM.close()
        resL.close()

def create_lines(met):
    lines = []
    lines.append("Detector\n")
    lines.append("Average IOU : {:.2f}\n".format(met[0]/2))
    lines.append("Precision : {:.2f}\n".format(met[1]/2))
    lines.append("Recall : {:.2f}\n".format(met[2]/2))
    lines.append("F1 Score : {:.2f}\n".format( divide(2 * (met[1]/2) * (met[2]/2) , (met[1]/2) +(met[2]/2) ) ) )
    lines.append("\n")
    lines.append("System\n")
    lines.append("TP : {}\n".format(met[3]))
    lines.append("FP : {}\n".format(met[4]))
    lines.append("FN : {}\n".format(met[5]))
    lines.append("Precision : {:.2f}\n".format(met[6]/2))
    lines.append("Recall : {:.2f}\n".format(met[7]/2))
    lines.append("F1 Score : {:.2f}\n".format( divide(2 * (met[6]/2) * (met[7]/2) , (met[6]/2) +(met[7]/2) ) ) )
    lines.append("\n")
    lines.append("Accuracy : {:.2f}".format(met[8]/2))

    return lines

def divide(n, d):
    return n/d if d else 0

def read_data(files):
    arr = [0,0,0,0,0,0,0,0,0]
    l1 = files.readline()
    l1 = l1.rstrip().split(',')
    arr[0:3] = l1[0:3]
    l2 = files.readline()
    l2 = l2.rstrip().split(',')
    arr[3:8] = l2[0:5]
    l3 = files.readline()
    l3 = l3.rstrip()
    arr[8] = l3

    return [float(i) for i in arr]

def get_folders(path):
    path_exp = os.path.expanduser(path)
    folders = []
    for root, dirs, files in os.walk(path_exp):
        for name in dirs:
            if os.path.isdir(os.path.join(root, name)):
                folders.append(os.path.join(root, name))
                
    return folders

if __name__ == "__main__":
    main()