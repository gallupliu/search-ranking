import os
import random
import shutil

INPUT_DIR_O = os.curdir + "/output_lib_svm/original/"
INPUT_DIR_S = os.curdir + "/output_lib_svm/shuffled/"
OUTPUT_DIR = os.curdir + "/output_buckets/"


def check_folders():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def pre_processing():
    folds = open("config.cfg", 'r')
    line_s = folds.readlines()
    return int(line_s[2])


def create_folders(folder_name):
    folder = OUTPUT_DIR + folder_name
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def clean_up(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)


for inputFile in os.listdir(INPUT_DIR_O):
    with open(INPUT_DIR_O + inputFile,'r') as source:
        lines = source.readlines()
        end = len(lines) - 1
        for entry in range(end + 1):
            choice = random.randint(entry, end)
            lines[entry], lines[choice] = lines[choice], lines[entry]
    if not os.path.exists(INPUT_DIR_S):
        os.makedirs(INPUT_DIR_S)
    with open(INPUT_DIR_S + os.path.splitext(inputFile)[0] + "_Shuffled", 'w') as result:
        result.writelines(lines)

FOLD = pre_processing()
check_folders()

for shuffledFile in os.listdir(INPUT_DIR_S):
    clean_up(OUTPUT_DIR + os.path.splitext(shuffledFile)[0])
    with open(INPUT_DIR_S + shuffledFile, 'r') as inputFile:
        content_file = inputFile.readlines()
        fold = 1
        pos = 0
        while fold <= FOLD:
            content = content_file[:]
            print(len(content_file))
            step = int(len(content)/FOLD)
            testSet = []
            trainSet = []
            # print(pos)
            # print(step)
            # print(fold)
            for element in content[pos:step*fold]:
                testSet.append(element)
                content.remove(element)
            with open(create_folders(shuffledFile) + "/" + os.path.splitext(shuffledFile)[0] + "_TEST" + str(fold), 'w') as testFile:
                for result_line in testSet:
                    testFile.write(str(result_line))
            for element in content:
                trainSet.append(element)
            with open(create_folders(shuffledFile) + "/" + os.path.splitext(shuffledFile)[0] + "_TRAIN" + str(fold), 'w') as trainFile:
                for result_line in trainSet:
                    trainFile.write(str(result_line))
            fold += 1
            pos += step



