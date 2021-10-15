'''
Author： YuYue Yang, ChihPin Chen
Date：2021/10/15~
Assignment 1 Kaggle score 0.64271
Create Dc*5000 and Qc*50. Both Dc and Qc are in dictionary type, and only the words that appear and their TFIDF value are recorded in them.
When calculating cos_sim, it should be noted that there may be keywords in Qc, which do not appear in Dc at all
so the design of the loop is in list(set(list(Dc.keys()) + list(Qc.keys( )))), a list of all words in Dc&Qc (set is used to remove duplicates)
Doc Part
    IDF weight calculation: math.log10(1 + (N / (value + 1))
    TF weight calculation: count> 0 -> 1+log(count)
Que Part
    TF weight calculation: count / temp_total => number of occurrences of a single word / total number of words in the article
cos_sim: A dot B/(||A|| * ||B||)
DataSet：5000 Documents, 50 Queries(.txt)
Running time: 0.03 hours
'''
import gc
import math
import os
import time


# Build Dictionary
def BuildDic(path):
    global wordSet
    wordSet = set()
    # search file list step by step
    for docfile in DocFileList:
        f = open(Doc_str + docfile)
        temp_split = f.read().split(' ')
        wordSet = wordSet.union(temp_split)  # Build word set
    f.close()

    print("wordList size : " + str(len(wordSet)))
    print("===============Build Dictionary End===============")


# ComputTF
def ComputeTF():
    # Initialization variable -----------------
    pass
    '''
    global Doc_TFList  # List size(5000 document * 66878 words)
    Doc_TFList = []
    tfDic = {}
    #   DocFileList  = os.listdir('F:/NLP_DataSet/q_50_d_5000/docs/')
    temp_wordDic = dict.fromkeys(wordSet, 0)

    for docfile in DocFileList:
        path_string = Doc_str + docfile
        file = open(path_string)
        temp_split = file.read().lower().split(' ')
        temp_total = len(temp_split)
        for word in temp_split:
            temp_wordDic[word] += 1
        for word, count in temp_wordDic.items():
            tfDic[word] = round((count / temp_total), 6)
        Doc_TFList.append(tfDic)
        # Zero setting
        tfDic = {}
        temp_wordDic = dict.fromkeys(wordSet, 0)
    # Doc_TFList.append(tfDic)
    print(len(Doc_TFList))
    print("===============ComputeTF End===============")
    print("\n")
    # print(wordDic_List)
    # print(wordDic_List.at[0,'greiff'])    # Get value from wordDic_List, ([row,clo])
    '''


# Just one document file for calculate TF
def ComputeTF2(path):
    # Initialization variable -----------------
    # global Doc_TFList2
    # Doc_TFList2 = []
    tfDic = {}
    # temp_wordDic = dict.fromkeys(wordSet, 0)

    file = open(path)
    temp_split = file.read().split(' ')
    temp_total = len(temp_split)
    for word in wordSet:
        # tfDic[word] = round((temp_split.count(word) / temp_total), 8)
        count = temp_split.count(word)
        if count > 0:  # Method2 <-- if count>0 , tf = 1+log(count)
            tfDic[word] = 1 + math.log(count)
        else:
            tfDic[word] = 0
    # print("===============ComputeTF2 End===============")
    # print("\n")
    return tfDic


# ComputeIDF
def ComputeIDF():
    # Initialization variable--------------------
    global Doc_IDFList
    global N, Ni
    Doc_IDFList = dict.fromkeys(wordSet, 0)
    # Total number of files  -> int
    N = DocFileList.__len__()
    # Total number of document that have appeared in index term(keyword)  -> dict
    Ni = dict.fromkeys(wordSet, 0)

    for docfile in DocFileList:
        f = open(Doc_str + docfile)
        temp_split = f.read().split(' ')
        temp_set = set(temp_split)  # Remove repeat word
        for word in temp_set:
            Ni[word] += 1
    # Start IDF calculation
    for word, value in Ni.items():
        Doc_IDFList[word] = round((math.log10(1 + (N / (value + 1)))), 7)
    print(len(Doc_IDFList))
    print("===============ComputeIDF End===============")


# Compute TFIDF
def computeTFIDF():
    # Initialization variable--------------------
    global Doc_TFIDFList
    Doc_TFList = []
    Doc_TFIDFList = []
    tfDic = {}
    tfidf = {}
    for docfile in DocFileList:
        path_string = Doc_str + docfile
        file = open(path_string)
        temp_split = file.read().split(' ')
        temp_set = set(temp_split)
        for word in temp_set:
            # tfDic[word] = round((count / temp_total), 6)
            count = temp_split.count(word)
            if count > 0:  # Method2 <-- if count>0 , tf = 1+log(count)
                # tfDic[word] = round((count / len(temp_split)), 6)
                tfDic[word] = 1 + math.log(count)
            else:
                tfDic[word] = 0
        Doc_TFList.append(tfDic)
        # Zero setting
        tfDic = {}
    print("===========Doc ComputeTF End=============")

    for i in range(0, len(Doc_TFList)):
        for word, tfval in Doc_TFList[i].items():
            tfidf[word] = round(tfval * Doc_IDFList[word], 7)
        Doc_TFIDFList.append(tfidf)
        tfidf = {}
    print("Doc_TFIDFList length:" + str(len(Doc_TFIDFList)))
    print("===============ComputeTFIDF End===============")


# Just one document file for calculate TFIDF
def computeTFIDF2(c_TFDict):
    # Initialization variable--------------------
    tfidf = {}
    for word, tfval in c_TFDict.items():
        tfidf[word] = round(tfval * Doc_IDFList[word], 8)
    # print("===============ComputeTFIDF2 End===============")
    # print("\n")
    return tfidf


# For all Query File(build TF、TFIDF)
def everyQue():
    print("Query Start")
    # Initialize-----------------TF
    global Que_TFIDFList
    Que_tfDic = {}
    tfidf = {}
    Que_TFIDFList = []
    temp_wordDic = dict.fromkeys(wordSet, 0)

    for quefile in QueFileList:
        path = Que_str + quefile
        file = open(path)
        temp_split = file.read().split(' ')
        temp_total = len(temp_split)
        for word in temp_split:
            if word in wordSet:
                temp_wordDic[word] += 1
        for word, count in temp_wordDic.items():
            Que_tfDic[word] = round((count / temp_total), 8)
        for word, tfval in temp_wordDic.items():
            if tfval > 0:
                tfidf[word] = round(tfval * Doc_IDFList[word], 8)
        Que_TFIDFList.append(tfidf)
        # Zero setting
        tfidf = {}
        temp_wordDic = dict.fromkeys(wordSet, 0)
        # print(str(quefile) + " End")
    print("Query End")


# Just cale current Query File(TFIDF)
def oneQue(path):
    # print("Query Start")
    # Initialize-----------------TF
    Que_tfDic = {}
    temp_wordDic = dict.fromkeys(wordSet, 0)

    # for quefile in QueFileList:
    f = open(path)
    temp_split = f.read().lower().split(' ')
    temp_total = len(temp_split)
    for word in temp_split:
        if word in wordSet:
            temp_wordDic[word] += 1
    # trigger --> KeyError: 'polygyni'
    for word, count in temp_wordDic.items():
        Que_tfDic[word] = round((count / temp_total), 8)
    f.close()
    # print("===============Que ComputeTF End===============")

    # Initialize-----------------IDF
    # IDF is the same, no matter whether it is document (DOC) or query (QUE)
    # print("===============Que ComputeIDF End===============")

    # Initialize-----------------TFIDF
    tfidf = {}
    # Que_TFIDFList = []
    for word, tfval in temp_wordDic.items():
        tfidf[word] = round(tfval * Doc_IDFList[word], 8)
    # Que_TFIDFList.append(tfidf)
    return tfidf


def cos_similarity(q_index):
    sumxx, sumxy, sumyy = 0, 0, 0
    Qc = Que_TFIDFList[q_index]
    for i in range(0, 4999):
        Dc = Doc_TFIDFList[i]
        for word in list(set(list(Dc.keys()) + list(Qc.keys()))):
            x = float(Dc.get(word, 0))
            y = float(Qc.get(word, 0))
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        tmp_ans = round(sumxy / math.sqrt(sumxx * sumyy), 8)
        # if math.sqrt(sumxx * sumyy) > 0:
        #     tmp_ans = round(sumxy / math.sqrt(sumxx * sumyy), 8)
        # else:
        #     tmp_ans = 0
        use2sort[DocFileList[i]] = tmp_ans
        sumxx, sumxy, sumyy = 0, 0, 0


# Cale cos_sim of [Query1] with 5000 doc
def cos_similarity2(q_TFIDF, doc_i, curt_TFIDF):
    sumxx, sumxy, sumyy = 0, 0, 0

    v1 = curt_TFIDF
    v2 = q_TFIDF
    # sumData = v1 * v2.T
    # denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    for word in wordSet:
        x = v1[word]
        y = v2[word]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    tmp_ans = round(sumxy / math.sqrt(sumxx * sumyy), 8)
    use2sort[DocFileList[doc_i]] = tmp_ans
    # print(tmp_ans)


# Write first line to result.txt
def pre_write2res():
    write_path = 'D:/Python/NLP_DataSet/q_50_d_5000/result3.txt'
    f = open(write_path, 'a')
    f.write("Query,RetrievedDocuments" + '\n')
    f.close()


# Write result to result.txt
def write2res(quefile, sort_res):
    write_path = 'D:/Python/NLP_DataSet/q_50_d_5000/result3.txt'
    f = open(write_path, 'a')
    f.write(quefile.replace('.txt', '') + ',')
    for i in range(0, 4999):
        f.write(sort_res[i][0].replace('.txt', ''))
        if i <= 4998:
            f.write(" ")
    f.write('\n')
    f.close()


if __name__ == '__main__':
    global DocFileList, QueFileList
    global Doc_str, Que_str
    global doc_index
    global use2sort
    doc_index = 0
    Doc_str = 'D:/Python/NLP_DataSet/q_50_d_5000/docs/'
    Que_str = 'D:/Python/NLP_DataSet/q_50_d_5000/queries/'
    DocFileList = os.listdir(Doc_str)
    QueFileList = os.listdir(Que_str)
    use2sort = dict.fromkeys(DocFileList, 0)
    que_index = 0
    start = time.time()

    pre_write2res()  # First. Write the first line
    BuildDic(Doc_str)
    ComputeIDF()
    everyQue()
    computeTFIDF()
    print("Build Time：" + str((time.time() - start) / 60))
    print("\n")

    gc.collect()
    # Plan1: one query with one -> remember 5000 score -> sort
    for quefile in QueFileList:
        cos_similarity(que_index)
        que_index += 1
        # Sort
        # print(use2sort["FBIS3-10082.txt"])
        sort_res = sorted(use2sort.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        print(str(quefile) + " End")

        # Zero setting
        use2sort = dict.fromkeys(DocFileList, 0)
        # write result in file
        write2res(str(quefile), sort_res)
    # Plan2:all query with 5000 document
    '''
    for quefile in QueFileList:
        Q_TFIDF = oneQue(Que_str + str(quefile))
        for doc_index in range(len(Doc_TFIDFList)):
            cos_similarity2(Q_TFIDF, doc_index, Doc_TFIDFList[doc_index])
            doc_index += 1
        # Sort
        sort_res = sorted(use2sort.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        # print(sort_res[0][0] + " , " + sort_res[1][0])
        print(str(quefile) + " End")
        # Zero setting
        doc_index = 0
        use2sort = dict.fromkeys(DocFileList, 0)
        # write result in file
        write2res(str(quefile), sort_res)
    '''
    endt = time.time()
    print("Running Time：" + str((endt - start) / 60 / 60))  # Print running time of system
