# _*_ coding: utf-8 _*_
# @Author :Menghang Ma
# @Email :mamenghang9@gmail.com
import pandas as pd
import os
import shutil
import warnings

def label_classification(pklpath, csvpath, savepath):
    # Splitting the dataset into respective files
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    train_path = savepath + os.sep + 'train'
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    valid_path = savepath + os.sep + 'valid'
    if not os.path.exists(valid_path):
        os.makedirs(valid_path)

    csv = pd.read_csv(csvpath)
    csv['ID'] = csv['ID'].astype(str)
    train_ID = csv.loc[csv['TRAIN8'] == 'T'].ID.tolist()
    test_ID = csv.loc[csv['TRAIN8'] == 'Test'].ID.tolist()

    for patientdir in os.listdir(pklpath):
        oripath = pklpath + os.sep + patientdir
        trainpath = train_path + os.sep + patientdir
        testpath = valid_path + os.sep + patientdir
        if str(patientdir[:9]) in train_ID:
            shutil.copy(oripath, trainpath)

        elif str(patientdir[:9]) in test_ID:
            shutil.copy(oripath, testpath)

        else:
            print(str(patientdir))
            warnings.warn('The patient neither in train set nor in test set')

    return


if __name__ == '__main__':
    pklpath = r""   
    csvpath = r""
    savepath = r""
    label_classification(pklpath, csvpath, savepath)
