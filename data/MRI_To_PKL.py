# _*_ coding: utf-8 _*_
# @Author :Menghang Ma
# @Email :mamenghang9@gmail.com
import pickle
import os
import numpy as np
import SimpleITK as sitk

modalities = ('Cmain', 'T1main', 'T2main')

data_set = {
        'root': '',
        'grade_file': '',
        'has_label': True,
        'out_path' : ''
        }


def sitk_load(file_name):
    """

    """
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')
    proxy = sitk.ReadImage(file_name)
    data = sitk.GetArrayFromImage(proxy)
    return data


def csv_load(file_name, subject_id):
    """
    :param file_name:
    :param subject_id:
    :param return_type: grade_idh= 0/1
    :return:
    """
    import pandas as pd
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')
    csv_data = pd.read_csv(file_name)

    idh = csv_data[csv_data.BraTS_2020_ID == int(subject_id)]['IDH_status'].values.ravel()

    if idh == 'WT':
        label_idh = 0
    elif idh == 'Mutant':
        label_idh = 1
    else:
        label_idh = -1
        print("the IDH of the case [{}] is {} (not in WT and Mutant)".format(subject_id, idh))

    return label_idh


def process_f32b0(path, seg_label=True, grade_file='',out_path=''):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    if seg_label:
        label_head = sitk.ReadImage(path + os.sep + str('main_mask.nii.gz'))
        label_dict = {
            "spacing": label_head.GetSpacing(),
            "direction": label_head.GetDirection(),
            "size": label_head.GetSize(),
            "origin": label_head.GetOrigin()
        }
        label_numpy = sitk.GetArrayFromImage(label_head)
        label = np.array(label_numpy, dtype='uint8', order='C')

    if os.path.exists(grade_file):

        subject_id = path.split('\\')[-1]
        subtype_label = csv_load(grade_file, subject_id)

    images = np.stack([np.array(sitk_load(path + os.sep + modal + '.nii.gz'), dtype='float32', order='C') for modal in modalities], -1)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    subject_id = path.split('\\')[-1]
    output = out_path + os.sep + subject_id + '_data.pkl'

    mask = images.sum(-1) > 0

    for k in range(3):

        x = images[..., k]  #
        y = x[mask]

        x[mask] -= y.mean()
        x[mask] /= y.std()

        images[..., k] = x

    with open(output, 'wb') as f:
        print(output)

        if seg_label and not os.path.exists(grade_file):
            pickle.dump((images, label,label_dict), f)
        elif seg_label and os.path.exists(grade_file):
            print("{} IDH_label:{}".format(subject_id, subtype_label))
            pickle.dump((images, label, subtype_label, label_dict), f)
        else:
            pickle.dump(images, f)



def doit(dset):
    root, has_label, grade_file, out_path = dset['root'], dset['has_label'], dset['grade_file'], dset['out_path']

    subjects = os.listdir(root)
    subjects.sort()
    names = subjects

    paths = [os.path.join(root, name) for name in names]

    for path in paths:
        process_f32b0(path, has_label, grade_file, out_path)


if __name__ == '__main__':
    doit(data_set)
