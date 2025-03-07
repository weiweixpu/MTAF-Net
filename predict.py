# _*_ coding: utf-8 _*_
# @Author :Menghang Ma
# @Email :mamenghang9@gmail.com
import os
import time
import imageio
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
cudnn.benchmark = True
import SimpleITK as sitk
from skimage.morphology import label as lc
from utils.metric import *
local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def dtype_torch_to_numpy(dtype: torch.dtype) -> np.dtype:
    """Convert a torch dtype to its numpy equivalent."""
    return torch.empty([], dtype=dtype).numpy().dtype  # type: ignore


def dtype_numpy_to_torch(dtype: np.dtype) -> torch.dtype:
    """Convert a numpy dtype to its torch equivalent."""
    return torch.from_numpy(np.empty([], dtype=dtype)).dtype


def get_equivalent_dtype(dtype, data_type):
    """Convert to the `dtype` that corresponds to `data_type`.

    The input dtype can also be a string. e.g., `"float32"` becomes `torch.float32` or
    `np.float32` as necessary.

    Example::

        im = torch.tensor(1)
        dtype = get_equivalent_dtype(np.float32, type(im))

    """
    if dtype is None:
        return None
    if data_type is torch.Tensor:
        if isinstance(dtype, torch.dtype):
            # already a torch dtype and target `data_type` is torch.Tensor
            return dtype
        return dtype_numpy_to_torch(dtype)
    if not isinstance(dtype, torch.dtype):
        # assuming the dtype is ok if it is not a torch dtype and target `data_type` is not torch.Tensor
        return dtype
    return dtype_torch_to_numpy(dtype)


def convert_to_numpy(data, dtype):
    if isinstance(data, torch.Tensor):
        dtype_ = get_equivalent_dtype(dtype, torch.Tensor)
        data = np.asarray(data.detach().to(device="cpu").numpy(), dtype=get_equivalent_dtype(dtype_, np.ndarray))
        return data
    else:
        raise ValueError(f"The input data type cannot be converted into  arrays!")

def tailor_and_concat(x, model):
    temp = []
    idh_temp = []

    temp.append(x[..., :128, :128, :128])
    temp.append(x[..., :128, 90:218, :128])
    temp.append(x[..., 54:182, :128, :128])
    temp.append(x[..., 54:182, 90:218, :128])
    temp.append(x[..., :128, :128, 54:182])
    temp.append(x[..., :128, 90:218, 54:182])
    temp.append(x[..., 54:182, :128, 54:182])
    temp.append(x[..., 54:182, 90:218, 54:182])

    y = x[:,:1,:,:,:].clone()
    for i in range(len(temp)):
        if isinstance(model, dict):
            seg_output, idh_out = model['en'](temp[i])  # encoder_outputs:x1_1, x2_1, x3_1,x4_1, encoder_output, intmd_encoder_outputs
            temp[i] = seg_output
            idh_temp.append(idh_out)
        else:
            temp[i], b = model(temp[i])

    y[..., :128, :128, :128] = temp[0]
    y[..., :128, 128:218, :128] = temp[1][..., :, 38:128, :]
    y[..., 128:182, :128, :128] = temp[2][..., 74:128, :, :]
    y[..., 128:182, 128:218, :128] = temp[3][..., 74:128, 38:128, :]
    y[..., :128, :128, 128:182] = temp[4][..., 74:128]
    y[..., :128, 128:218, 128:182] = temp[5][..., :, 38:128, 74:128]
    y[..., 128:182, :128, 128:182] = temp[6][..., 74:128, :, 74:128]
    y[..., 128:182, 128:218, 128:182] = temp[7][..., 74:128, 38:128, 74:128]

    if isinstance(model, dict):
        idh_out = torch.mean(torch.stack(idh_temp), dim=0)
        return y[..., :182], idh_out    # grade_out
    else:
        return y[..., :182]


def apply_mask(mask, out_fname):
    t1ce_Path = r'./data/MNI152_T1_1mm_brain.nii.gz'
    img = sitk.ReadImage(t1ce_Path)
    out = sitk.GetImageFromArray(mask)
    out.CopyInformation(img)
    sitk.WriteImage(out, out_fname)


def postprocess_prediction(seg):
    # basically look for connected components and choose the largest one, delete everything else
    print("running postprocessing... ")
    mask = seg != 0
    lbls = lc(mask, connectivity=mask.ndim)
    lbls_sizes = [np.sum(lbls == i) for i in np.unique(lbls)]
    largest_region = np.argmax(lbls_sizes[1:]) + 1
    seg[lbls != largest_region] = 0
    return seg


def validate_softmax(
        valid_loader,
        model,
        savepath='',          # when in validation set, you must specify the path to save the 'nii' segmentation results here
        use_TTA=False,        # Test time augmentation, False as default!
        save_format=None,     # ['nii','npy'], use 'nii' as default. Its purpose is for submission.
        snapshot=False,       # for visualization. Default false. It is recommended to generate the visualized figures.
        visual='',            # the path to save visualization
        postprocess=False,    # Default False, when use postprocess, the score of dice_ET would be changed.
        valid_in_train=False, # if you are valid when train
        save_csv=''           # evaluation metric saving file name
        ):

    H, W, T = 182, 218, 182

    if isinstance(model, dict):
        model['en'].eval()
    else:
        model.eval()

    idh_prob = []
    idh_class = []
    idh_truth = []
    idh_error_case = []
    ids, dice, iou, = [], [], []
    hd95, assd = [], []
    for i, data in enumerate(valid_loader):
        print('-------------------------------------------------------------------')
        msg = 'Subject {}/{}, '.format(i + 1, len(valid_loader))
        if valid_in_train:
            data = [t.cuda(non_blocking=True) for t in data]
            x, mask_train, idh, patient_ID = data.values()
        else:
            if len(data) == 4:
                x, mask_train, idh, patient_ID = data.values()
                patient_ID = patient_ID[0]
                print("image", x.shape, 'IDH', idh)
                x, idh = x.cuda(), idh.cuda()
            else:
                x = data
                x.cuda()

        if not use_TTA:
            torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
            logit, pred = tailor_and_concat(x, model)
            output = torch.sigmoid(logit)
            idh_pred = F.softmax(pred, dim=1)
            print("idh_pred:", idh_pred)
            idh_prob.append(idh_pred[0][1].item())
            idh_pred_class = torch.argmax(idh_pred, dim=1)
            idh_class.append(idh_pred_class.item())
            print('id:', patient_ID, 'IDH_truth:', idh.item(), 'IDH_pred:', idh_pred_class.item())
            ids.append(patient_ID)
            idh_truth.append(idh.item())
            if not (idh_pred_class.item() == idh.item()):
                idh_error_case.append({'id':patient_ID,'truth:':idh.item(),'pred':idh_pred_class.item()})

        else:
            x = x[..., :182]
            TTA_1,TTA_2,TTA_3,TTA_4,TTA_5,TTA_6,TTA_7,TTA_8 = tailor_and_concat(x, model),tailor_and_concat(x.flip(dims=(2,)), model),\
                                                                tailor_and_concat(x.flip(dims=(3,)), model),tailor_and_concat(x.flip(dims=(4,)), model),\
                                                                tailor_and_concat(x.flip(dims=(2, 3)), model),tailor_and_concat(x.flip(dims=(2, 4)), model),\
                                                                tailor_and_concat(x.flip(dims=(3, 4)), model),tailor_and_concat(x.flip(dims=(2, 3, 4)), model)
            logit = torch.sigmoid(TTA_1[0])  # no flip
            logit += torch.sigmoid(TTA_2[0].flip(dims=(2,)))  # flip H
            logit += torch.sigmoid(TTA_3[0].flip(dims=(3,)))  # flip W
            logit += torch.sigmoid(TTA_4[0].flip(dims=(4,)))  # flip D
            logit += torch.sigmoid(TTA_5[0].flip(dims=(2, 3)))  # flip H, W
            logit += torch.sigmoid(TTA_6[0].flip(dims=(2, 4)))  # flip H, D
            logit += torch.sigmoid(TTA_7[0].flip(dims=(3, 4)))  # flip W, D
            logit += torch.sigmoid(TTA_8[0]).flip(dims=(2, 3, 4))  # flip H, W, D
            output = logit / 8.0

            idh_probs = []
            for pred in [TTA_1, TTA_2, TTA_3, TTA_4, TTA_5, TTA_6, TTA_7, TTA_8]:
                idh_probs.append(F.softmax(pred[1], 1))

            idh_pred = torch.mean(torch.stack(idh_probs),dim=0)
            print("idh_pred:", idh_pred)

            idh_prob.append(idh_pred[0][1].item())

            idh_pred_class = torch.argmax(idh_pred, dim=1)

            idh_class.append(idh_pred_class.item())
            print('id:',patient_ID,'IDH_truth:', idh.item(),'IDH_pred:',idh_pred_class.item())

            ids.append(patient_ID)
            idh_truth.append(idh.item())
            if not (idh_pred_class.item() == idh.item()):
                idh_error_case.append({'id':patient_ID,'truth:':idh.item(),'pred':idh_pred_class.item()})

        output[output > 0.5] = 1
        output[output != 1] = 0
        output = output[0, 0, :H, :W, :T]
        output = convert_to_numpy(output, dtype=torch.uint8)
        mask_train = convert_to_numpy(mask_train.squeeze(), dtype=torch.uint8)

        if postprocess:
            output = postprocess_prediction(output)


        patient_dice = dc(output,  mask_train)
        patient_iou = jc(output,  mask_train)
        patient_hd95 = hausdorff_distance(output, mask_train, voxelspacing=1, percentile=95)
        patient_assd = average_ssd(output, mask_train, voxelspacing=1)
        print('{} dice:{}, iou:{}, hd95:{}, assd:{}'.format(patient_ID, patient_dice,
                                                            patient_iou, patient_hd95,patient_assd))
        dice.append(patient_dice)
        iou.append(patient_iou)
        hd95.append(patient_hd95)
        assd.append(patient_assd)
        msg += '{:>20}, '.format(patient_ID)

        print(msg)

        if savepath:
            # .npy for further model ensemble
            # .nii for directly model submission
            assert save_format in ['npy', 'nii']
            if save_format == 'npy':
                np.save(os.path.join(savepath, patient_ID + '_preds'), output)

            if save_format == 'nii':
                oname = os.path.join(savepath, patient_ID + '.nii.gz')
                seg_img = np.zeros(shape=(H, W, T), dtype=np.uint8)
                seg_img[np.where(output == 1)] = 1
                apply_mask(seg_img, oname)
                print('Successfully save {}'.format(oname))

                if snapshot:
                    """ --- colorful figure--- """
                    Snapshot_img = np.zeros(shape=(H, W, T, 1), dtype=np.uint8)
                    Snapshot_img[:, :, :, 0][np.where(output == 1)] = 255

                    for frame in range(T):
                        if not os.path.exists(os.path.join(visual, patient_ID)):
                            os.makedirs(os.path.join(visual, patient_ID))
                        # imageio.imwrite(os.path.join(visual, patient_ID, str(frame)+'.png'), Snapshot_img[frame, ::-1, :])
                        imageio.imwrite(os.path.join(visual, patient_ID, str(frame)+'.png'), np.repeat(Snapshot_img[frame, ::-1, :], 3, axis=2))
    if isinstance(model, dict):
        print("--------------------------------seg evaluation report---------------------------------------")
        mean_dice = sum(dice)/len(dice)
        mean_iou = sum(iou)/len(iou)
        mean_hd95 = sum(hd95)/len(hd95)
        mean_assd = sum(assd)/len(assd)
        print('mean_dice', mean_dice, 'mean_iou', mean_iou, 'HD95', mean_hd95,
              'ASSD', mean_assd)

        print("--------------------------------IDH evaluation report---------------------------------------")

        from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,classification_report
        import pandas as pd

        save_dir = './statistic/MTAF/best/{}'.format(str(local_time.split(' ')[0]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        data = pd.DataFrame({"ID":ids,"pred":idh_prob,"pred_class":idh_class,"idh_truth":idh_truth,"dice":dice,
                             'iou':iou,'hd95':hd95,'assd':assd})
        data.to_csv(os.path.join(save_dir, save_csv))
        confusion = confusion_matrix(idh_truth, idh_class)
        print(confusion)
        labels = [0, 1]
        target_names = ["wild", "Mutant"]
        print(classification_report(idh_truth, idh_class, labels=labels, target_names=target_names))
        

        auc = roc_auc_score(idh_truth, idh_prob)
        acc = accuracy_score(idh_truth, idh_class)
        print("AUC:", str(auc))
        print("ACC:", str(acc))
        
        if float(np.sum(confusion)) != 0:
            accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
        print("Global Accuracy: " + str(accuracy))
        specificity = 0
        if float(confusion[0, 0] + confusion[0, 1]) != 0:
            specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
        print("Specificity: " + str(specificity))
        sensitivity = 0
        if float(confusion[1, 1] + confusion[1, 0]) != 0:
            sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
        print("Sensitivity: " + str(sensitivity))
        precision = 0
        if float(confusion[1, 1] + confusion[0, 1]) != 0:
            precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
        print("Precision: " + str(precision))

        numerator = 2 * precision * sensitivity
        Denominator = precision + sensitivity
        f1_score = numerator/Denominator
        print("f1_score " + str(f1_score))
        print("-------------------------- error cases----------------------------------------")
        for case in idh_error_case:
            print(case)
        final_df = pd.DataFrame({'dice': mean_dice, 'iou':  mean_iou, 'hd95': mean_hd95, 'assd': mean_assd,
                                 'auc': auc, 'acc': accuracy, 'specificity': specificity,
                                 'Sensitivity': sensitivity, 'precision': precision,'f1_score': f1_score},index=[0])
        save_metric = 'metric_' + save_csv
        final_df.to_csv(os.path.join(save_dir, save_metric))
