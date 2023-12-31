# _*_ coding: utf-8 _*_
# @Author :Menghang Ma
# @Email :mamenghang9@gmail.com
import time
import numpy as np
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from scipy.ndimage import _ni_support
from scipy.ndimage import distance_transform_edt, binary_erosion,\
    generate_binary_structure
from scipy.ndimage import find_objects, label
local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def matrix_minor(arr, i, j):
    return np.delete(np.delete(arr,i,axis=0), j, axis=1)

def caculate_metric_multiclass_classification(confusion,labels,target_names):
    assert confusion.shape[0] == confusion.shape[1]
    accuracy = []
    precision = []
    recall = []
    sensitivity = []
    specificity = []
    f1_score = []
    for i in labels:
        print(confusion[i, i])
        if i==0:
            FN = confusion[i, 1] + confusion[i, 2]
            FP = confusion[1, i] + confusion[2, i]
        elif i==1:
            FN = confusion[i, 0] + confusion[i, 2]
            FP = confusion[0, i] + confusion[2, i]
        elif i==2:
            FN = confusion[i, 0] + confusion[i, 1]
            FP = confusion[0, i] + confusion[1, i]
        TP = confusion[i, i]
        TN = matrix_minor(confusion, i, i).sum()

        acc = (TP + TN) / (TP + FN + FP + TN)
        prec = TP / (TP + FP)
        reca = TP / (TP + FN)
        sens = TP / (TP + FN)
        spes = TN / (TN + FP)
        f1 = 2 * TP / (2 * TP + FP + FN)
        print("%s:accuracy:%0.5f,sensitivity:%0.5f,specificity:%0.5f,precision:%0.5f,recall:%0.5f,f1_score:%0.5f"
              % (target_names[i],acc,sens,spes,prec,reca,f1))
        accuracy.append(acc)
        precision.append(prec)
        recall.append(reca)
        sensitivity.append(sens)
        specificity.append(spes)
        f1_score.append(f1)
    return {"accuracy":accuracy,"sensitivity":sensitivity,"specificity":specificity,
            "precision":precision,"recall":recall,"f1_score":f1_score}


def dice_score(o, t, eps=1e-8):
    num = 2*(o*t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num/den


def mIOU(o, t, eps=1e-8):
    num = (o*t).sum() + eps
    den = (o | t).sum() + eps
    return num/den



def dc(result, reference):
    r"""
    Dice coefficient

    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.

    The metric is defined as

    .. math::

        DC=\frac{2|A\cap B|}{|A|+|B|}

    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```result``` and the
        object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 1.0

    return dc


def jc(result, reference):
    """
    Jaccard coefficient

    Computes the Jaccard coefficient between the binary objects in two images.

    Parameters
    ----------
    result: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
    reference: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.

    Returns
    -------
    jc: float
        The Jaccard coefficient between the object(s) in `result` and the
        object(s) in `reference`. It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    intersection = np.count_nonzero(result & reference)
    union = np.count_nonzero(result | reference)

    try:
        jc = float(intersection) / float(union)
    except ZeroDivisionError:
        jc = 1.0

    return jc


def precision(result, reference):
    """
    Precison.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    precision : float
        The precision between two binary datasets, here mostly binary objects in images,
        which is defined as the fraction of retrieved instances that are relevant. The
        precision is not symmetric.

    See also
    --------
    :func:`recall`

    Notes
    -----
    Not symmetric. The inverse of the precision is :func:`recall`.
    High precision means that an algorithm returned substantially more relevant results than irrelevant.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Precision_and_recall
    .. [2] http://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    tp = np.count_nonzero(result & reference)
    fp = np.count_nonzero(result & ~reference)

    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    return precision


def recall(result, reference):
    """
    Recall.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    recall : float
        The recall between two binary datasets, here mostly binary objects in images,
        which is defined as the fraction of relevant instances that are retrieved. The
        recall is not symmetric.

    See also
    --------
    :func:`precision`

    Notes
    -----
    Not symmetric. The inverse of the recall is :func:`precision`.
    High recall means that an algorithm returned most of the relevant results.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Precision_and_recall
    .. [2] http://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    tp = np.count_nonzero(result & reference)
    fn = np.count_nonzero(~result & reference)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall


def sensitivity(result, reference):
    """
    Sensitivity.
    Same as :func:`recall`, see there for a detailed description.

    See also
    --------
    :func:`specificity`
    """
    return recall(result, reference)


def specificity(result, reference):
    """
    Specificity.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    specificity : float
        The specificity between two binary datasets, here mostly binary objects in images,
        which denotes the fraction of correctly returned negatives. The
        specificity is not symmetric.

    See also
    --------
    :func:`sensitivity`

    Notes
    -----
    Not symmetric. The completment of the specificity is :func:`sensitivity`.
    High recall means that an algorithm returned most of the irrelevant results.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Sensitivity_and_specificity
    .. [2] http://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tn = np.count_nonzero(~result & ~reference)
    fp = np.count_nonzero(result & ~reference)

    try:
        specificity = tn / float(tn + fp)
    except ZeroDivisionError:
        specificity = 0.0

    return specificity


def true_negative_rate(result, reference):
    """
    True negative rate.
    Same as :func:`specificity`, see there for a detailed description.

    See also
    --------
    :func:`true_positive_rate`
    :func:`positive_predictive_value`
    """
    return specificity(result, reference)


def true_positive_rate(result, reference):
    """
    True positive rate.
    Same as :func:`recall` and :func:`sensitivity`, see there for a detailed description.

    See also
    --------
    :func:`positive_predictive_value`
    :func:`true_negative_rate`
    """
    return recall(result, reference)


def positive_predictive_value(result, reference):
    """
    Positive predictive value.
    Same as :func:`precision`, see there for a detailed description.

    See also
    --------
    :func:`true_positive_rate`
    :func:`true_negative_rate`
    """
    return precision(result, reference)


def hd(result, reference, voxelspacing=None, connectivity=1):
    """
    Hausdorff Distance.

    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. It is defined as the maximum surface distance between the objects.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`assd`
    :func:`asd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd


def hausdorff_distance(output, target, voxelspacing=None, connectivity=1, percentile=95):
    """
    95th percentile of the Hausdorff Distance.

    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`hd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """

    # output = convert_to_numpy(output, dtype=torch.Tensor)
    # target = convert_to_numpy(target, dtype=torch.Tensor)
    hd1 = __surface_distances(output, target, voxelspacing, connectivity)
    hd2 = __surface_distances(output, target, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), percentile)
    return hd95


def average_ssd(result, reference, voxelspacing=None, connectivity=1):
    """
    Average symmetric surface distance.

    Computes the average symmetric surface distance (ASD) between the binary objects in
    two images.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.generate_binary_structure` and should usually be :math:`> 1`.
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.

    Returns
    -------
    assd : float
        The average symmetric surface distance between the object(s) in ``result`` and the
        object(s) in ``reference``. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`asd`
    :func:`hd`

    Notes
    -----
    This is a real metric, obtained by calling

    >>> __surface_distances(result, reference)

    and

    >>> __surface_distances(reference, result)

    and then averaging the two lists. The binary images can therefore be supplied in any order.
    """
    # assd = np.mean((__surface_distances(result, reference, voxelspacing, connectivity),
    #                    __surface_distances(reference, result, voxelspacing, connectivity)))

    surface_distance = np.hstack((__surface_distances(result, reference, voxelspacing, connectivity),
                   __surface_distances(reference, result, voxelspacing, connectivity)))

    # surface_distance = np.concatenate([__surface_distances(result, reference, voxelspacing, connectivity),
    #                                    __surface_distances(reference, result, voxelspacing, connectivity)])

    assd = np.nan if surface_distance.shape == (0,) else surface_distance.mean()

    return assd


def asd(result, reference, voxelspacing=None, connectivity=1):
    """
    Average surface distance metric.

    Computes the average surface distance (ASD) between the binary objects in two images.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.generate_binary_structure` and should usually be :math:`> 1`.
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.

    Returns
    -------
    asd : float
        The average surface distance between the object(s) in ``result`` and the
        object(s) in ``reference``. The distance unit is the same as for the spacing
        of elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`assd`
    :func:`hd`


    Notes
    -----
    This is not a real metric, as it is directed. See `assd` for a real metric of this.

    The method is implemented making use of distance images and simple binary morphology
    to achieve high computational speed.

    Examples
    --------
    The `connectivity` determines what pixels/voxels are considered the surface of a
    binary object. Take the following binary image showing a cross

    >>> from scipy.ndimage import generate_binary_structure
    >>> cross = generate_binary_structure(2, 1)
    array([[0, 1, 0],
           [1, 1, 1],
           [0, 1, 0]])

    With `connectivity` set to `1` a 4-neighbourhood is considered when determining the
    object surface, resulting in the surface

    .. code-block:: python

        array([[0, 1, 0],
               [1, 0, 1],
               [0, 1, 0]])

    Changing `connectivity` to `2`, a 8-neighbourhood is considered and we get:

    .. code-block:: python

        array([[0, 1, 0],
               [1, 1, 1],
               [0, 1, 0]])

    , as a diagonal connection does no longer qualifies as valid object surface.

    This influences the  results `asd` returns. Imagine we want to compute the surface
    distance of our cross to a cube-like object:

    >>> cube = generate_binary_structure(2, 1)
    array([[1, 1, 1],
           [1, 1, 1],
           [1, 1, 1]])

    , which surface is, independent of the `connectivity` value set, always

    .. code-block:: python

        array([[1, 1, 1],
               [1, 0, 1],
               [1, 1, 1]])

    Using a `connectivity` of `1` we get

    >>> asd(cross, cube, connectivity=1)
    0.0

    while a value of `2` returns us

    >>> asd(cross, cube, connectivity=2)
    0.20000000000000001

    due to the center of the cross being considered surface as well.

    """
    sds = __surface_distances(result, reference, voxelspacing, connectivity)
    asd = sds.mean()
    return asd


def obj_assd(result, reference, voxelspacing=None, connectivity=1):
    """
    Average symmetric surface distance.

    Computes the average symmetric surface distance (ASSD) between the binary objects in
    two images.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining what accounts
        for a distinct binary object as well as when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.generate_binary_structure` and should usually be :math:`> 1`.
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.

    Returns
    -------
    assd : float
        The average symmetric surface distance between all mutually existing distinct
        binary object(s) in ``result`` and ``reference``. The distance unit is the same as for
        the spacing of elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`obj_asd`

    Notes
    -----
    This is a real metric, obtained by calling

    >>> __obj_surface_distances(result, reference)

    and

    >>> __obj_surface_distances(reference, result)

    and then averaging the two lists. The binary images can therefore be supplied in any order.
    """
    assd = np.mean((__obj_surface_distances(result, reference, voxelspacing, connectivity),
                       __obj_surface_distances(reference, result, voxelspacing, connectivity)))
    return assd


def obj_asd(result, reference, voxelspacing=None, connectivity=1):
    """
    Average surface distance between objects.

    First correspondences between distinct binary objects in reference and result are
    established. Then the average surface distance is only computed between corresponding
    objects. Correspondence is defined as unique and at least one voxel overlap.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining what accounts
        for a distinct binary object as well as when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.generate_binary_structure` and should usually be :math:`> 1`.
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.

    Returns
    -------
    asd : float
        The average surface distance between all mutually existing distinct binary
        object(s) in ``result`` and ``reference``. The distance unit is the same as for the
        spacing of elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`obj_assd`
    :func:`obj_tpr`
    :func:`obj_fpr`

    Notes
    -----
    This is not a real metric, as it is directed. See `obj_assd` for a real metric of this.

    For the understanding of this metric, both the notions of connectedness and surface
    distance are essential. Please see :func:`obj_tpr` and :func:`obj_fpr` for more
    information on the first and :func:`asd` on the second.

    Examples
    --------
    >>> arr1 = numpy.asarray([[1,1,1],[1,1,1],[1,1,1]])
    >>> arr2 = numpy.asarray([[0,1,0],[0,1,0],[0,1,0]])
    >>> arr1
    array([[1, 1, 1],
           [1, 1, 1],
           [1, 1, 1]])
    >>> arr2
    array([[0, 1, 0],
           [0, 1, 0],
           [0, 1, 0]])
    >>> obj_asd(arr1, arr2)
    1.5
    >>> obj_asd(arr2, arr1)
    0.333333333333

    With the `voxelspacing` parameter, the distances between the voxels can be set for
    each dimension separately:

    >>> obj_asd(arr1, arr2, voxelspacing=(1,2))
    1.5
    >>> obj_asd(arr2, arr1, voxelspacing=(1,2))
    0.333333333333

    More examples depicting the notion of object connectedness:

    >>> arr1 = numpy.asarray([[1,0,1],[1,0,0],[0,0,0]])
    >>> arr2 = numpy.asarray([[1,0,1],[1,0,0],[0,0,1]])
    >>> arr1
    array([[1, 0, 1],
           [1, 0, 0],
           [0, 0, 0]])
    >>> arr2
    array([[1, 0, 1],
           [1, 0, 0],
           [0, 0, 1]])
    >>> obj_asd(arr1, arr2)
    0.0
    >>> obj_asd(arr2, arr1)
    0.0

    >>> arr1 = numpy.asarray([[1,0,1],[1,0,1],[0,0,1]])
    >>> arr2 = numpy.asarray([[1,0,1],[1,0,0],[0,0,1]])
    >>> arr1
    array([[1, 0, 1],
           [1, 0, 1],
           [0, 0, 1]])
    >>> arr2
    array([[1, 0, 1],
           [1, 0, 0],
           [0, 0, 1]])
    >>> obj_asd(arr1, arr2)
    0.6
    >>> obj_asd(arr2, arr1)
    0.0

    Influence of `connectivity` parameter can be seen in the following example, where
    with the (default) connectivity of `1` the first array is considered to contain two
    objects, while with an increase connectivity of `2`, just one large object is
    detected.

    >>> arr1 = numpy.asarray([[1,0,0],[0,1,1],[0,1,1]])
    >>> arr2 = numpy.asarray([[1,0,0],[0,0,0],[0,0,0]])
    >>> arr1
    array([[1, 0, 0],
           [0, 1, 1],
           [0, 1, 1]])
    >>> arr2
    array([[1, 0, 0],
           [0, 0, 0],
           [0, 0, 0]])
    >>> obj_asd(arr1, arr2)
    0.0
    >>> obj_asd(arr1, arr2, connectivity=2)
    1.742955328

    Note that the connectivity also influence the notion of what is considered an object
    surface voxels.
    """
    sds = __obj_surface_distances(result, reference, voxelspacing, connectivity)
    asd = np.mean(sds)
    return asd


def obj_fpr(result, reference, connectivity=1):
    """
    The false positive rate of distinct binary object detection.

    The false positive rates gives a percentage measure of how many distinct binary
    objects in the second array do not exists in the first array. A partial overlap
    (of minimum one voxel) is here considered sufficient.

    In cases where two distinct binary object in the second array overlap with a single
    distinct object in the first array, only one is considered to have been detected
    successfully and the other is added to the count of false positives.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    connectivity : int
        The neighbourhood/connectivity considered when determining what accounts
        for a distinct binary object. This value is passed to
        `scipy.ndimage.generate_binary_structure` and should usually be :math:`> 1`.
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.

    Returns
    -------
    tpr : float
        A percentage measure of how many distinct binary objects in ``results`` have no
        corresponding binary object in ``reference``. It has the range :math:`[0, 1]`, where a :math:`0`
        denotes an ideal score.

    Raises
    ------
    RuntimeError
        If the second array is empty.

    See also
    --------
    :func:`obj_tpr`

    Notes
    -----
    This is not a real metric, as it is directed. Whatever array is considered as
    reference should be passed second. A perfect score of :math:`0` tells that there are no
    distinct binary objects in the second array that do not exists also in the reference
    array, but does not reveal anything about objects in the reference array also
    existing in the second array (use :func:`obj_tpr` for this).

    Examples
    --------
    >>> arr2 = numpy.asarray([[1,0,0],[1,0,1],[0,0,1]])
    >>> arr1 = numpy.asarray([[0,0,1],[1,0,1],[0,0,1]])
    >>> arr2
    array([[1, 0, 0],
           [1, 0, 1],
           [0, 0, 1]])
    >>> arr1
    array([[0, 0, 1],
           [1, 0, 1],
           [0, 0, 1]])
    >>> obj_fpr(arr1, arr2)
    0.0
    >>> obj_fpr(arr2, arr1)
    0.0

    Example of directedness:

    >>> arr2 = numpy.asarray([1,0,1,0,1])
    >>> arr1 = numpy.asarray([1,0,1,0,0])
    >>> obj_fpr(arr1, arr2)
    0.0
    >>> obj_fpr(arr2, arr1)
    0.3333333333333333

    Examples of multiple overlap treatment:

    >>> arr2 = numpy.asarray([1,0,1,0,1,1,1])
    >>> arr1 = numpy.asarray([1,1,1,0,1,0,1])
    >>> obj_fpr(arr1, arr2)
    0.3333333333333333
    >>> obj_fpr(arr2, arr1)
    0.3333333333333333

    >>> arr2 = numpy.asarray([1,0,1,1,1,0,1])
    >>> arr1 = numpy.asarray([1,1,1,0,1,1,1])
    >>> obj_fpr(arr1, arr2)
    0.0
    >>> obj_fpr(arr2, arr1)
    0.3333333333333333

    >>> arr2 = numpy.asarray([[1,0,1,0,0],
                              [1,0,0,0,0],
                              [1,0,1,1,1],
                              [0,0,0,0,0],
                              [1,0,1,0,0]])
    >>> arr1 = numpy.asarray([[1,1,1,0,0],
                              [0,0,0,0,0],
                              [1,1,1,0,1],
                              [0,0,0,0,0],
                              [1,1,1,0,0]])
    >>> obj_fpr(arr1, arr2)
    0.0
    >>> obj_fpr(arr2, arr1)
    0.2
    """
    _, _, _, n_obj_reference, mapping = __distinct_binary_object_correspondences(reference, result, connectivity)
    return (n_obj_reference - len(mapping)) / float(n_obj_reference)


def obj_tpr(result, reference, connectivity=1):
    """
    The true positive rate of distinct binary object detection.

    The true positive rates gives a percentage measure of how many distinct binary
    objects in the first array also exists in the second array. A partial overlap
    (of minimum one voxel) is here considered sufficient.

    In cases where two distinct binary object in the first array overlaps with a single
    distinct object in the second array, only one is considered to have been detected
    successfully.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    connectivity : int
        The neighbourhood/connectivity considered when determining what accounts
        for a distinct binary object. This value is passed to
        `scipy.ndimage.generate_binary_structure` and should usually be :math:`> 1`.
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.

    Returns
    -------
    tpr : float
        A percentage measure of how many distinct binary objects in ``result`` also exists
        in ``reference``. It has the range :math:`[0, 1]`, where a :math:`1` denotes an ideal score.

    Raises
    ------
    RuntimeError
        If the reference object is empty.

    See also
    --------
    :func:`obj_fpr`

    Notes
    -----
    This is not a real metric, as it is directed. Whatever array is considered as
    reference should be passed second. A perfect score of :math:`1` tells that all distinct
    binary objects in the reference array also exist in the result array, but does not
    reveal anything about additional binary objects in the result array
    (use :func:`obj_fpr` for this).

    Examples
    --------
    >>> arr2 = numpy.asarray([[1,0,0],[1,0,1],[0,0,1]])
    >>> arr1 = numpy.asarray([[0,0,1],[1,0,1],[0,0,1]])
    >>> arr2
    array([[1, 0, 0],
           [1, 0, 1],
           [0, 0, 1]])
    >>> arr1
    array([[0, 0, 1],
           [1, 0, 1],
           [0, 0, 1]])
    >>> obj_tpr(arr1, arr2)
    1.0
    >>> obj_tpr(arr2, arr1)
    1.0

    Example of directedness:

    >>> arr2 = numpy.asarray([1,0,1,0,1])
    >>> arr1 = numpy.asarray([1,0,1,0,0])
    >>> obj_tpr(arr1, arr2)
    0.6666666666666666
    >>> obj_tpr(arr2, arr1)
    1.0

    Examples of multiple overlap treatment:

    >>> arr2 = numpy.asarray([1,0,1,0,1,1,1])
    >>> arr1 = numpy.asarray([1,1,1,0,1,0,1])
    >>> obj_tpr(arr1, arr2)
    0.6666666666666666
    >>> obj_tpr(arr2, arr1)
    0.6666666666666666

    >>> arr2 = numpy.asarray([1,0,1,1,1,0,1])
    >>> arr1 = numpy.asarray([1,1,1,0,1,1,1])
    >>> obj_tpr(arr1, arr2)
    0.6666666666666666
    >>> obj_tpr(arr2, arr1)
    1.0

    >>> arr2 = numpy.asarray([[1,0,1,0,0],
                              [1,0,0,0,0],
                              [1,0,1,1,1],
                              [0,0,0,0,0],
                              [1,0,1,0,0]])
    >>> arr1 = numpy.asarray([[1,1,1,0,0],
                              [0,0,0,0,0],
                              [1,1,1,0,1],
                              [0,0,0,0,0],
                              [1,1,1,0,0]])
    >>> obj_tpr(arr1, arr2)
    0.8
    >>> obj_tpr(arr2, arr1)
    1.0
    """
    _, _, n_obj_result, _, mapping = __distinct_binary_object_correspondences(reference, result, connectivity)
    return len(mapping) / float(n_obj_result)


def __distinct_binary_object_correspondences(reference, result, connectivity=1):
    """
    Determines all distinct (where connectivity is defined by the connectivity parameter
    passed to scipy's `generate_binary_structure`) binary objects in both of the input
    parameters and returns a 1to1 mapping from the labelled objects in reference to the
    corresponding (whereas a one-voxel overlap suffices for correspondence) objects in
    result.

    All stems from the problem, that the relationship is non-surjective many-to-many.

    @return (labelmap1, labelmap2, n_lables1, n_labels2, labelmapping2to1)
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # label distinct binary objects
    labelmap1, n_obj_result = label(result, footprint)
    labelmap2, n_obj_reference = label(reference, footprint)

    # find all overlaps from labelmap2 to labelmap1; collect one-to-one relationships and store all one-two-many for later processing
    slicers = find_objects(labelmap2)  # get windows of labelled objects
    mapping = dict()  # mappings from labels in labelmap2 to corresponding object labels in labelmap1
    used_labels = set()  # set to collect all already used labels from labelmap2
    one_to_many = list()  # list to collect all one-to-many mappings
    for l1id, slicer in enumerate(slicers):  # iterate over object in labelmap2 and their windows
        l1id += 1  # labelled objects have ids sarting from 1
        bobj = (l1id) == labelmap2[slicer]  # find binary object corresponding to the label1 id in the segmentation
        l2ids = np.unique(labelmap1[slicer][
                                 bobj])  # extract all unique object identifiers at the corresponding positions in the reference (i.e. the mapping)
        l2ids = l2ids[0 != l2ids]  # remove background identifiers (=0)
        if 1 == len(
                l2ids):  # one-to-one mapping: if target label not already used, add to final list of object-to-object mappings and mark target label as used
            l2id = l2ids[0]
            if not l2id in used_labels:
                mapping[l1id] = l2id
                used_labels.add(l2id)
        elif 1 < len(l2ids):  # one-to-many mapping: store relationship for later processing
            one_to_many.append((l1id, set(l2ids)))

    # process one-to-many mappings, always choosing the one with the least labelmap2 correspondences first
    while True:
        one_to_many = [(l1id, l2ids - used_labels) for l1id, l2ids in
                       one_to_many]  # remove already used ids from all sets
        one_to_many = [x for x in one_to_many if x[1]]  # remove empty sets
        one_to_many = sorted(one_to_many, key=lambda x: len(x[1]))  # sort by set length
        if 0 == len(one_to_many):
            break
        l2id = one_to_many[0][1].pop()  # select an arbitrary target label id from the shortest set
        mapping[one_to_many[0][0]] = l2id  # add to one-to-one mappings
        used_labels.add(l2id)  # mark target label as used
        one_to_many = one_to_many[1:]  # delete the processed set from all sets

    return labelmap1, labelmap2, n_obj_result, n_obj_reference, mapping


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def __obj_surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel between all corresponding binary
    objects in result and reference. Correspondence is defined as unique and at least one voxel overlap.
    """
    sds = list()
    labelmap1, labelmap2, _a, _b, mapping = __distinct_binary_object_correspondences(result, reference, connectivity)
    slicers1 = find_objects(labelmap1)
    slicers2 = find_objects(labelmap2)
    for lid2, lid1 in list(mapping.items()):
        window = __combine_windows(slicers1[lid1 - 1], slicers2[lid2 - 1])
        object1 = labelmap1[window] == lid1
        object2 = labelmap2[window] == lid2
        sds.extend(__surface_distances(object1, object2, voxelspacing, connectivity))
    return sds

def __combine_windows(w1, w2):
    """
    Joins two windows (defined by tuple of slices) such that their maximum
    combined extend is covered by the new returned window.
    """
    res = []
    for s1, s2 in zip(w1, w2):
        res.append(slice(min(s1.start, s2.start), max(s1.stop, s2.stop)))
    return tuple(res)
