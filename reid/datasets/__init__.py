from __future__ import absolute_import
import warnings

from .cuhk03 import CUHK03
from .market1501 import Market1501
from .duke_reid import DukeMTMC_reID
from .msmt17 import MSMT17

from .prid2011 import PRID2011
from .ilids_vid import iLIDS_VID
from .mars import Mars
from .duke_vidreid import DukeMTMC_VidReID
from .duke_si_tkl import DukeMTMC_SITKL
from .duke_mr_tkl import DukeMTMC_MRTKL


__factory = {
    # image-based
    'CUHK03': CUHK03,
    'Market1501': Market1501,
    'DukeMTMC-reID': DukeMTMC_reID,
    'MSMT17': MSMT17,

    # video-based
    'PRID2011': PRID2011,
    'iLIDS-VID': iLIDS_VID,
    'Mars': Mars,
    'DukeMTMC-VideoReID': DukeMTMC_VidReID,
    'DukeMTMC-SI-Tracklet': DukeMTMC_SITKL,
    'DukeMTMC-MR-Tracklet': DukeMTMC_MRTKL
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'viper', 'cuhk01', 'cuhk03',
        'market1501', and 'dukemtmc'.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
