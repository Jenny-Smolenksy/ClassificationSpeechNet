# Yossi Adi wrote ClassificationLoader  (GCommandLoader with few changes)
# Yosi Shrem wrote some of ImbalancedDatasetSampler

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import os
import torch
import utils

import warnings
import random
warnings.simplefilter(action='ignore', category=FutureWarning)

class ClassificationLoader(Dataset):
    """

    A  data set loader where the wavs are arranged in this way:
        root/one/xxx.wav
        root/one/xxy.wav
        root/one/xxz.wav
        root/head/123.wav
        root/head/nsdf3.wav
        root/head/asd932_.wav
    Args:
        root (string): Root directory path.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101):
        classes, class_to_idx = self.find_classes(root)

        spects = []
        dir = os.path.expanduser(root)
        count = 0
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            # if count > 20:
            #     break
            for root, _, fnames in sorted(os.walk(d)):
                # if count > 20:
                #     break
                for fname in sorted(fnames):
                    # if count > 20:
                    #     break
                    if utils.is_audio_file(fname):
                        path = os.path.join(root, fname)
                        label = os.path.join(root, fname.replace(".wav", ".wrd"))
                        item = (path, class_to_idx[target], label)
                        spects.append(item)
                        count += 1
        if len(spects) == 0:
            raise (RuntimeError("Found 0 sound files in subfolders of: " + root +
                                "Supported audio file extensions are: " + ",".join(utils.AUDIO_EXTENSIONS)))

        random.shuffle(spects)
        self.root = root
        self.type = type
        self.spects = spects
        self.classes = classes

        self.class_to_idx = class_to_idx
        self.loader = utils.spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len

    def __getitem__(self, index):
        self.class__ = """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, target, label_path = self.spects[index]
        spect, _, _, _ = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)

        # return features, target
        return spect, target

    def __len__(self):
        return len(self.spects)

    def get_class(self, idx):
         path, target, label_path = self.spects[idx]
         return target

    @staticmethod
    def find_classes(dir):
        classes = [d for d in os.listdir(dir) if (os.path.isdir(os.path.join(dir, d)))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

class ImbalancedDatasetSampler(Sampler):

    def __init__(self, dataset, indices=None):
        self.dataset = dataset
        if indices != None:
            self.indices = indices
        else:
            self.indices = list(range(len(dataset)))

        self.num_samples = len(self.indices)
        labels_list = []
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(idx)
            labels_list.append(label)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        weights = [1.0 / label_to_count[labels_list[idx]] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, idx):
        return self.dataset.get_class(idx)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


