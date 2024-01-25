from torchvision import transforms
from PIL import Image
from torchvision.datasets import DatasetFolder
from utils.randaugment import RandAugmentMC
import torch
import numpy as np
import random
from sklearn.model_selection import train_test_split
from PIL import ImageFile
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Seeds
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class TransformFixMatch(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        if self.dataset_name in ["terra", "pacs", "office_home", "vlcs"]:
            self.weak = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(
                        size=224,
                        padding=int(224 * 0.125),
                        pad_if_needed=True,
                        padding_mode="reflect",
                    ),
                ]
            )
            self.strong = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(
                        size=224,
                        padding=int(224 * 0.125),
                        pad_if_needed=True,
                        padding_mode="reflect",
                    ),
                    RandAugmentMC(n=2, m=10),
                ]
            )
            self.normalize = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


train_tfm = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ]
)

test_tfm = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def load_dataset(data_path, dataset_name, domain):
    DATA_TRAIN_SET = data_path + "train"
    parent_path = os.path.dirname(os.path.dirname(os.path.dirname(data_path)))
    unlabeled_path = os.path.join(parent_path, "unlabeled")

    DATA_TEST_SET = os.path.join(parent_path, "test", domain)

    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif")

    train_set = DatasetFolder(
        DATA_TRAIN_SET,
        loader=lambda x: Image.open(x).convert("RGB"),
        extensions=IMG_EXTENSIONS,
        transform=train_tfm,
    )

    test_set = DatasetFolder(
        DATA_TEST_SET,
        loader=lambda x: Image.open(x).convert("RGB"),
        extensions=IMG_EXTENSIONS,
        transform=test_tfm,
    )

    unlabeled_set = DatasetFolder(
        unlabeled_path,
        loader=lambda x: Image.open(x).convert("RGB"),
        extensions=IMG_EXTENSIONS,
        transform=TransformFixMatch(dataset_name),
    )

    val_set = DatasetFolder(
        unlabeled_path,
        loader=lambda x: Image.open(x).convert("RGB"),
        extensions=IMG_EXTENSIONS,
        transform=test_tfm,
    )

    unlabeled_set_samples = unlabeled_set.samples[:]
    val_set_samples = val_set.samples[:]

    train_image_names = [
        example[0].split("/train/")[1] for example in train_set.samples
    ]
    test_image_names = [
        example[0].split(f"/{domain}/")[1] for example in test_set.samples
    ]

    unlabeled_set_samples = [
        example
        for example in unlabeled_set_samples
        if example[0].split("/unlabeled/")[1] not in train_image_names
    ]
    unlabeled_set_samples = [
        example
        for example in unlabeled_set_samples
        if example[0].split("/unlabeled/")[1] not in test_image_names
    ]
    unlabeled_set.samples = unlabeled_set_samples
    unlabeled_set.targets = [target for (example, target) in unlabeled_set_samples]

    val_set_samples = [
        example
        for example in val_set_samples
        if example[0].split("/unlabeled/")[1] not in train_image_names
    ]
    val_set_samples = [
        example
        for example in val_set_samples
        if example[0].split("/unlabeled/")[1] not in test_image_names
    ]
    val_set.samples = val_set_samples
    val_set.targets = [target for (example, target) in val_set_samples]

    targets = unlabeled_set.targets
    unlabeled_idx, valid_idx = train_test_split(
        np.arange(len(targets)), test_size=0.1, shuffle=True, stratify=targets
    )
    un_dataset = torch.utils.data.Subset(unlabeled_set, unlabeled_idx)
    v_dataset = torch.utils.data.Subset(val_set, valid_idx)
    return train_set, un_dataset, v_dataset, test_set
