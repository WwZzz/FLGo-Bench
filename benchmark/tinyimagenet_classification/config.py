import random
import shutil
from hashlib import md5
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import cv2
import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as models

root = os.path.abspath(os.path.dirname(__file__))
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    ),
}

class TinyImageNet(ImageFolder):
    # DATASET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    DATASET_URL = (
        "https://github.com/tjmoon0104/pytorch-tiny-imagenet/releases/download/tiny-imagenet-dataset/tiny-imagenet-200.zip"
    )
    DATASET_MD5_HASH = "90528d7ca1a48142e341f4ef8d21d0de"
    def __init__(self, root:str, split:str='train',transform=None, resolution=224):
        self.root = root
        self.zip_file = Path(os.path.join(self.root, "tiny-imagenet-200.zip"))
        if resolution==64:
            self.images_path = 'tiny-imagenet-200'
        elif resolution==224:
            self.images_path = 'tiny-224'
        else:
            raise ValueError("Not Supported resolution, which should be 64 or 224")
        self.load_data()
        super().__init__(os.path.join(root, self.images_path, split), transform)

    def load_data(self):
        DATASET_DIR = Path(os.path.join(self.root, "tiny-imagenet-200"))
        if DATASET_DIR.exists():return
        # Download Dataset if needed
        if not self.zip_file.exists():
            print("Downloading the dataset, this may take a while...")
            with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=self.DATASET_URL.split("/")[-1]) as t:
                def show_progress(block_num, block_size, total_size):
                    t.total = total_size
                    t.update(block_num * block_size - t.n)

                urlretrieve(url=self.DATASET_URL, filename=self.zip_file, reporthook=show_progress)

        # Check MD5 Hash
        with self.zip_file.open("rb") as f:
            assert (
                    md5(f.read()).hexdigest() == self.DATASET_MD5_HASH
            ), "The dataset zip file seems corrupted. Try to download it again."

        # Remove existing data set
        ORIGINAL_DATASET_DIR = Path(os.path.join(self.root, "original"))
        if ORIGINAL_DATASET_DIR.exists():
            shutil.rmtree(ORIGINAL_DATASET_DIR)
        if not ORIGINAL_DATASET_DIR.exists():
            print("Extracting the dataset, this may take a while...")
            # Unzip the dataset
            with ZipFile(self.zip_file, "r") as zip_ref:
                for member in tqdm(zip_ref.infolist(), desc="Extracting"):
                    zip_ref.extract(member, ORIGINAL_DATASET_DIR)

        # Remove existing data set
        if DATASET_DIR.exists():
            shutil.rmtree(DATASET_DIR)
        # Create the dataset directory
        if not DATASET_DIR.exists():
            print("Creating the dataset directory...")
            os.makedirs(DATASET_DIR)

        # Move train images to dataset directory
        ORIGINAL_TRAIN_DIR = Path(os.path.join(ORIGINAL_DATASET_DIR, "tiny-imagenet-200", "train"))
        if ORIGINAL_TRAIN_DIR.exists():
            print("Moving train images...")
            ORIGINAL_TRAIN_DIR.replace(DATASET_DIR / "train")

        # Get validation images and annotations
        val_dict = {}
        ORIGINAL_VAL_DIR = ORIGINAL_DATASET_DIR / "tiny-imagenet-200" / "val"
        with (ORIGINAL_VAL_DIR / "val_annotations.txt").open("r") as f:
            for line in f.readlines():
                split_line = line.split("\t")
                if split_line[1] not in val_dict.keys():
                    val_dict[split_line[1]] = [split_line[0]]
                else:
                    val_dict[split_line[1]].append(split_line[0])

        def split_list_randomly(input_list: list[str], split_ratio=0.5) -> dict[str, list[str]]:
            # Shuffle the input list in-place
            random.shuffle(input_list)

            # Calculate the index to split the list
            split_index = int(len(input_list) * split_ratio)

            # Split the list into two parts
            return {"val": input_list[:split_index], "test": input_list[split_index:]}

        # Sample from validation images randomly into validation and test sets (50/50)
        print("Splitting original dataset images...")
        with tqdm(val_dict.items(), desc="Splitting images", unit="class") as t:
            for image_label, images in t:
                for split_type, split_images in split_list_randomly(images, split_ratio=0.5).items():
                    for image in split_images:
                        src = ORIGINAL_VAL_DIR / "images" / image
                        dest_folder = DATASET_DIR / split_type / image_label / "images"
                        dest_folder.mkdir(parents=True, exist_ok=True)
                        src.replace(dest_folder / image)
                t.update()

        # Remove original directory
        shutil.rmtree(ORIGINAL_DATASET_DIR)

        # Remove resized data set directory
        RESIZED_DIR = Path(os.path.join(self.root, "tiny-224"))
        if RESIZED_DIR.exists():
            shutil.rmtree(RESIZED_DIR)

        # Copy processed dataset to tiny-224
        print("Copying processed dataset to tiny-224...")
        shutil.copytree(DATASET_DIR, RESIZED_DIR)

        # Resize images to 224x224
        def resize_img(image_path: Path, size: int = 224) -> None:
            img = cv2.imread(image_path.as_posix())
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(image_path.as_posix(), img)

        all_images = [*RESIZED_DIR.glob('*/*/*/*.JPEG')]
        print("Resizing images...")
        with tqdm(all_images, desc="Resizing images", unit="file") as t:
            for image in t:
                resize_img(image, 224)

train_data = TinyImageNet(root, 'train', data_transforms['train'], resolution=224)
test_data = TinyImageNet(root, 'test', data_transforms['test'], resolution=224)
val_data = TinyImageNet(root, 'val', data_transforms['val'], resolution=224)

def get_model():
    model_ft = models.resnet18()
    # Finetune Final few layers to adjust for tiny imagenet input
    model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
    num_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_features, 200)
    return model_ft


