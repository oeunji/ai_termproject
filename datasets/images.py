import os
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class ImagesDataset(VisionDataset):
    """
    Custom dataset for images organized in a directory structure.

    Directory structure:
    root/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            ...

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        split (string): Dataset split, one of {"train", "valid", "test"}.
        split_ratio (tuple): Ratios for splitting dataset into train/valid/test, e.g., (0.8, 0.1, 0.1).
    """

    # Supported image file extensions
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    def __init__(self, root, transform=None, target_transform=None, split="train", split_ratio=(0.8, 0.1, 0.1)):
        super(ImagesDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.split_ratio = split_ratio

        # Load all image paths and labels
        self.classes, self.class_to_idx = self._find_classes(root)
        self.samples = self._make_dataset(root, self.class_to_idx)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Found 0 files in subfolders of: {root}. Supported file extensions are {', '.join(self.IMG_EXTENSIONS)}")

        # Split the dataset into train/valid/test
        self.train_samples, self.valid_samples, self.test_samples = self._split_dataset(self.samples, split_ratio)

        # Assign the appropriate subset based on split
        if split == "train":
            self.data = self.train_samples
        elif split == "valid":
            self.data = self.valid_samples
        elif split == "test":
            self.data = self.test_samples
        else:
            raise ValueError(f"Invalid split: {split}. Supported splits are 'train', 'valid', 'test'.")

    def _find_classes(self, root):
        """
        Find classes (subfolder names) and map them to indices.
        """
        classes = [d.name for d in os.scandir(root) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, root, class_to_idx):
        """
        Create a dataset of image paths and corresponding labels.
        """
        instances = []
        for target_class in sorted(class_to_idx.keys()):
            class_idx = class_to_idx[target_class]
            target_dir = os.path.join(root, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if self._is_valid_file(path):
                        instances.append((path, class_idx))
        return instances

    def _is_valid_file(self, path):
        """
        Check if a file is a valid image file.
        """
        return path.lower().endswith(tuple(self.IMG_EXTENSIONS))

    def _split_dataset(self, samples, split_ratio):
        """
        Split dataset into train/valid/test based on the provided ratio.
        """
        num_samples = len(samples)
        train_end = int(num_samples * split_ratio[0])
        valid_end = train_end + int(num_samples * split_ratio[1])

        train_samples = samples[:train_end]
        valid_samples = samples[train_end:valid_end]
        test_samples = samples[valid_end:]
        return train_samples, valid_samples, test_samples

    def __getitem__(self, index):
        """
        Return an item from the dataset.
        """
        path, target = self.data[index]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return f"Split: {self.split}, Total samples: {len(self.samples)}"
