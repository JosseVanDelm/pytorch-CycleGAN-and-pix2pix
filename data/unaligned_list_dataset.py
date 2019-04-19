from data.base_dataset import BaseDataset, get_transform
from data.image_folder import is_image_file
from PIL import Image
import os.path


def make_dataset_from_list(file_list, max_dataset_size=float("inf")):
    images = []
    assert os.path.isfile(file_list), "%s is not a file" % file_list

    with open(file_list) as f:
        lines = f.read().splitlines()
    for line in lines:
        if os.path.exists(line) and is_image_file(os.path.basename(line)):
            images.append(line)
    return images[:min(max_dataset_size, len(images))]


class UnalignedListDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets that are given in a file with a new file (given each line).

    It requires two text files to host training images from domain A '/path/to/data/trainA.txt'
    and from domain B '/path/to/data/trainB.txt' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two text files:
    '/path/to/data/testA.txt' and '/path/to/data/testB.txt' during test time.
    """
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.train_list_A = os.path.join(opt.dataroot, opt.phase + 'A.txt') # create a path 'path/to/data/trainA.txt'
        self.train_list_B = os.path.join(opt.dataroot, opt.phase + 'B.txt') # create a path 'path/to/data/trainB.txt'
        self.A_paths = sorted(make_dataset_from_list(self.train_list_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset_from_list(self.train_list_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc        # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc       # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information. (borrowed from unaligned_dataset.py)

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset. (borrowed from unaligned_dataset.py)

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
