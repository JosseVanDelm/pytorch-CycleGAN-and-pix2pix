from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset_from_list
from PIL import Image
import os.path


class AlignedListDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that there are two files in your directory '/path/to/data/trainA.txt' and '/path/to/data/trainB.txt'
    These files have to represent pairs of images.
    During test time, you need to prepare a file '/path/to/data/testA.txt' or '/path/to/data/testB.txt'.
    """
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.train_list_A = os.path.join(opt.dataroot, opt.phase + 'A.txt')  # create a path/to/data/trainA.txt
        self.train_list_B = os.path.join(opt.dataroot, opt.phase + 'B.txt')  # create a path/to/data/trainB.txt
        self.A_paths = sorted(make_dataset_from_list(self.train_list_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset_from_list(self.train_list_B, opt.max_dataset_size))
        assert(self.opt.load_size >= self.opt.crop_size)  # crop size should be smaller than the size of the loaded image
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        # Get number of input and output channels
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        assert A.size == B.size, "%s and %s are not the same size, A and B must have the same image size" % (A_path, B_path)
        # Apply the same transform on A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        A = A_transform(A)
        B = B_transform(B)
        return {'A': A, 'B':B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images."""
        assert self.A_size == self.B_size, "A and B don't contain an equal number of images."
        return self.A_size
