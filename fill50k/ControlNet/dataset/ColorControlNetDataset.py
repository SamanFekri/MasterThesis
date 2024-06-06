import json
import cv2
import numpy as np
import os

from torch.utils.data import Dataset

from annotator.canny import CannyDetector
from annotator.util import resize_image, HWC3


class ColorControlNetDataset(Dataset):

    def __init__(self, dataset_path, backward: bool = False, data_file: str = 'prompt.json', source: str = 'source', target: str = 'target', prompt: str = 'prompt'):
        """
        Args:
        param dataset_path: Path to the dataset folder.
        param backward: Whether to use backward or no.
        param data_file: Name of the data file.
        param source: key name of the source image.
        param target: key name of the target image.
        param prompt: key name of the prompt.
        """
        self.data = []
        self.dataset_path = dataset_path
        self.backward = backward
        self.data_file = data_file
        self.source = source
        self.target = target
        self.prompt = prompt
        
        # For Adding Canny edges
        self.canny_low = 100
        self.canny_high = 200
        self.resolution = 512
        self.apply_canny = CannyDetector()
        
        # For resizing the original Image
        self.small_dim = (8, 8)
        self.original_dim = (512, 512)


        temp_path = os.path.join(dataset_path, data_file)

        with open(temp_path, 'rt') as f:
            for i, line in enumerate(f):
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item[self.source]
        target_filename = item[self.target]
        # Allow the thing do it backward
        if self.backward:
            source_filename, target_filename = target_filename, source_filename
        
        if self.prompt != "null":
            prompt = ", ".join(item[key.strip()] for key in self.prompt.split(','))
        else:
            prompt = ""


        source = cv2.imread(f'{self.dataset_path}/{source_filename}')
        target = cv2.imread(f'{self.dataset_path}/{target_filename}')

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
        # Add canny detector as second control
        detected_map = resize_image(HWC3(target), self.resolution)
        detected_map = self.apply_canny(detected_map, self.canny_low, self.canny_high)
        canny = HWC3(detected_map)
        
        # Resize the original Image
        resize = cv2.resize(target, self.small_dim, interpolation = cv2.INTER_AREA)
        resize = cv2.resize(resize, self.original_dim, interpolation = cv2.INTER_AREA)

        
        # concat the channels to source
        source = np.concatenate((source, resize), axis=2)
        
        # Normalize source images to [0, 1].
        # source = np.transpose(source, (1, 2, 0))
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
