import json
import cv2
import numpy as np
import os

from torch.utils.data import Dataset


class ControlNetDataset(Dataset):

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
            
        prompt = item[self.prompt]


        source = cv2.imread(f'{self.dataset_path}/{source_filename}')
        target = cv2.imread(f'{self.dataset_path}/{target_filename}')

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
    