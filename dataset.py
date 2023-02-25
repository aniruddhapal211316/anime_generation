from torch.utils.data import Dataset
import torchvision.transforms as transforms
from glob import glob
from PIL import Image
from tqdm import tqdm

class Anime(Dataset): 

    def __init__(self, dataset_dir, resolution): 

        self.transform = transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
                    ])

        print("Loading Anime Dataset")
        self.data = list()
        for image_dir in glob(f'{dataset_dir}/*.png'):
            image = Image.open(image_dir)
            iamge = image.convert('RGB')
            self.data.append(self.transform(image))

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

        