from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset

class data_load(Dataset):
    def __init__(self, data_path):
        self.data_pat = data_path
    #data_transform = transforms.Compose([transforms.RandomSizedCrop(224),
    #                                    transforms.RandomHorizontalFlip(),
    #                                    transforms.ToTensor(),
    #                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

    data_root='/scratch/connectome/GANBERT/data/sample/final/'
    train_dataset = datasets.ImageFolder(root=data_root+'train')
    val_dataset = datasets.ImageFolder(root=data_root+'test')
    #val_dataset = datasets.ImageFolder(root=data_root+'test', transform=data_transform)