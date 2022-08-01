import torch
from torch.utils.data import Dataset
class SMIL_Dataset(Dataset):
    def __init__(self, datas):
        self.input_idss, self.relevancy = [], []
        for input_idss, label in datas:
            input_idss = [torch.LongTensor(input_ids) for input_ids in input_idss]
            self.input_idss.append(input_idss)
            self.relevancy.append(label)
            
    def __getitem__(self, index):
        return self.input_idss[index], \
               self.relevancy[index]

    def __len__(self):
        return len(self.input_idss)