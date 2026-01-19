from torch.utils.data import Dataset

class ImagenDataset(Dataset):
    def __init__(self, lista_imagenes, lista_clases):
        self.lista_imagenes = lista_imagenes
        self.lista_clases = lista_clases
    
    def __len__(self):
        return len(self.lista_clases)
    
    def __getitem__(self, index):
        return self.lista_imagenes[index], self.lista_clases[index]