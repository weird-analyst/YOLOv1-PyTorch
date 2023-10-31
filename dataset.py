import torch
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, img_size=448, transforms = None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.C = C
        self.imgsize = img_size
        self.transform = transforms

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n","").split()
                ]

                boxes.append([class_label, x, y, width, height])
        
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        #Resizing to fixed size
        w, h = image.size
        image = image.resize((self.imgsize, self.imgsize))
        width = (width*self.imgsize)/w
        height = (height*self.imgsize)/h

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        #Creating label matrix for training purposes
        label_matrix = torch.zeros((self.S, self.S, self.C+ 5*self.B))

        #Filling label matrix by seeing which block the box belongs to and changing those 
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            
            #Taking the box and the height and width relative to cell
            i, j = int(self.S*y), int(self.S*x)
            x_cell, y_cell = self.S*x-j, self.S*y-i

            width_cell, height_cell = (
                width*self.S,
                height*self.S
            )

            #Check to see if already something present in box
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
