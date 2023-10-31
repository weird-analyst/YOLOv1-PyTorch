import torch
import torch.optim as optim
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YOLOv1
from dataset import VOCDataset
from utils import(
    nms, IoU, mean_average_precision, cellboxes_to_boxes, get_bboxes, plot_image, save_checkpoint, load_checkpoint
)
from loss import YoloLoss

seed = 420
torch.manual_seed(seed)

##### HYPERPARAMETERS
LEARNINGRATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "trial.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
IMG_SIZE = 448;

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.ToTensor(),])



def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #UPDATE PROGRESS BAR
        loop.set_postfix(loss = loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
    return sum(mean_loss)/len(mean_loss)


def main():
    model = YOLOv1(split_size = 7, num_boxes = 2, num_classes = 20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr= LEARNINGRATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        "data/8examples.csv",
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        img_size=IMG_SIZE,
        transforms = transform
    )

    test_dataset = VOCDataset(
        "data/test.csv",
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        img_size=IMG_SIZE,
        transforms = transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )

    for epoch in range(EPOCHS):
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4, device= DEVICE
        )

        map = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )

        train_fn(train_loader, model, optimizer, loss_fn)
        
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)


if __name__ == "__main__":
    main()