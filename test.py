# import  libraries
import random
from pathlib import Path
import toml
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as tt
from tqdm import tqdm

from model import CNN
from util import ToDeviceLoader, get_device, to_device, accuracy, model_save, create_directory


def predict_image(img, model, device, test_dataset):
    xb = to_device(img, device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return test_dataset.classes[preds[0].item()]


if __name__ == "__main__":
    device = get_device()
    config = toml.load(Path("./options/test_config.toml"))
    test_transform = tt.Compose([
        tt.ToTensor(),
    ])
    test_dataset = MNIST(download=False, root=Path(config["data_root"]).absolute().__str__(), train=False,
             transform=test_transform)

    length_of_test_dataset = len(test_dataset)
    random_integer = random.randint(0, length_of_test_dataset)
    img, label = test_dataset[random_integer]
    model = CNN()
    model = to_device(model, device)
    if Path(config["model_save_path"]).is_file():
        checkpoint = torch.load(Path(config["model_save_path"]), map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    else:
        raise ValueError("Missing trained model weights.")
    test_dl = DataLoader(test_dataset, config["batch_size"], num_workers=4, pin_memory=True)
    test_dl = ToDeviceLoader(test_dl, device)
    for index, batch in enumerate(tqdm(test_dl)):
        img, label = batch
        print('Label:', test_dataset.classes[label.item()], ', Predicted:',
              predict_image(img, model, device, test_dataset))




