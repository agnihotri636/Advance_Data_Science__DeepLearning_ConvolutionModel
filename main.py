# import libraries
import math
from pathlib import Path
import toml
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as tt
from tqdm import tqdm

from model import CNN
from util import ToDeviceLoader, get_device, to_device, model_save, create_directory, accuracy

if __name__ == "__main__":
    # basic configuration
    config = toml.load(Path("./options/config.toml"))
    epoch = 0
    max_acc = -math.inf
    max_epochs = config["max_epochs"]
    create_directory(Path(config["tensorboard_log"]).absolute())
    writer = SummaryWriter(Path(config["tensorboard_log"]).absolute().__str__())

    # image transforms
    train_transform = tt.Compose([
        tt.RandomRotation(degrees=(-20, 20), interpolation=tt.InterpolationMode.BILINEAR),
        tt.ToTensor(),
    ])

    test_transform = tt.Compose([
        tt.ToTensor(),
    ])

    # loading dataset and do not set download to True
    train_data = MNIST(download=config["download"], root=Path(config["data_root"]).absolute().__str__(),
                       transform=train_transform)
    test_data = MNIST(download=False, root=Path(config["data_root"]).absolute().__str__(), train=False,
                      transform=test_transform)

    # test whether the data loaded correctly
    for image, label in train_data:
        print("Image shape: ", image.shape)
        print("Image tensor: ", image)
        print("Label: ", label)
        break

    # check number of test and train classes
    train_classes_items = dict()

    for train_item in train_data:
        label = train_data.classes[train_item[1]]
        if label not in train_classes_items:
            train_classes_items[label] = 1
        else:
            train_classes_items[label] += 1

    print(train_classes_items)

    test_classes_items = dict()
    for test_item in test_data:
        label = test_data.classes[test_item[1]]
        if label not in test_classes_items:
            test_classes_items[label] = 1
        else:
            test_classes_items[label] += 1

    print(test_classes_items)

    # creating Dataloader
    train_dl = DataLoader(train_data, config["batch_size"], num_workers=4, pin_memory=True, shuffle=True)
    test_dl = DataLoader(test_data, config["batch_size"], num_workers=4, pin_memory=True)

    # creating model and loading to device
    device = get_device()
    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["max_epochs"],
                                                           eta_min=config["min_lr"])

    if Path(config["model_save_path"]).is_dir():
        if config["load"] == "current":
            checkpoint = torch.load((Path(config["model_save_path"]) / "current_checkpoint.ckpt").absolute(),
                                    map_location=device)
        else:
            checkpoint = torch.load((Path(config["model_save_path"]) / "best_checkpoint.ckpt").absolute(),
                                    map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["opt"])
        scheduler.load_state_dict(checkpoint["sch"])
        epoch = checkpoint["epoch"]

    model = to_device(model, device)
    train_dl = ToDeviceLoader(train_dl, device)
    test_dl = ToDeviceLoader(test_dl, device)

    # training
    while epoch < max_epochs:
        epoch += 1
        print(f"epoch no: {epoch}")
        train_loss = 0.0
        valid_loss = 0.0
        accuracy_val = 0.0

        model.train()
        
        # training
        for index, batch in enumerate(tqdm(train_dl)):
            optimizer.zero_grad()
            # get the input from batch
            inp_data, label = batch
            # pass the input to model and get prediction
            output = model(inp_data)
            # calculate the loss and store it in variable loss using the criterion
            loss = F.cross_entropy(output, label)
            # backpropagate the loss
            loss.backward()
            train_loss = train_loss + ((1 / (index + 1)) * (loss - train_loss))

            optimizer.step()
        scheduler.step()

        model.eval()
        # validation
        with torch.no_grad():
            for index, batch in enumerate(tqdm(test_dl)):
                # get the input from batch
                inp_data, label = batch
                # pass the input to model and get prediction
                output = model(inp_data)
                # calculate the loss and store it in variable loss using the criterion
                loss = F.cross_entropy(output, label)
                # calculate the accuracy and store it in variable acc using the function called accuracy
                acc = accuracy(output, label)
                valid_loss = valid_loss + ((1 / (index + 1)) * (loss - valid_loss))
                accuracy_val = accuracy_val + ((1 / (index + 1)) * (acc - accuracy_val))


        print(f"\ntrain_loss: {train_loss:.2f} \n"
              f"valid_loss: {valid_loss:.2f} \n"
              f"acc: {accuracy_val:.2f}")

        writer.add_scalar("train_loss", train_loss, global_step=epoch)
        writer.add_scalar("valid_loss", valid_loss, global_step=epoch)
        writer.add_scalar("accuracy", accuracy_val, global_step=epoch)

        if epoch % config["save_epoch"] == 0:
            model_save(Path(config["model_save_path"]), model, optimizer, scheduler, epoch, "current")
            if accuracy_val > max_acc:
                max_acc = accuracy_val
                model_save(Path(config["model_save_path"]), model, optimizer, scheduler, epoch, "best")

    writer.flush()
    writer.close()

