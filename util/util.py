from pathlib import Path
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(data ,device):
    if isinstance(data ,(list ,tuple)):
        return [to_device(x ,device) for x in data]
    return data.to(device ,non_blocking=True)


class ToDeviceLoader:
    def __init__(self ,data ,device):
        self.data = data
        self.device = device

    def __iter__(self):
        for batch in self.data:
            yield to_device(batch ,self.device)

    def __len__(self):
        return len(self.data)

def create_directory(*argv):
    """
    :param argv: list directories to be created
    """
    for arg in argv:
        if not Path(arg).is_dir():
            Path(arg).mkdir(parents=True)

def remove_files(*argv):
    """
    :param argv: list of files that has to be deleted
    """
    for file in argv:
        if Path(file).is_file():
            Path(file).unlink()


# defining accuracy function
def accuracy(predicted, actual):
    _, predictions = torch.max(predicted, dim=1)
    return torch.tensor(torch.sum(predictions == actual).item() / len(predictions))


def model_save(model_save_dir, model, optimizer, scheduler, epoch, name):

    model_save_dir = Path(model_save_dir)
    create_directory(model_save_dir)
    model_save_path = model_save_dir / f"{name}_checkpoint.ckpt"

    checkpoint = {"model": model.state_dict(),
                  "opt": optimizer.state_dict(),
                  "epoch": epoch,
                  "sch": scheduler.state_dict()}

    torch.save(checkpoint, model_save_path.absolute().__str__())

