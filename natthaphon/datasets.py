from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms


class Loader(DataLoader):
    def __len__(self):
        return int(round(len(self.dataset) / self.batch_size))


def cifarloader(root):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR10(root=root, train=True, download=True, transform=transform_train)
    trainloader = Loader(trainset, batch_size=64, shuffle=True, num_workers=0)

    testset = CIFAR10(root=root, train=False, download=True, transform=transform_test)
    testloader = Loader(testset, batch_size=100, shuffle=False, num_workers=0)
    return trainloader, testloader
