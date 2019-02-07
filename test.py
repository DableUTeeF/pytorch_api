import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time
from natthaphon.utils import GeneratorEnqueuer


class Loader(DataLoader):
    def __next__(self):
        return


# if __name__ == '__main__':
"""
MP 8 w: 13.74183440208435
MT 8 w: 51.31627321243286
MP 0 w: 46.57823419570923
"""
train_dataset = datasets.ImageFolder(
    '/root/palm/DATA/plant/train',
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]))
generator = Loader(train_dataset,
                   batch_size=32,
                   shuffle=True,
                   num_workers=8,
                   pin_memory=False)
val_enqueuer = GeneratorEnqueuer(generator.__iter__())
val_enqueuer.start(workers=8)
trainloader = val_enqueuer.get()
starttime = time.time()
batch_time = 0
for idx, _ in enumerate(generator):
    # time.sleep(1)
    print(time.time() - batch_time)
    batch_time = time.time()
print(time.time() - starttime)
