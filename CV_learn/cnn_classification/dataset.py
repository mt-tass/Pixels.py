import jax.numpy as jnp
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

cifar_mean = (0.4914,0.4822,0.4465)
cifar_std = (0.2470,0.2435,0.2616)

def getData(batch_size=64,val_split=0.1):
    train_transform = T.Compose([T.RandomHorizontalFlip(),T.RandomCrop(32,padding=4),T.ToTensor(),T.Normalize(cifar_mean,cifar_std)])
    test_transform = T.Compose([T.ToTensor(),T.Normalize(cifar_mean,cifar_std)])
    train = torchvision.datasets.CIFAR10(root="./data",train=True,transform=train_transform,download=True)
    test = torchvision.datasets.CIFAR10(root="./data",train=False,transform=test_transform,download=True)
    n_val = int(len(train)*val_split)
    n_train = int(len(train)-n_val)
    train_data,val_data = torch.utils.data.random_split(train,[n_train,n_val],generator=torch.Generator().manual_seed(7))
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_data,batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(test,batch_size=batch_size,shuffle=False)
    jax_train_loader = ((jnp.array(images.permute(0,2,3,1).numpy()),jnp.array(labels.numpy())) for images,labels in train_loader)
    jax_val_loader = ((jnp.array(images.permute(0,2,3,1).numpy()),jnp.array(labels.numpy())) for images,labels in val_loader)
    jax_test_laoder = ((jnp.array(images.permute(0,2,3,1).numpy()),jnp.array(labels.numpy())) for images,labels in test_loader)

    return jax_train_loader,jax_val_loader,jax_test_laoder

# if __name__ == "__main__":
#     train,val,test = getData()
#     images,labels = next(iter(train))
#     print(images.shape)
#     print(labels.shape)
#     print("works")

