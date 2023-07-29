import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()
        self.conv1=nn.Conv2d(3,16,3,padding=1)
        self.conv2=nn.Conv2d(16,32,3,padding=1)
        self.conv3=nn.Conv2d(32,64,3,padding=1)
        self.conv4=nn.Conv2d(64,128,3,padding=1)
        self.conv5=nn.Conv2d(128,256,3,padding=1)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(256*28*28,2048)
        self.fc2=nn.Linear(2048,num_classes)
        self.batchnorm1=nn.BatchNorm2d(32)
        self.batchnorm2=nn.BatchNorm2d(128)
        self.batchnorm3=nn.BatchNorm1d(2048)
        self.dropout=nn.Dropout(p=dropout)
        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.relu(self.conv1(x))
        x=self.maxpool(self.relu(self.batchnorm1(self.conv2(x))))
        x=self.relu(self.conv3(x))
        x=self.maxpool(self.relu(self.batchnorm2(self.conv4(x))))
        x=self.maxpool(self.relu(self.conv5(x)))
        x=x.view(x.size(0),-1)
        x=self.dropout(x)
        x=self.relu(self.batchnorm3(self.fc1(x)))
        x=self.dropout(x)
        x=self.fc2(x)
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
