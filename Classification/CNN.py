import torch


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5

        # L1 ImgIn shape=(16, 256, 256, 256)
        #    Conv     -> (16, 256, 256, 256)
        #    Pool     -> (16, 256, 256, 256)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(4, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=4, stride=8))

        # L2 ImgIn shape=(16, 63, 63, 32)
        #    Conv      ->(16, 63, 63, 64)
        #    Pool      ->(16, 16, 16, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=4, stride=4, padding=1))

        # L3 ImgIn shape=(16, 16, 16, 64)
        #    Conv      ->(16, 16, 16, 128)
        #    Pool      ->(16, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=4, stride=4))

        # L5 FC 4x4x128(=2048) inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))

        # L6 Final FC 625 inputs -> 20 outputs
        self.fc2 = torch.nn.Linear(625, 20, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # Flatten them for FC
        out = self.layer4(out)
        out = self.fc2(out)
        return out