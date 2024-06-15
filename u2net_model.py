import torch
import torch.nn as nn
import torch.nn.functional as F

class u2net_model(nn.Module):
    def __init__(self):
        super(u2net_model, self).__init__()
        
        # Contracting Path
        self.c1 = nn.Conv2d(3, 64, 3, padding=1)
        self.c2 = nn.Conv2d(64, 64, 3, padding=1)
        self.p1 = nn.MaxPool2d(2)

        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.c4 = nn.Conv2d(128, 128, 3, padding=1)
        self.p2 = nn.MaxPool2d(2)

        self.c5 = nn.Conv2d(128, 256, 3, padding=1)
        self.c6 = nn.Conv2d(256, 256, 3, padding=1)
        self.p3 = nn.MaxPool2d(2)

        self.c7 = nn.Conv2d(256, 512, 3, padding=1)
        self.c8 = nn.Conv2d(512, 512, 3, padding=1)
        self.p4 = nn.MaxPool2d(2)

        self.c9 = nn.Conv2d(512, 1024, 3, padding=1)
        self.c10 = nn.Conv2d(1024, 1024, 3, padding=1)

        # Expansive Path
        self.u6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.c11 = nn.Conv2d(1024, 512, 3, padding=1)
        self.c12 = nn.Conv2d(512, 512, 3, padding=1)

        self.u7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.c13 = nn.Conv2d(512, 256, 3, padding=1)
        self.c14 = nn.Conv2d(256, 256, 3, padding=1)

        self.u8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c15 = nn.Conv2d(256, 128, 3, padding=1)
        self.c16 = nn.Conv2d(128, 128, 3, padding=1)

        self.u9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c17 = nn.Conv2d(128, 64, 3, padding=1)
        self.c18 = nn.Conv2d(64, 64, 3, padding=1)

        # Output layer
        self.output = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Contracting Path
        c1 = F.relu(self.c1(x))
        c1 = F.relu(self.c2(c1))
        p1 = self.p1(c1)

        c2 = F.relu(self.c3(p1))
        c2 = F.relu(self.c4(c2))
        p2 = self.p2(c2)

        c3 = F.relu(self.c5(p2))
        c3 = F.relu(self.c6(c3))
        p3 = self.p3(c3)

        c4 = F.relu(self.c7(p3))
        c4 = F.relu(self.c8(c4))
        p4 = self.p4(c4)

        c5 = F.relu(self.c9(p4))
        c5 = F.relu(self.c10(c5))

        # Expansive Path
        u6 = self.u6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = F.relu(self.c11(u6))
        c6 = F.relu(self.c12(c6))

        u7 = self.u7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = F.relu(self.c13(u7))
        c7 = F.relu(self.c14(c7))

        u8 = self.u8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = F.relu(self.c15(u8))
        c8 = F.relu(self.c16(c8))

        u9 = self.u9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = F.relu(self.c17(u9))
        c9 = F.relu(self.c18(c9))

        # Output layer
        outputs = torch.sigmoid(self.output(c9))
        
        return outputs

# Create the model instance
model = u2net_model()

# Print model summary
print(model)