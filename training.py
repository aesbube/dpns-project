import os
import torch
import torch.nn as nn
import torch.optim as optim
from train_val_sets import sets

def train(epochs, model_save_path, model, flag):
    input_shape = (256, 256, 3)
    batch_size = 16

    train_x, _, train_y, _ = sets(input_shape, True)

    model.cuda()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i in range(0, len(train_x), batch_size):
            optimizer.zero_grad()
            if flag is True:
                outputs = model(train_x[i:i+batch_size])['out']
                outputs = torch.sigmoid(outputs)
            else:
                outputs = model(train_x[i:i+batch_size])
            loss = criterion(outputs, train_y[i:i+batch_size])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_x)}')
        if (epoch+1) % 50 == 0:
            os.makedirs(os.path.dirname(
                f'{model_save_path}_{epoch+1}.pth'), exist_ok=True)
            torch.save(model.state_dict(), f'{model_save_path}_{epoch+1}.pth')