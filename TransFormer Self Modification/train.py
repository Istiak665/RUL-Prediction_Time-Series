import torch
torch.manual_seed(1)
from model import *
from loading_data import *
from testing import *
from visualize import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse




# Define the gradient clipping function
def clip_gradient(grad, max_norm):
    norm = grad.norm()
    if norm > max_norm:
        clip_coef = max_norm / (norm + 1e-6)
        return grad * clip_coef
    return grad

def training():
    for epoch in range(num_epochs):
        i = 1
        epoch_loss = 0

        model.train()

        while i <= 100:
            x = train_group.get_group(i).to_numpy()
            total_loss = 0
            optim.zero_grad()

            for t in range(x.shape[0] - 1):
                if t == 0:
                    continue
                else:
                    X = x[t - 1:t + 2, 2:-1]

                y = x[t, -1:]

                X_train_tensors = Variable(torch.Tensor(X))
                y_train_tensors = Variable(torch.Tensor(y))

                X_train_tensors_final = X_train_tensors.reshape(
                    (1, 1, X_train_tensors.shape[0], X_train_tensors.shape[1]))

                outputs = model.forward(X_train_tensors_final, t)

                loss = criterion(outputs, y_train_tensors)

                total_loss += loss.item()

                loss = loss / (x.shape[0] - 2)
                loss.backward()

                if t == x.shape[0] - 2:
                    optim.step()
                    optim.zero_grad()

                    for group in optim.param_groups:
                        for param in group['params']:
                            if param.grad is not None:
                                param.grad.data = clip_gradient(param.grad.data, max_norm=1.0)

            i += 1
            epoch_loss += total_loss / x.shape[0]

        model.eval()

        with torch.no_grad():
            rmse, result = testing(group_test, y_test, model)

        print("Epoch: %d, training loss: %1.5f, testing rmse: %1.5f" % (epoch, epoch_loss / 100, rmse))

    return result, rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FD001', help='which dataset to run')
    opt = parser.parse_args()

    num_epochs = 4
    d_model = 128
    heads = 4
    N = 2
    m = 84

    if opt.dataset == 'FD001':
        train_group, y_test, group_test = loading_FD001()
        dropout = 0.1

        model = Transformer(m, d_model, N, heads, dropout)

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        result, rmse = training()

        visualize(result, rmse)
    else:
        print('Either dataset not implemented or not defined')

