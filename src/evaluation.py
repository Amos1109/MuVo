import torch

def evaluation(loader, model):
    model.eval()
    acc, cnt = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.float().cuda(), y.long().cuda()
            feat = model.get_features(x)
            out = model.get_predictions(feat)
            pred = out.argmax(dim=1)
            acc += (pred == y).float().sum().item()
            cnt += len(x)
    model.train()
    return 100 * acc / cnt


def prediction(loader, model):
    model.eval()
    P, F = [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.cuda().float()
            F.append(model.get_features(x))
            P.append(model.get_predictions(F[-1]))
    model.train()
    return torch.vstack(P), torch.vstack(F)
