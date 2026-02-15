import torch

def monte_carlo_dropout(model,x,passes=10):
    model.train()
    preds=[]
    for _ in range(passes):
        preds.append(model(x))
    stack=torch.stack(preds)
    return stack.mean(0), stack.std(0)