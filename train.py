import torch
import torch.optim as optim
from model import DomainTransformer
from preprocess import get_data_loaders
import torch.nn.functional as F
import json

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for src, tgt in train_loader:
        src = src.to(device).transpose(0, 1)
        tgt = tgt.to(device).transpose(0, 1)
        tgt_input = tgt[:-1, :]
        targets = tgt[1:, :].contiguous().view(-1)

        optimizer.zero_grad()
        output = model(src, tgt_input)
        output = output.view(-1, output.size(-1))

        loss = F.cross_entropy(output, targets, ignore_index=0)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, test_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in test_loader:
            src = src.to(device).transpose(0, 1)
            tgt = tgt.to(device).transpose(0, 1)
            tgt_input = tgt[:-1, :]
            targets = tgt[1:, :].contiguous().view(-1)

            output = model(src, tgt_input)
            output = output.view(-1, output.size(-1))
            loss = F.cross_entropy(output, targets, ignore_index=0)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def main():
    train_loader, test_loader, token_to_id = get_data_loaders('training_set.csv')
    vocab_size = len(token_to_id)
    model = DomainTransformer(vocab_size)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, 11):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = evaluate(model, test_loader, device)
        print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')
    
    save_path = 'model.pth'
    torch.save(model.state_dict(), save_path)

    mapping_save_path = 'token_to_id.json'
    with open(mapping_save_path, 'w') as f:
        json.dump(token_to_id, f)


if __name__ == '__main__':
    main()
