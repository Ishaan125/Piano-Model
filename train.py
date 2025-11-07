import torch

class train:
    def __init__(self, model, train_loader, test_loader, loss_func, optimizer, epochs):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = loss_func
        self.optimizer = optimizer
        self.epochs = epochs

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch.targets)
                loss.backward()
                self.optimizer.step()
            self.test()

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch.targets)
                # Compute test metrics