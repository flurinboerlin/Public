import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader



m = nn.Sigmoid()
input = torch.randn(1)
output = m(input)




features = torch.tensor([[1,2,3], [4,5,6]])
target = torch.tensor([[1,1],[0,1]])

model = nn.Linear(3,2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.95)


dataSet = TensorDataset(features.float(), target.float())
dataLoader = DataLoader(dataSet, batch_size = 1, shuffle = True)

num_epochs = 2

for i in range(num_epochs):
  for data in dataLoader:
    # Set the gradients to zero
    optimizer.zero_grad()
    # Run a forward pass
    feature, target = data
    prediction = model(feature)    
    # Calculate the loss
    loss = criterion(prediction, target)    
    # Compute the gradients
    loss.backward()
    # Update the model's parameters
    optimizer.step()
# show_results(model, dataloader)



