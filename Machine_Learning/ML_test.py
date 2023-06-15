# 
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

repoDir = "C:/Users/fluri/OneDrive/Documents/github/Public/"
creditDf = pd.read_csv(f'{repoDir}Machine_Learning/02_Data/AER_credit_card_data.csv')
# FB: Remap 'yes' and 'no' strings to binary
creditDf[['card', 'owner', 'selfemp']] = creditDf[['card', 'owner', 'selfemp']].replace({'yes':1, 'no':0})

# FB: 'card': Whether credit card request was requested
# target = torch.tensor(creditDf['card']).type(torch.LongTensor)
target = torch.tensor(creditDf['card'])
# FB:
# 'reports': Amount of major derogatory/past due reports
# 'age': Age in years
# 'income': Yearly income / 10,000
# 'share': Ratio of monthly credit card expenditure to yearly income
# 'owner': 1 if homeowner
# 'selfemp': 1 if self-employed
# 'months': Monthls living at current address
# 'majorcards': Number of major credit cards held
# 'active': Number of active credit accounts
# FB: Two things to watch out for: i) Ensure that all of the columns have the same data type, ii) transform the pandas data frame to numerical values
# through either .values or .to_numpy()
features = torch.tensor(creditDf[['reports', 'age', 'income', 'share', 'owner', 'selfemp', 'months', 'majorcards', 'active']].to_numpy())


# FB: First entry of the size() call to a torch object is the amount of observations (N), second entry is the amount of features per observation (X).
num_features = features.size()[1]
# FB: The number of classes is the amount of outputs we need to draw a conclusion. If we have a binary outcome, we only need one output, ergo one class. 
# Same for a regression. For a multinomial outcome, we would need an amount of classes equal to the amount of outcomes to show the individual probability
# of every outcome.
# num_classes = 2

# FB: The error message 'torch.nn.modules.activation.Sigmoid is not a Module subclass' is caused by defining nn.Sigmoid instead of nn.Sigmoid()
# FB: The nn.Sigmoid() function needs more than one input node.
model = nn.Sequential(nn.Linear(num_features, 1),
#                     #   nn.Linear(4, 1), 
                      nn.Sigmoid())
# model = nn.Linear(num_features, 1)


# To get the amunt of parameters in a model (degrees of freedom) we can use the following below code snippets:
# total = 0
# for parameter in model.parameters():
# total += parameter.numel()
# print(total)

# FB: We use CrossEntropyLoss() for classification problems and MSELoss() is used for regression problems.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.95)

# FB: VERY IMPORTANT: The combination of having target defined as a pandas data frame slice AND loading target.float() into the TensorDataset()
# function leads to an issue with aligning the data types for the CrossEntropyLoss function. Features on the other hand has to be defined as a float().
dataSet = TensorDataset(features.float(), target)
dataLoader = DataLoader(dataSet, batch_size = 1, shuffle = True)

num_epochs = creditDf.shape[0]
# FB: Loop through the data set spanned by dataLoader
for epoch in range(num_epochs):
    for data in dataLoader:
        # Set gradients to zero
        optimizer.zero_grad()
        # Get feature and target from the data loader
        feature, target = data
        # Run a forward pass
        prediction = model(feature)
        # Compute loss and gradients
        loss = criterion(prediction, target)
        loss.backward()
        # Update the parameters
        optimizer.step()

# show_results(model, dataLoader)






# FB: Use a learning rate of 10^-4 as default.
optimizer = optim.SGD(model.parameters(), lr = 0.001)

# FB: Accessing the layer gradients:
model[0].weight.grad, model[0].bias.grad
model[1].weight.grad, model[1].weight.grad

F.one_hot(torch.tensor(0), num_classes = 3)




output = model(input_data)
# nn.Softmax() is equivalent for multinomial outcomes. nn.Softmax(dim=-1) can be seen as the default option since that parameterisation leads to
# Softmax() being applied to the last layer, which is what we traditionally want to do.

print('end')