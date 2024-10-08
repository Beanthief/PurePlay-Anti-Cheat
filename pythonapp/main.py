import torch
import pandas

data = pandas.read_csv("inputs.csv", header=None)
dataTensor = torch.tensor(data.values)
print(dataTensor)