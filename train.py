import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy

data = pd.read_csv("train.csv")
#short for now
# data = data.head()
X = data.drop(["Survived"], axis=1)
# y = data[["PassengerId","Survived"]].values #has two different state classes, dead or alive
y = data[["Survived"]].values
# print(y)

# print(X.dropna())
from sklearn.impute import KNNImputer
#somehow group the classes to the cabins maybe get the percentages of people in cabin currently, and assgin like that
X["Cabin"]=X["Cabin"].replace(regex=r"[\d]+",value="")
X["Cabin"]=X["Cabin"].replace(regex=r"[ A]{2,}",value="A")
X["Cabin"]=X["Cabin"].replace(regex=r"[ B]{2,}",value="B")
X["Cabin"]=X["Cabin"].replace(regex=r"[ C]{2,}",value="C")
X["Cabin"]=X["Cabin"].replace(regex=r"[ D]{2,}",value="D")
X["Cabin"]=X["Cabin"].replace(regex=r"[ E]{2,}",value="E")
X["Cabin"]=X["Cabin"].replace(regex=r"[ F]{2,}",value="F")
X["Cabin"]=X["Cabin"].replace(regex=r"[ G]{2,}",value="G")

xx = X.drop(["Name"],axis=1)
xx["Ticket"] = xx["Ticket"].replace(regex="[\W]",value="")
xx["Ticket"] = xx["Ticket"].replace(regex="LINE",value="0")
xx["Ticket"] = xx["Ticket"].replace(regex="[a-zA-Z]",value="")
xx["Sex"] = xx["Sex"].replace(regex="female", value="1")
xx["Sex"] = xx["Sex"].replace(regex="male", value="0")
xx["Embarked"] = xx["Embarked"].replace(regex="S", value="0") #Southampton
xx["Embarked"] = xx["Embarked"].replace(regex="C", value="1") #Cherbourg
xx["Embarked"]=xx["Embarked"].replace(regex="Q",value="2") #Queenstown

xx["Cabin"]=xx["Cabin"].replace(regex=r"A",value="1")
xx["Cabin"]=xx["Cabin"].replace(regex=r"B",value="2")
xx["Cabin"]=xx["Cabin"].replace(regex=r"C",value="3")
xx["Cabin"]=xx["Cabin"].replace(regex=r"D",value="4")
xx["Cabin"]=xx["Cabin"].replace(regex=r"E",value="5")
xx["Cabin"]=xx["Cabin"].replace(regex=r"F",value="6")
xx["Cabin"]=xx["Cabin"].replace(regex=r"G",value="7")
xx["Cabin"]=xx["Cabin"].replace(regex=r"T",value="0")

# xx = pd.get_dummies(xx)
# print(xx)
# colm_Names = ["PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare","Cabin","Embarked"] # no need to convert, just send it!
impute = KNNImputer(n_neighbors=9,weights="uniform")
X_data = impute.fit_transform(xx.values)
# print(pd.DataFrame(data=X_data)) # no need to convert, just send it!
print(X_data)
device = torch.device("cuda")
#FOR NOW I AM NOT GOING TO INCLUDE THE NAMES 
X_tensor = torch.tensor(X_data,dtype=torch.float32).to(device)
y_tensor = torch.tensor(y,dtype=torch.float32).to(device)
print("X Tensor Shape",X_tensor.shape)
print("Y Tensor Shape",y_tensor.shape)


class Model(nn.Module):#   9 elements, Alive or Dead, Bc why not
    def __init__(self, input_size=10, output_size=2, hidden_layers=64):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_layers)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layers,hidden_layers)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_layers,output_size)
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x

model = Model().to(device)

criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01)

epochs = 3500
losses = []
for epoch in range(epochs):
    
    prediction = model(X_tensor)
    
    loss = criterion(prediction, y_tensor)
    
    losses.append(loss.item())
    
    optimizer.zero_grad() #reset gradients
    loss.backward() #run it through net>?
    optimizer.step()
    
    if epoch % 1000 == 0:
        print("Epoch:", epoch, "Loss:", loss.item())
print("Epoch:", epoch, "Loss:", loss.item())

torch.save(model.state_dict(), 'model_weights.pth')

with torch.no_grad():
    X = pd.read_csv("test.csv")
    X["Cabin"]=X["Cabin"].replace(regex=r"[\d]+",value="")
    X["Cabin"]=X["Cabin"].replace(regex=r"[ A]{2,}",value="A")
    X["Cabin"]=X["Cabin"].replace(regex=r"[ B]{2,}",value="B")
    X["Cabin"]=X["Cabin"].replace(regex=r"[ C]{2,}",value="C")
    X["Cabin"]=X["Cabin"].replace(regex=r"[ D]{2,}",value="D")
    X["Cabin"]=X["Cabin"].replace(regex=r"[ E]{2,}",value="E")
    X["Cabin"]=X["Cabin"].replace(regex=r"[ F]{2,}",value="F")
    X["Cabin"]=X["Cabin"].replace(regex=r"[ G]{2,}",value="G")
    
    xx = X.drop(["Name"],axis=1)
    xx["Ticket"] = xx["Ticket"].replace(regex="[\W]",value="")
    xx["Ticket"] = xx["Ticket"].replace(regex="LINE",value="0")
    xx["Ticket"] = xx["Ticket"].replace(regex="[a-zA-Z]",value="")
    # xx = pd.get_dummies(xx,["age","Sex","Embarked"]) # can't use because it gets rid of nans
    xx["Sex"] = xx["Sex"].replace(regex="female", value="1")
    xx["Sex"] = xx["Sex"].replace(regex="male", value="0")
    xx["Embarked"] = xx["Embarked"].replace(regex="S", value="0") #Southampton
    xx["Embarked"] = xx["Embarked"].replace(regex="C", value="1") #Cherbourg
    xx["Embarked"]=xx["Embarked"].replace(regex="Q",value="2") #Queenstown
    xx["Cabin"]=xx["Cabin"].replace(regex=r"A",value="1")
    xx["Cabin"]=xx["Cabin"].replace(regex=r"B",value="2")
    xx["Cabin"]=xx["Cabin"].replace(regex=r"C",value="3")
    xx["Cabin"]=xx["Cabin"].replace(regex=r"D",value="4")
    xx["Cabin"]=xx["Cabin"].replace(regex=r"E",value="5")
    xx["Cabin"]=xx["Cabin"].replace(regex=r"F",value="6")
    xx["Cabin"]=xx["Cabin"].replace(regex=r"G",value="7")
    xx["Cabin"]=xx["Cabin"].replace(regex=r"T",value="0")
    impute = KNNImputer(n_neighbors=5,weights="uniform")
    X_data = impute.fit_transform(xx.values)
    # print(xx)
    count = 0
    offset = 892
    pred_list = []
    for personData in X_data:
        # person = numpy.array([904,1,1,23,1,0,82.2667,2,0])#,make it an array
        person = personData
        info = torch.tensor(person,dtype=torch.float32).to(device)
        pred = model(info)
        # print(numpy.argmax(prediction.cpu()).item())
        # print(pred.argmax().item()) #prediction if dead or alive
        pred_list.append({
        "PassengerId": offset + count,
        "Survived": pred.argmax().item()
        })
        count += 1

predictions = pd.DataFrame(pred_list)
# predictions = predictions.rename(columns={"PassengerId":"Survived"})
predictions.to_csv("Predictions.csv",index=False)
print("Last prediction:",pred)
print(pred.argmax())
import matplotlib.pyplot as plt
plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel('Epoch')
plt.show()

