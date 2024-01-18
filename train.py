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

xx = X.drop(["Name","Ticket"],axis=1)
# xx = pd.get_dummies(xx,["age","Sex","Embarked"]) # can't use because it gets rid of nans
xx["Sex"] = xx["Sex"].replace(regex="female", value="1")
xx["Sex"] = xx["Sex"].replace(regex="male", value="0")
xx["Embarked"] = xx["Embarked"].replace(regex="S", value="0") #Southampton
xx["Embarked"] = xx["Embarked"].replace(regex="C", value="1") #Cherbourg
xx["Embarked"]=xx["Embarked"].replace(regex="Q",value="2") #Queenstown

cabDict={
    "A":1,
    "B":2,
    "C":3,
    "D":4,
    "E":5,
    "F":6,
    "G":7,
    "T":0, # 0 because its the deck
}

# for val in xx["Cabin"]:
# for val in xx.items("Cabin"):
#     if "nan" in str(val):
#         pass
#     else:
#         if match(r"[A-Z]",val): #ie: fg, [f], [g], choose the first one, so f
#             valLetter = findall(r"[A-Z]",val)[0]
#             NewVal = cabDict[valLetter]
#             xx.at["Cabin",val] = NewVal
## Hashtag: IGIVEUPTIMEFORTHEDIRTYBUTEASYWAY
xx["Cabin"]=xx["Cabin"].replace(regex=r"A",value="1")
xx["Cabin"]=xx["Cabin"].replace(regex=r"B",value="2")
xx["Cabin"]=xx["Cabin"].replace(regex=r"C",value="3")
xx["Cabin"]=xx["Cabin"].replace(regex=r"D",value="4")
xx["Cabin"]=xx["Cabin"].replace(regex=r"E",value="5")
xx["Cabin"]=xx["Cabin"].replace(regex=r"F",value="6")
xx["Cabin"]=xx["Cabin"].replace(regex=r"G",value="7")
xx["Cabin"]=xx["Cabin"].replace(regex=r"T",value="0")


print(xx)
# colm_Names = ["PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare","Cabin","Embarked"] # no need to convert, just send it!
impute = KNNImputer(n_neighbors=5,weights="uniform")
X_data = impute.fit_transform(xx.values)
# print(pd.DataFrame(data=X_data)) # no need to convert, just send it!

device = torch.device("cuda")
#FOR NOW I AM NOT GOING TO INCLUDE THE NAMES AND TICKETS
X_tensor = torch.tensor(X_data,dtype=torch.float32).to(device)
y_tensor = torch.tensor(y,dtype=torch.float32).to(device)

class Model(nn.Module):#   9 elements, Alive or Dead, Bc why not
    def __init__(self, input_size=9, output_size=2, hidden_layers=32):
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
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000

for epoch in range(epochs):
    
    prediction = model(X_tensor)
    
    loss = criterion(prediction, y_tensor)
    
    optimizer.zero_grad() #reset gradients
    loss.backward() #run it through net>?
    optimizer.step()
    
    if epoch % 100 == 0:
        print("Epoch:", epoch, "Loss:", loss.item())

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

    xx = X.drop(["Name","Ticket"],axis=1)
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
    print(xx)
    count = 0
    offset = 892
    pred_list = []
    for personData in X_data:
        # person = numpy.array([904,1,1,23,1,0,82.2667,2,0])#,make it an array
        person = personData
        info = torch.tensor(person,dtype=torch.float32).to(device)
        pred = model(info)
        # print(pred.argmax().item()) #prediction if dead or alive
        pred_list.append({
        "PassengerId": offset + count,
        "Survived": pred.argmax().item()
        })
        count += 1

predictions = pd.DataFrame(pred_list)
# predictions = predictions.rename(columns={"PassengerId":"Survived"})
predictions.to_csv("Predictions.csv",index=False)