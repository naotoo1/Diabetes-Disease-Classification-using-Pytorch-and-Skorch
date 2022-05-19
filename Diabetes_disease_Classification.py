import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load data set
data = pd.read_csv('File from your local directory')

# Check for null values
print(data.isnull().sum())

# Summary of data
print(data)

#  Split data into Train and Test set
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#  Convert data into Float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


# Build ANN
class ANN_Module(nn.Module):
    def __init__(self, m, input_features=8, hidden1=20, hidden2=20, hidden3=20, output_features=2):
        super(ANN_Module, self).__init__()
        self.m = m
        self.f_connected1 = nn.Linear(input_features, hidden1)
        self.dropout = nn.Dropout(0.2)
        self.f_connected2 = nn.Linear(hidden1, hidden2)
        self.f_connected3 = nn.Linear(hidden2, hidden3)
        self.out = nn.Linear(hidden3, output_features)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = F.relu(self.f_connected1(x))
        x = self.dropout(x)
        x = F.relu(self.f_connected2(x))
        x = self.dropout(x)
        x = F.relu(self.f_connected3(x))
        x = self.dropout(x)
        if self.m == 'm':
            x = self.out(x)
            return x
        if self.m == 'a':
            x = self.softmax(self.out(x))
            return x


# Seed for reproducible results
torch.manual_seed(20)

# Create the model
model = ANN_Module(m='m')

# Summary of model parameters
print(model.parameters)

# Set loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the ANN model using the train set
epoch = 500
final_losses = []
for i in range(epoch):
    i = i + 1
    y_pred = model.forward(X_train)
    loss = loss_function(y_pred, y_train)
    final_losses.append(loss)
    if i % 10 == 1:
        print('Epoch number:{} and the loss:{}'.format(i, loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Visualize training performance
plt.plot(range(epoch), final_losses)
plt.xlabel('Train Loss')
plt.ylabel('Number of epochs')
plt.show()

# Make Predictions using the ANN model
final_prediction = []
with torch.no_grad():
    for i, datas in enumerate(X_test):
        y_pred = model(datas)
        final_prediction.append(y_pred.argmax().item())
final_prediction = torch.tensor(final_prediction)
print(final_prediction)
print(y_test)

# Accuracy of the model for the test set
acc = accuracy_score(y_test, final_prediction)
print(acc)

# summary of confusion matrix
cm = confusion_matrix(y_test, final_prediction)
print(cm)

# Visualize the confusion matrix
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True)
plt.show()

# Save model
torch.save(model, 'diabetes.pt')

# Load model
model1 = torch.load('diabetes.pt')

# Summary of model evaluation
print(model.eval())

# Predict using the saved model
new_data = X[2:10]
new_data = torch.FloatTensor(new_data)
pred = [model1(data).argmax().item() for data in new_data]
print(pred)

# Automate the ANN for optimal hyperparameter search
net = NeuralNetClassifier(
    ANN_Module(m='a'),
    max_epochs=10,
    lr=0.1,
    # optimizer=torch.optim.Adam, criterion=torch.nn.CrossEntropyLoss(),
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
)

# Fit the model to train set
net.fit(X_train, y_train)
y_proba = net.predict_proba(X_test)

# Set paremeters for optimal search
from sklearn.model_selection import GridSearchCV

net.set_params(train_split=False, verbose=0)
params = {
    'lr': [0.01, 0.02],
    'max_epochs': [10, 500],
    'module__hidden1': [20, 50],
    'module__hidden2': [20, 50],
    'module__hidden3': [20, 50],
    'module__m': ['a'],
}
gs = GridSearchCV(net, params, refit=True, cv=3, scoring='accuracy', verbose=2)

# Fit the model to the train data
gs.fit(X_train, y_train)

# Summary of best parameters
print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))
d = list(gs.best_params_.values())

# Predict test set
y_pred = gs.predict(X_test)
print(accuracy_score(y_test, y_pred))

# Save the ANN model for the GridSearch
torch.save(gs, 'gs_ann.pt')

# Load saved model
model2 = torch.load('gs_ann.pt')

# Predict using the saved model
new_data = X[2:10]
new_data = torch.FloatTensor(new_data)
pred = model2.predict(new_data)
print(pred)




