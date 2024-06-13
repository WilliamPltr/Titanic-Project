### Librairies ### 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



### Task 1 ### 
data = pd.read_csv('titanic.csv')

def Task1():
    print(f"\nThe first five rows of the DataFrame :")
    #display the first five rows of the dataframe
    print("\n",data.head())




### Task 2 ###
#only map 'Sex' and 'Embarked' columns
data1 = data.copy()
if data1['Sex'].dtype == object:
    #map 'Sex' column: 'male' to 0 and 'female' to 1
    data1['Sex'] = data1['Sex'].map({'male': 0, 'female': 1})

#fill missing values in 'Embarked' column with the most frequent category : 'S'
data1['Embarked'] = data1['Embarked'].fillna('S')

if data1['Embarked'].dtype == object:
    #map 'Embarked' column: 'C' to 1, 'Q' to 2, 'S' to 3
    data1['Embarked'] = data1['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

#fill missing values in 'Age', 'Fare', and 'Embarked' columns with their median values
data1['Age'] = data1['Age'].fillna(data1['Age'].median())
data1['Fare'] = data1['Fare'].fillna(data1['Fare'].median())

def Task2():
    print(f"\nThe first five rows of the new DataFrame :")
    #display the first five rows of the dataframe
    print("\n",data1.head())




### Task 3 ###
#drop columns that are not useful for correlation : Name, Ticket, Cabin
data2 = data1.copy()
data2 = data2.drop(columns=['Name', 'Ticket', 'Cabin'])

def Task3A():
    print(f"\nData Frame without Name, Ticket and Cabin :")
    print("\n",data2.head())

def Task3B():
    correlation_matrix = data2.corr()
    #extract correlation with 'Survived'
    correlation_with_survived = correlation_matrix["Survived"].drop("Survived").sort_values()
    print(f"\nThe correlation coefficients with Survived in ascending order :")
    print("\n",correlation_with_survived)
    return correlation_matrix

def Task3C():
    correlation_matrix = data2.corr()
    print(f"\nThe correlation matrix :")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', vmin=-1, vmax=1, center=0, cbar_kws={'orientation': 'vertical'})
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title('Correlation Matrix', pad=20)
    plt.show()
    
def Task3D(correlation_matrix):
    #correlation_matrix = data2.corr()
    correlation_with_survived = correlation_matrix["Survived"].drop("Survived").sort_values()
    #identify the variable with the least correlation with 'Survived'
    least_correlated_variable = correlation_with_survived.abs().idxmin()
    least_correlation_value = correlation_with_survived[least_correlated_variable]
    #print the sentence identifying the least correlated variable
    print(f"\nThe variable with the least correlation with the survival status is '{least_correlated_variable}' with a correlation value of {least_correlation_value}.")
    




### Task 4 ###
def Task4():
    #separate the features (X) and the target variable (y)
    X = data2.drop(columns=['Survived'])
    y = data2['Survived']

    #split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #standardize the feature data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #convert the data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    #create DataLoader objects for batch processing
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    #define the neural network architecture
    class NeuralNet(nn.Module):
        def __init__(self, input_size):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x

    #determine the input size from the training data
    input_size = X_train.shape[1]
    model = NeuralNet(input_size)

    #define the loss function and the optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #train the neural network and store the loss values
    num_epochs = 20
    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

    #plot the training loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

    #evaluate the neural network on the test set and calculate the confusion matrix
    model.eval()
    all_predicted = []
    all_actual = []

    with torch.no_grad():
        correct = 0
        total = 0
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predicted = (outputs > 0.5).float()
            all_predicted.extend(predicted.numpy().flatten().tolist())
            all_actual.extend(y_batch.numpy().flatten().tolist())
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy * 100}%')

    #compute the confusion matrix
    cm = confusion_matrix(all_actual, all_predicted)
    cmd = ConfusionMatrixDisplay(cm, display_labels=["Did not survive", "Survived"])

    #plot the confusion matrix
    plt.figure(figsize=(8, 6))
    cmd.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()


