# NAME: Chichari Anusha
# ROLL: 21CH10020

# In[108]:


# Importing all the libraries required for us
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import seaborn as sns


# In[109]:


# loading the dataset
df = pd.read_csv('Iris.csv')


# In[110]:


# Encoding the 'Species' column to numerical labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Species'] = label_encoder.fit_transform(df['Species'])


# In[111]:


# Splitting the dataset into features (X) and target (y)
X = df.iloc[:, 1:5].values  # Assuming columns 1 to 4 are the features
y = df['Species'].values


# In[112]:


# Splitting the dataset into train, validation, and test sets (60:20:20 ratio)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# In[113]:


# Defining the softmax function to compute class probabilities
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# In[114]:


# Implementing logistic regression with gradient descent
def logistic_regression(X, y, learning_rate, num_epochs, batch_size):
    num_classes = len(np.unique(y))
    num_features = X.shape[1]
    m = X.shape[0]

    # Initialize weights and bias
    W = np.zeros((num_features, num_classes))
    b = np.zeros((1, num_classes))

    for epoch in range(num_epochs):
        for i in range(0, m, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            # Compute logits
            z = np.dot(X_batch, W) + b

            # Compute softmax
            y_pred = softmax(z)

            # Compute gradients
            gradient = (1 / batch_size) * np.dot(X_batch.T, (y_pred - np.eye(num_classes)[y_batch]))

            # Update weights and bias
            W -= learning_rate * gradient
            b -= learning_rate * np.sum(y_pred - np.eye(num_classes)[y_batch], axis=0)

    return W, b


# In[115]:


# Befter feature scaling
# Training the logistic regression model
learning_rate = 0.1
num_epochs = 50
batch_size = 30

W, b = logistic_regression(X_train, y_train, learning_rate, num_epochs, batch_size)

# Evaluating the model on the validation set
z_val = np.dot(X_val, W) + b
y_val_pred = softmax(z_val)

# Calculating predictions and metrics
y_val_pred_classes = np.argmax(y_val_pred, axis=1)
accuracy = accuracy_score(y_val, y_val_pred_classes)
conf_matrix = confusion_matrix(y_val, y_val_pred_classes)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_val, y_val_pred_classes, average='macro')

# Printing evaluation metrics before feature scaling
print("Before feature scaling: ")
print(f"Accuracy on validation set: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")


# In[116]:


# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# In[117]:


# After feature scaling
# Training the logistic regression model
learning_rate = 0.1
num_epochs = 50
batch_size = 30

W, b = logistic_regression(X_train, y_train, learning_rate, num_epochs, batch_size)


# In[118]:


# Evaluating the model on the validation set
z_val = np.dot(X_val, W) + b
y_val_pred = softmax(z_val)


# In[119]:


# Calculating predictions and metrics
y_val_pred_classes = np.argmax(y_val_pred, axis=1)
accuracy = accuracy_score(y_val, y_val_pred_classes)
conf_matrix = confusion_matrix(y_val, y_val_pred_classes)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_val, y_val_pred_classes, average='macro')


# In[120]:


# Printing evaluation metrics after feature scaling
print("After feature scaling: ")
print(f"Accuracy on validation set: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")


# In[121]:


# Experiment-1

# Defining the range of learning rates to test
learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1]

# Initialising lists to store learning rates and corresponding validation accuracies
lr_list = []
accuracy_list = []

for learning_rate in learning_rates:
    # Training the logistic regression model with the current learning rate
    W, b = logistic_regression(X_train, y_train, learning_rate, num_epochs, batch_size)

    # Evaluating the model on the validation set
    z_val = np.dot(X_val, W) + b
    y_val_pred = softmax(z_val)

    # Calculating accuracy on the validation set
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
    accuracy = accuracy_score(y_val, y_val_pred_classes)

    # Storing the results
    lr_list.append(learning_rate)
    accuracy_list.append(accuracy)


# In[122]:


# Finding the best learning rate
best_learning_rate = lr_list[np.argmax(accuracy_list)]
best_accuracy = max(accuracy_list)

print("Learning Rate vs. Validation Accuracy:")
for lr, acc in zip(lr_list, accuracy_list):
    print(f"Learning Rate: {lr:.6f}, Accuracy: {acc:.4f}")

print(f"Best Learning Rate: {best_learning_rate:.6f}, Best Accuracy: {best_accuracy:.4f}")


# In[123]:


# Plotting the accuracy vs. learning rate
plt.figure(figsize=(10, 6))
plt.semilogx(lr_list, accuracy_list, marker='o', linestyle='-')
plt.title("Learning Rate vs. Validation Accuracy")
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.show()


# In[124]:


# Experiment-2

# Training the logistic regression model with the optimal learning rate
optimal_learning_rate = best_learning_rate  # Replacing with the best learning rate from Experiment 1
W, b = logistic_regression(X_train, y_train, optimal_learning_rate, num_epochs, batch_size)


# In[125]:


# Segregating the training data into three different sets based on true class labels
class_0_indices = np.where(y_train == 0)[0]
class_1_indices = np.where(y_train == 1)[0]
class_2_indices = np.where(y_train == 2)[0]


# In[126]:


# Initialising lists to store class probabilities for each class and each set after each epoch
class_0_probs = []
class_1_probs = []
class_2_probs = []

for epoch in range(num_epochs):
    # Calculating class probabilities for each data point using the updated weights and biases
    z_train = np.dot(X_train, W) + b
    y_train_pred = softmax(z_train)

    # Calculating mean probabilities for each class in each set
    class_0_mean_prob = np.mean(y_train_pred[class_0_indices], axis=0)
    class_1_mean_prob = np.mean(y_train_pred[class_1_indices], axis=0)
    class_2_mean_prob = np.mean(y_train_pred[class_2_indices], axis=0)

    # Appending the mean probabilities to the respective lists
    class_0_probs.append(class_0_mean_prob)
    class_1_probs.append(class_1_mean_prob)
    class_2_probs.append(class_2_mean_prob)


# In[127]:


# Plotting the average class probabilities vs. epochs for each class and each set
plt.figure(figsize=(12, 8))
epochs = np.arange(1, num_epochs + 1)

plt.subplot(3, 1, 1)
plt.plot(epochs, class_0_probs)
plt.title("Average Class Probabilities for Iris-setosa")
plt.xlabel("Epochs")
plt.ylabel("Mean Probability")
plt.legend(["Class 0", "Class 1", "Class 2"])

plt.subplot(3, 1, 2)
plt.plot(epochs, class_1_probs)
plt.title("Average Class Probabilities for Iris-versicolor")
plt.xlabel("Epochs")
plt.ylabel("Mean Probability")
plt.legend(["Class 0", "Class 1", "Class 2"])

plt.subplot(3, 1, 3)
plt.plot(epochs, class_2_probs)
plt.title("Average Class Probabilities for Iris-virginica")
plt.xlabel("Epochs")
plt.ylabel("Mean Probability")
plt.legend(["Class 0", "Class 1", "Class 2"])

plt.tight_layout()
plt.show()


# In[131]:


# Experiment-3

from sklearn.metrics import confusion_matrix, classification_report

# Defining a function to evaluate the model
def evaluate_model(X, y, W, b):
    # Calculate class probabilities for each instance
    z = np.dot(X, W) + b
    y_pred = softmax(z)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate confusion matrix
    confusion_mat = confusion_matrix(y, y_pred_classes)
    # Printing confusion matrix
    print("Confusion matrix is: ")
    print(confusion_mat)
    
    # Plot confusion matrix as a heatmap
    print("Printing confusion matrix as heatmap: ")
    class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"] 
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Calculate and print classification report
    classification_rep = classification_report(y, y_pred_classes, target_names=class_names)
    print("Classification Report:")
    print(classification_rep)


# In[132]:


# Training the model with the best learning rate obtained from Experiment 1
W_best, b_best = logistic_regression(X_train, y_train, best_learning_rate, num_epochs, batch_size)


# In[133]:


# Evaluating the model with the optimal learning rate (best_learning_rate)
evaluate_model(X_test, y_test, W_best, b_best)







