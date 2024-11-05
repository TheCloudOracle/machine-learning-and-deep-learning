# Import the dependecies
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from torch import nn
from pathlib import Path


# Hyperparameters
INPUT_FEATURES = 2
OUTPUT_FEATURES = 1
HIDDEN_FEATURES = 10
LEARNING_RATE = 1e-2
RANDOM_SEED = 8192
N_SAMPLES = 1000

# Create the model
class BinaryClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Feel free to play around with the layers
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=INPUT_FEATURES, out_features=HIDDEN_FEATURES),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_FEATURES, out_features=HIDDEN_FEATURES),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_FEATURES, out_features=OUTPUT_FEATURES),
            nn.ReLU()
        )

    # Override the forward method in nn.Module
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x) # x -> layer_stack() -> output


# The accuracy metric
def accuracy(y_pred, y_true):
    correct = torch.eq(y_pred, y_true).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc


# Decision boundary
def plot_decision_boundary(model, X, y):
    # Momve everything to the CPU
    model.to('cpu')
    X, y = X.to('cpu'), y.to('cpu')

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

        # Test for multi-class or binary and adjust logits to prediction labels
        if len(torch.unique(y)) > 2:
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # Multi-class
        else:
            y_pred = torch.round(torch.sigmoid(y_logits)) # Binary

        # Reshape preds and plot
        y_pred = y_pred.reshape(xx.shape).detach().numpy()
        plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

def train_test_loop(model_, loss_fn_, optimizer_, train_features, train_labels, test_features, test_labels):
    epochs = int(input('Enter the number of epochs: '))
    
    for epoch in range(epochs):
        #### Training ####
        model_.train() # Put the model in training mode
        # Forward pass
        train_logits = model_(train_features).squeeze()
        train_prob_preds = torch.round(torch.sigmoid(train_logits))
        # Calculate loss and accuracy
        train_loss = loss_fn_(train_logits, train_labels)
        train_acc = accuracy(train_prob_preds, train_labels)
        # Optimizer zero grad
        optimizer_.zero_grad()
        # Backpropagation
        train_loss.backward()
        # Optimizer step 
        optimizer_.step()

        #### Testing ####
        model_.eval()  # Put the model in evaluation mode
        with torch.inference_mode(): # Turn off gradient tracking
            # Forward pass
            test_logits = model_(test_features).squeeze()
            test_prob_preds = torch.round(torch.sigmoid(test_logits))
            # Calculate loss and accuracy
            test_loss = loss_fn_(test_logits, test_labels)
            test_acc = accuracy(test_prob_preds, test_labels)

            if (epoch + 1) % 100 == 0:
                print(f'Epoch #{epoch} | Train Loss: {train_loss:.5f}, Train Accuracy: {train_acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%')

def main():
    print('Generating data...')
    
    X, y = make_circles(
        n_samples=N_SAMPLES,
        noise=0.03,
        random_state=RANDOM_SEED
    )

    print('Creating the model...')
    model = BinaryClassificationModel()

    print('Setting up optimizer and loss function')
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

    print('Displaying data...')
    plt.scatter(x=X[:, 0],
                y=X[:, 1],
                c=y,
                cmap=plt.cm.RdYlBu)
    plt.show()
    print('Displaying numerical data...\n\n')
    print('Sampling the data')
    X_sample = X[0]
    y_sample = y[0]
    print(f'Value of the first feature: \n{X_sample}')
    print(f'Value of the first label: \n{y_sample}\n\n') 
    print(f'Shape of the first feature: \n{X_sample.shape}')
    print(f'Shape of the first label: \n{y_sample.shape}\n\n')
    print(f'Shape of all the features: \n{X.shape}')
    print(f'Shape of all the labels: \n{y.shape}\n\n')
    print(f'Datatype of the features: \n{X.dtype}')
    print(f'Type of the features: \n{type(X)}\n\n')
    print(f'Datatype of the labels: \n{y.dtype}')
    print(f'Type of the labels: \n{type(y)}\n\n')

    print('Transforming the data...')
    try:
        X = torch.from_numpy(X).type(torch.float)
        y = torch.from_numpy(y).type(torch.float)
    except TypeError:
        pass
    finally:
        print(f'New datatype of the features: \n{X.dtype}')
        print(f'New type of the features: \n{type(X)}\n\n')
        print(f'New datatype of the labels: \n{y.dtype}')
        print(f'New type of the labels: \n{type(y)}\n\n')

    print('Creating train test split...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_SEED, train_size=0.8)

    print('Training and test...')
    train_test_loop(model, loss_fn, optimizer, X_train, y_train, X_test, y_test)
    
    # Display the decision boundary
    plt.subplot(1, 2, 1)
    plt.title('Train')
    plot_decision_boundary(model, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title('Test')
    plot_decision_boundary(model, X_test, y_test)
    plt.show()

    # Save the model
    MODEL_PATH = Path('./')
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_PATH_NAME = 'binary_classification.pth'
    MODEL_PATH_SAVE = MODEL_PATH / MODEL_PATH_NAME

    print(f'Saving the model to {MODEL_PATH_SAVE}')
    torch.save(obj=model.state_dict(), f=MODEL_PATH_SAVE)


if __name__ == '__main__':
    main()