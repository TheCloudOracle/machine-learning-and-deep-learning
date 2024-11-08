# Import the dependecies
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch import nn
from pathlib import Path

import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from utils import plot_decision_boundary, train_test_loop



# Hyperparameters
INPUT_FEATURES = 2
HIDDEN_FEATURES  = 10
N_CENTERS = 9
N_SAMPLES = 1000
RANDOM_SEED = 8192
LEARNING_RATE = 1e-2


# Create the model
class MultiClassClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Create the layer stack
        self.layer_stack = nn.Sequential(
                            nn.Linear(in_features=INPUT_FEATURES, out_features=HIDDEN_FEATURES),
                            nn.ReLU(),
                            nn.Linear(in_features=HIDDEN_FEATURES, out_features=HIDDEN_FEATURES),
                            nn.ReLU(),
                            nn.Linear(in_features=HIDDEN_FEATURES, out_features=HIDDEN_FEATURES),
                            nn.ReLU(),
                            nn.Linear(in_features=HIDDEN_FEATURES, out_features=HIDDEN_FEATURES),
                            nn.ReLU(),
                            nn.Linear(in_features=HIDDEN_FEATURES, out_features=HIDDEN_FEATURES),
                            nn.ReLU(),
                            nn.Linear(in_features=HIDDEN_FEATURES, out_features=HIDDEN_FEATURES),
                            nn.ReLU(),
                            nn.Linear(in_features=HIDDEN_FEATURES, out_features=N_CENTERS),
                            nn.ReLU()
                        )

    # Override the forward method in nn.Module
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)




def main():
    print('Generating data...')
    
    X, y = make_blobs(
        n_samples=N_SAMPLES,
        centers=N_CENTERS,
        n_features=INPUT_FEATURES,
        random_state=RANDOM_SEED
    )

    print('Creating the model...')
    model = MultiClassClassificationModel()

    print('Setting up optimizer and loss function')
    loss_fn = nn.CrossEntropyLoss()
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

    MODEL_PATH_NAME = 'multiclass_classification.pth'
    MODEL_PATH_SAVE = MODEL_PATH / MODEL_PATH_NAME

    print(f'Saving the model to {MODEL_PATH_SAVE}')
    torch.save(obj=model.state_dict(), f=MODEL_PATH_SAVE)


if __name__ == '__main__':
    main()