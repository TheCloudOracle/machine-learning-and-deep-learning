# Import dependecies
import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path
from utils import plot_predictions

# Hyperparameters
RANDOM_SEED = 8192
INPUT_FEATURES = 1
OUTPUT_FEATURES = 1
LEARNING_RATE = 0.01

# Lists for plotting loss curves
epoch_count = []
train_loss_values = []
test_loss_values = []


# Create the model
class LinearRegressionModel(nn.Module):
    """
    The linear regression model. 

    To ensure reproducibility, we include a RANDOM_SEED that you can alter at anytime.
    The hyperparameters INPUT_FEATURES and OUTPUT_FEATURES will define the structure of the model.
    """
    def __init__(self):
        super().__init__()
        # Set a random seed
        torch.manual_seed(RANDOM_SEED)
        # Create the linear layer
        self.linear_layer = nn.Linear(in_features=INPUT_FEATURES, out_features=OUTPUT_FEATURES)

    # Override the forward method in nn.Module
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


def train_test_split(X:torch.Tensor, y:torch.Tensor):
    """
    A very basic train test split. Splits the data into 80% training
    data and 20% testing data.

    Parameters
    -----------
    X: torch.Tensor
        A tensor of all the features
    y: torch.Tensor
        A tensor of all the labels

    Returns
    --------
    X_train: torch.Tensor
        A portion of the features that is used for training
    X_test: torch.Tensor
        A portion of the features that is used for testing
    y_train: torch.Tensor
        A portion of the labels that is used for training
    y_test: torch.Tensor
        A portion of the labels that is used for testing
    """
    split = int(0.8 * len(X)) # This will account for 80% of the data

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test



def visualize(predictions=None):
    plot_data = input('Do you want to plot the data [Y/N]: ')
    while plot_data.lower()[0] not in ['y', 'n']:
        plot_data = input('Invalid input. Do you want to plot your data [Y/N]: ')

    if plot_data.lower()[0] == 'y':
        plot_predictions(X_train, y_train, X_test, y_test, predictions)


def train_and_test(_model, train_features, train_labels, test_features, test_labels, loss_function, _optimizer):
    """
    Runs a training loop on the data

    Parameters
    -----------
    _model: LinearRegressionModel
        The model in question
     train_features: torch.Tensor
        A portion of the features that is used for training
    train_labels: torch.Tensor
        A portion of the labels that is used for training
    test_features: torch.Tensor
        A portion of the features that is used for testing
    test_labels: torch.Tensor
        A portion of the labels that is used for testing
    loss_function: torch.nn.modules.loss.L1Loss
        The loss function that will be used in training and testing
    _optimizer: torch.optim.sgd.SGD
        The optimizer that will be used in training
    
    Returns
    --------
    None
    """

    epochs = int(input('How many epochs should the model train for: '))
    
    for epoch in range(epochs):
        #### Training ####
        _model.train() # Put the model in training mode
        # Forward pass
        train_preds = _model(train_features)
        # Calculate loss
        train_loss = loss_function(train_preds, train_labels)
        # Optimizer zero grad
        _optimizer.zero_grad()
        # Backpropagation
        train_loss.backward()
        # Optimizer step zero
        _optimizer.step()

        #### Testing ####
        _model.eval() # Put the model in evaluation mode
        with torch.inference_mode(): # Disable gradient tracking
            # Forward pass
            test_preds = _model(test_features)
            # Calculate loss
            test_loss = loss_function(test_preds, test_labels)

            if epoch % 10 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(train_loss.detach().numpy())
                test_loss_values.append(test_loss.detach().numpy())
                print(f'Epoch #{epoch} | Train Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f}')


def make_predictions(_model, test_features):
    """
    Make predictions after training and testing

    Parameters
    -----------
    _model: LinearRegressionModel
        The model in question
    test_features: torch.Tensor
        A portion of the features that is used for testing
    
    Returns
    --------
    None
    """
    predictions = _model(test_features)
    visualize(predictions)


def plot_loss_curves():
    plt.plot(epoch_count, train_loss_values, label='Training Loss Curve')
    plt.plot(epoch_count, test_loss_values, label='Testing Loss Curve')
    plt.title('Training and testing loss curves')
    plt.xlabel('Epoch Count')
    plt.ylabel('Loss Values')
    plt.legend(prop={'size': 10})
    plt.show()


# __main__ section
if __name__ == '__main__':
    # Create the data (ask from user)
    start = float(input('Please enter start value: '))
    end = float(input('Please enter the end value: '))
    step = float(input('Please enter the step attribute: '))
    print('Generating data...')
    X = torch.arange(start=start, end=end, step=step).unsqueeze(dim=1)

    weight = float(input('Please enter the weight: '))
    bias = float(input('Please enter the bias: '))
    y = weight * X + bias
    print(f'Formula generated: y = {weight}x + {bias}')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    visualize()

    # Create an instance of the model then train and test
    print('Creating the machine learning model...')
    model = LinearRegressionModel()
    print('Setting up loss function and optimizer')
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
    print('Making untrained predictions')
    with torch.inference_mode():
        untrained_preds = model(X_test)
    visualize(untrained_preds)
    print('Training the model')
    train_and_test(model, X_train, y_train, X_test, y_test, loss_fn, optimizer)
    make_predictions(model, X_test)
    plot_loss_curves()

    # Save the model
    MODEL_PATH = Path('./')
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_PATH_NAME = 'linear_regression_model.pth'
    MODEL_PATH_SAVE = MODEL_PATH / MODEL_PATH_NAME

    print(f'Saving the model to {MODEL_PATH_SAVE}')
    torch.save(f=MODEL_PATH_SAVE, obj=model.state_dict())
