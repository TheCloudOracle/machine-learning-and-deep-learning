import torch
import matplotlib.pyplot as plt
import numpy as np

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



def plot_predictions(train_features,
                   train_labels,
                   test_features,
                   test_labels,
                   predictions=None):
    """
    Use this method for data visualization.

    Parameters
    -----------
    train_features: torch.Tensor
        A portion of the features that is used for training
    train_labels: torch.Tensor
        A portion of the labels that is used for training
    test_features: torch.Tensor
        A portion of the features that is used for testing
    test_labels: torch.Tensor
        A portion of the labels that is used for testing

    Returns
    --------
    None
    """
    # Scatter plot the training data in blue
    plt.scatter(x=train_features,
                y=train_labels,
                c='b',
                s=5,
                label='Training data')
    # Scatter plot the testing data in green
    plt.scatter(x=test_features,
                y=test_labels,
                c='g',
                s=5,
                label='Testing data')
    # Check if there are any predictions
    if predictions is not None:
        # Scatter plot the prediction data in red
        plt.scatter(x=test_features,
                    y=predictions.detach().numpy(),
                    c='r',
                    s=5,
                    label='Prediction Data')
    # Show the legend
    plt.legend(prop={'size': 10})
    # Show the graph
    plt.show()


# The accuracy metric
def accuracy(y_pred, y_true):
    correct = torch.eq(y_pred, y_true).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc


def train_test_loop(model_, loss_fn_, optimizer_, train_features, train_labels, test_features, test_labels):
    epochs = int(input('Please enter the number of epochs to train for: '))
    model_type = ''
    if len(torch.unique(train_labels)) > 2:
        model_type = 'mc'
    else:
        model_type = 'bc'

    for epoch in range(epochs):
        #### Training ####
        model_.train() # Put the model in training mode
        # Forward pass
        if model_type == 'mc':
            train_logits = model_(train_features)
            train_prop_preds = torch.softmax(train_logits, dim=1).argmax(dim=1)
        else:
            train_logits = model_(train_features).squeeze()
            train_prop_preds = torch.round(torch.sigmoid(train_logits))
        # Calculate loss and accuracy
        train_loss = loss_fn_(train_logits, train_labels.long())
        train_acc = accuracy(train_prop_preds, train_labels)
        # Optimizer zero grad
        optimizer_.zero_grad()
        # Backpropagation
        train_loss.backward()
        # Optimizer step
        optimizer_.step()

        #### Testing ####
        model_.eval()
        with torch.inference_mode(): # Turn off gradient tracking
            # Forward pass
            test_logits = model_(test_features).squeeze()
            if model_type == 'mc':
                test_prop_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)
            else:
                test_prop_preds = torch.round(torch.sigmoid(test_logits))
            # Calculate loss and accuracy
            test_loss = loss_fn_(test_logits, test_labels.long())
            test_acc = accuracy(test_prop_preds, test_labels)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch #{epoch + 1} | Train Loss: {train_loss:.5f}, Train Accuracy: {train_acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%')

