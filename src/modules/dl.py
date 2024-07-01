import torch
from tqdm import tqdm
from livelossplot import PlotLosses
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device('cpu')

def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler=None, visualize=None, device=device, save=None):
    """
    This function trains a PyTorch model using the given data loaders and hyperparameters.
    The function also supports visualization using liveloss or matplotlib.

    Parameters:
    - model: object: A PyTorch model object.
    - train_loader: DataLoader: A DataLoader object containing the training data.
    - val_loader: DataLoader: A DataLoader object containing the validation data.
    - criterion: object: A loss function object.
    - optimizer: object: An optimizer object.
    - epochs: int: The number of epochs to train the model.
    - scheduler: object: A learning rate scheduler object.
    - visualize: str: A string that determines the visualization method. Choose from [None, 'liveloss', 'matplotlib'].
    - save_path: str: A string containing the path to save the best model.

    Returns:
    Model performance metrics and visualization.
    """
    best_val_loss = float('inf')
    best_val_accuracy = float('inf')
    best_model_wts = None
    
    assert visualize in [None, 'liveloss', 'matplotlib'], "Invalid visualization method. Choose from [None, 'liveloss', 'matplotlib']."

    if visualize == 'liveloss':
        liveloss = PlotLosses()

    if visualize == 'matplotlib':
        train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Training Phase
        logs = {}
        model.train()
        train_loss, train_corrects, train_total = 0.0, 0, 0

        for data, labels in tqdm(train_loader, desc="Training", leave=False):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels.data)
            train_total += data.size(0)

        train_loss /= len(train_loader.dataset)
        train_accuracy = train_corrects.double() / train_total

        model.eval()
        val_loss, val_corrects, val_total = 0.0, 0, 0

        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc="Validation", leave=False):
                data, labels = data.to(device), labels.to(device)

                outputs = model(data)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * data.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_total += data.size(0)

        val_loss /= len(val_loader.dataset)
        val_accuracy = val_corrects.double() / val_total

        if visualize != 'liveloss':
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
            print(f"  Val Loss: {val_loss:.4f} |   Val Acc: {val_accuracy:.4f}")

        if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Save logs for visualization
        if visualize == 'liveloss':
            logs['log loss'] = train_loss
            logs['accuracy'] = train_accuracy.item()
            logs['val_log loss'] = val_loss
            logs['val_accuracy'] = val_accuracy.item()
            liveloss.update(logs)
            liveloss.send()
        elif visualize == 'matplotlib':
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy.cpu())  # Move to CPU
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy.cpu())  # Move to CPU

        # Check if current model is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_model_wts = model.state_dict().copy()
            print(f"👉 New best model with val loss: {val_loss:.4f}!")

        print("-" * 30)

    # Save best model
    if save and best_model_wts:
        torch.save(best_model_wts, save)
        print(f"Best model saved:")
        print(f"Val Loss: {best_val_loss:.4f} | Val Acc: {best_val_accuracy:.4f}")

    # Plot in matplotlib
    if visualize == 'matplotlib':
        plt.figure(figsize=(10, 5))

        # Plot training and validation losses
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()

        # Plot training and validation accuracies
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    print("✅ Training complete!")

def test_model(model, test_loader, criterion, device=device):
    """
    This function tests a PyTorch model using the given data loader and loss function.

    Parameters:
    - model: object: A PyTorch model object.
    - test_loader: DataLoader: A DataLoader object containing the testing data.
    - criterion: object: A loss function object.

    Returns:
    Model performance metrics.
    """
    model.eval()
    test_loss, test_corrects, test_total = 0.0, 0, 0

    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc="Testing", leave=False):
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
            test_total += data.size(0)

    test_loss /= len(test_loader.dataset)
    test_accuracy = test_corrects.double() / test_total

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f}")
    print("✅ Testing complete!")

    return test_loss, test_accuracy

def train_BERT(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler=None, visualize=None, save=None):
    """
    This function trains a PyTorch model using the given data loaders and hyperparameters.
    The function also supports visualization using liveloss or matplotlib.

    Parameters:
    - model: object: A PyTorch model object.
    - train_loader: DataLoader: A DataLoader object containing the training data.
    - val_loader: DataLoader: A DataLoader object containing the validation data.
    - criterion: object: A loss function object.
    - optimizer: object: An optimizer object.
    - epochs: int: The number of epochs to train the model.
    - scheduler: object: A learning rate scheduler object.
    - visualize: str: A string that determines the visualization method. Choose from [None, 'liveloss', 'matplotlib'].
    - save_path: str: A string containing the path to save the best model.

    Returns:
    Model performance metrics and visualization.
    """
    best_val_loss = float('inf')
    best_model_wts = None
    
    assert visualize in [None, 'liveloss', 'matplotlib'], "Invalid visualization method. Choose from [None, 'liveloss', 'matplotlib']."

    if visualize == 'liveloss':
        liveloss = PlotLosses()

    if visualize == 'matplotlib':
        train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Training Phase
        logs = {}
        model.train()
        train_loss, train_corrects, train_total = 0.0, 0, 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            data, mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()

            outputs = model(input_ids=data, attention_mask=mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels.data)
            train_total += data.size(0)

        train_loss /= len(train_loader.dataset)
        train_accuracy = train_corrects.double() / train_total

        model.eval()
        val_loss, val_corrects, val_total = 0.0, 0, 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                data, mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

                outputs = model(input_ids=data, attention_mask=mask) 
                loss = criterion(outputs, labels)

                val_loss += loss.item() * data.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_total += data.size(0)

        val_loss /= len(val_loader.dataset)
        val_accuracy = val_corrects.double() / val_total

        if visualize != 'liveloss':
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
            print(f"  Val Loss: {val_loss:.4f} |   Val Acc: {val_accuracy:.4f}")

        if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Save logs for visualization
        if visualize == 'liveloss':
            logs['log loss'] = train_loss
            logs['accuracy'] = train_accuracy.item()
            logs['val_log loss'] = val_loss
            logs['val_accuracy'] = val_accuracy.item()
            liveloss.update(logs)
            liveloss.send()
        elif visualize == 'matplotlib':
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy.cpu())  # Move to CPU
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy.cpu())  # Move to CPU

        # Check if current model is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_model_wts = model.state_dict().copy()
            print(f"👉 New best model with val loss: {val_loss:.4f}!")

        print("-" * 30)

    # Save best model
    if save and best_model_wts:
        torch.save(best_model_wts, save)
        print(f"Best model saved:")
        print(f"Val Loss: {best_val_loss:.4f} | Val Acc: {best_val_accuracy:.4f}")

    # Plot in matplotlib
    if visualize == 'matplotlib':
        plt.figure(figsize=(10, 5))

        # Plot training and validation losses
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()

        # Plot training and validation accuracies
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    print("✅ Training complete!")

def test_BERT(model, test_loader, criterion):
    """
    This function tests a PyTorch model using the given data loader and loss function.

    Parameters:
    - model: object: A PyTorch model object.
    - test_loader: DataLoader: A DataLoader object containing the testing data.
    - criterion: object: A loss function object.

    Returns:
    Model performance metrics.
    """
    model.eval()
    test_loss, test_corrects, test_total = 0.0, 0, 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            data, mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

            outputs = model(data, mask)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
            test_total += data.size(0)

    test_loss /= len(test_loader.dataset)
    test_accuracy = test_corrects.double() / test_total

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f}")
    print("✅ Testing complete!")

    return test_loss, test_accuracy

