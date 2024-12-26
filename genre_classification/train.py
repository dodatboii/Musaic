from torch.utils.data import random_split
from data import MusicDataset
from model import MusicCNN
import torch
import torch.nn as nn
import joblib

def train(model, train_dl, valid_dl, num_epochs, device, patience=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    # Early stopping variables
    best_valid_loss = float('inf')
    counter = 0
    best_model_path = 'genre_classification/src/best_model.pt'

    # To store metrics history
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        for i, data in enumerate(train_dl):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            if i % 10 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')

        # Calculate training metrics
        train_loss = running_loss / len(train_dl)
        train_acc = correct_prediction / total_prediction
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for data in valid_dl:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

        # Calculate validation metrics
        valid_loss = valid_loss / len(valid_dl)
        valid_acc = valid_correct / valid_total
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        # Print epoch statistics
        print(f'Epoch: {epoch + 1}')
        print(f'Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')
        print(f'Validation - Loss: {valid_loss:.4f}, Accuracy: {valid_acc:.4f}')
        print('-' * 60)

        # Early stopping check
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'train_acc': train_acc,
                'valid_acc': valid_acc,
            }, best_model_path)
            print(f"Saved best model to {best_model_path}")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")

        if counter >= patience:
            print("Early stopping triggered")
            break

    print('Finished Training')

    # Save final model
    final_model_path = 'genre_classification/src/final_model.pt'

    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'train_acc': train_acc,
        'valid_acc': valid_acc,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_accs': train_accs,
        'valid_accs': valid_accs,
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")

    return train_losses, valid_losses, train_accs, valid_accs


def main():
    # Define the path to the GTZAN dataset
    gtzan_path = 'genre_classification/dataset/genres_original'  # Modify this with your dataset path

    # Initialize dataset
    myds = MusicDataset(gtzan_path)
    label_encoder_path = 'genre_classification/src/label_encoder.joblib'

    # Save label encoder
    joblib.dump(myds.label_encoder, label_encoder_path)
    print(f"Saved label encoder to {label_encoder_path}")

    # Random split of 80:20 between training and validation
    num_items = len(myds)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

    num_genres = len(myds.label_encoder.classes_)
    myModel = MusicCNN(num_genres)
    myModel.initialize_weights()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    myModel = myModel.to(device)
    train(myModel, train_dl, val_dl, 100, device)

if __name__ == '__main__':
    main()
