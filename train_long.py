from ai import train_model

if __name__ == '__main__':
    # Retrain with more epochs to improve model confidence
    train_model(num_epochs=200, model_path='model.pth', print_progress=True)
