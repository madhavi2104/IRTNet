import torch

def generate_model_predictions(model, dataset):
    model.eval()
    predictions = []
    for image in dataset:
        with torch.no_grad():
            predictions.append(model(image.unsqueeze(0)))
    return predictions

def save_model_responses(predictions, path):
    torch.save(predictions, path)
