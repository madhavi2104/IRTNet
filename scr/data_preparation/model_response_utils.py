import numpy as np

def format_responses_for_irt(predictions):
    # Convert predictions to IRT-compatible format
    return np.array(predictions)

def compare_responses_across_models(response_set):
    # Compare responses from multiple models
    return np.mean(response_set, axis=0)
