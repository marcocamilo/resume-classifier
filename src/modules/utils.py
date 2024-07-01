import numpy as np
import json
import os

def top_resumes(index: str, matrix: np.ndarray, postings, resumes, n: int = 5, save: bool = False):
    # write doc string
    """
    Function to get top n resumes for a given posting index

    Parameters:
            index (int): index of the posting
            matrix (np.ndarray): similarity matrix
            postings (list): list of postings
            resumes (list | np.ndarray): list of resumes
            n (int): number of top resumes
            save (bool): save or print

    Returns:
    Indexed document and top n matches
    """
    top_n = np.argsort(matrix[index])[::-1][:n]
    top_resumes = resumes[top_n]
    scores = matrix[index][top_n]
    if save == False:
        print(f"Posting: {postings[index]}\n")
        print(f"Top {n} Resumes:\n")
        for i, resume, score in zip(range(n), top_resumes, scores):
            print(f"{i+1}. {np.round(score,2)}\n{resume}\n")
    else:
        return list(zip(scores, top_resumes))

def save_performance(model_name, architecture, embed_size, learning_rate, epochs, optimizer, criterion, accuracy, path='./output/classifier_performance.json'):
    # Load existing data
    data = []
    if os.path.exists(path):
        with open(path, 'r') as file:
            data = json.load(file)

    # Create or update the model entry
    model_data = {
        'model': model_name,
        'architecture': architecture,
        'embed_size': embed_size,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'optimizer': optimizer,
        'criterion': criterion,
        'accuracy': accuracy
    }
    
    # Check if the model already exists and update it
    for i, model in enumerate(data):
        if model['model'] == model_name:
            data[i] = model_data
            break
    else:
        data.append(model_data)

    # Save the updated data
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)
