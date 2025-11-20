from src_SimpleNeuralNetwork.Function.Main_model import model
import numpy as np

def save(
    model,
    path
):
    if path[-4:] != ".npz":
        path += ".npz"
    np.savez(path, **model.parametres)

def load(
    path
):
    if path[-4:] != ".npz":
        path += ".npz"
    data = np.load(path)
    parametres = {key: data[key] for key in data.files}

    return model(
        NeuralNetwork_structure = None,
        Patametres_exist = parametres
    )