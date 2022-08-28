from model import ddGPredictor
from itertools import product



def grid_search(train_loader, val_loader,
                grid_search_spaces = {
                    
                    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
                    "": [1e-4, 1e-5, 1e-6]
                },
                model_class=ddGPredictor, epochs=20, patience=5):
    """
    A simple grid search based on nested loops to tune learning rate and
    regularization strengths.
    Keep in mind that you should not use grid search for higher-dimensional
    parameter tuning, as the search space explodes quickly.

    Required arguments:
        - train_dataloader: A generator object returning training data
        - val_dataloader: A generator object returning validation data

    Optional arguments:
        - grid_search_spaces: a dictionary where every key corresponds to a
        to-tune-hyperparameter and every value contains a list of possible
        values. Our function will test all value combinations which can take
        quite a long time. If we don't specify a value here, we will use the
        default values of both our chosen model as well as our solver
        - model: our selected model for this exercise
        - epochs: number of epochs we are training each model
        - patience: if we should stop early in our solver

    Returns:
        - The best performing model
        - A list of all configurations and results
    """
    configs = []

    """
    # Simple implementation with nested loops
    for lr in grid_search_spaces["learning_rate"]:
        for reg in grid_search_spaces["reg"]:
            configs.append({"learning_rate": lr, "reg": reg})
    """

    # More general implementation using itertools
    for instance in product(*grid_search_spaces.values()):
        configs.append(dict(zip(grid_search_spaces.keys(), instance)))

    return findBestConfig(train_loader, val_loader, configs, epochs, patience,
                          model_class)


def findBestConfig(train_loader, val_loader, configs, EPOCHS, PATIENCE,
                   model_class):
    """
    Get a list of hyperparameter configs for random search or grid search,
    trains a model on all configs and returns the one performing best
    on validation set
    """
    
    best_val = None
    best_config = None
    best_model = None
    results = []
    
    for i in range(len(configs)):
        print("\nEvaluating Config #{} [of {}]:\n".format(
            (i+1), len(configs)),configs[i])

        model = model_class(**configs[i])
        solver = Solver(model, train_loader, val_loader, **configs[i])
        solver.train(epochs=EPOCHS, patience=PATIENCE)
        results.append(solver.best_model_stats)

        if not best_val or solver.best_model_stats["val_loss"] < best_val:
            best_val, best_model,\
            best_config = solver.best_model_stats["val_loss"], model, configs[i]
            
    print("\nSearch done. Best Val Loss = {}".format(best_val))
    print("Best Config:", best_config)
    return best_model, list(zip(configs, results))