import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
import joblib
import os
import joblib
from sklearn.preprocessing import LabelEncoder
matplotlib.use("Agg")  # Use non-interactive backend to avoid threading issues


random_state = 343


def plot_confusion_matrix(y_test, y_test_fit, CV_suffix = "", class_names=[]):
    save_path="./plots_use"
    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    cm = confusion_matrix(y_test, y_test_fit, labels=class_names, normalize='true')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation='vertical',cmap=plt.cm.Blues, values_format=".2f")
    for text in disp.ax_.texts:
        text.set_fontsize(8)
    plt.tight_layout()
    #plt.show()

    save_file = os.path.join(save_path, f"{CV_suffix}_confusion_matrix.png")
    plt.savefig(save_file, dpi=300)
    plt.close()

" -- This file is from Roya, I have just change from regression to classifyer --"

def run_NN(X_train, y_train, X_test, y_test, class_names=[], CV_suffix = "", param_grid=None, n_jobs=-1, opt = None, time_window = None):
     
    save_base_path="./models"
    # Create a directory for the model

    from sklearn.utils import shuffle
    #Shuffle training samples before anything else
    X_train, y_train = shuffle(X_train, y_train, random_state=random_state)

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    assert not np.isnan(X_train).any(), "NaN in X_train"
    assert not np.isinf(X_train).any(), "Inf in X_train"

    if opt==True:
        print('Optimizing NN hyperparameters...')
        #  """Neural Network Classification with optional hyperparameter optimization"""
        # Default parameter grid for MLPClassifier if none is provided
        if param_grid is None:
            param_grid = {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (150, 100)],
                    'activation': ['relu'],  # Only ReLU for stable gradients
                    'solver': ['adam'],  # Adam is adaptive & robust
                    'alpha': [0.01, 0.1, 0.5],  # Stronger regularization
                    'learning_rate': ['adaptive'],  # Avoids aggressive weight updates
                    'max_iter': [1000],  # Early stopping will take care of overtraining
                    'batch_size': [16, 32],  # Regularization via smaller batch size
                    'early_stopping': [True]  # Enable early stopping

                    # 'hidden_layer_sizes': [
                    #             (50,),         # One hidden layer with 50 units
                    #             (100,),        # One hidden layer with 100 units
                    #             (50, 50),      # Two hidden layers
                    #             (100, 50),     # Two hidden layers
                    #             (150, 100),    # Two hidden layers
                    #             (50, 50, 50),  # Three hidden layers
                    #             (100, 50, 50), # Three hidden layers
                    #             (150, 100, 50) # Three hidden layers
                    #         ],              
                    # 'activation': ['relu', 'tanh'],  # Activation functions
                    # 'solver': ['adam', 'sgd'],       # Optimization algorithms
                    # 'alpha': [0.0001, 0.001, 0.01],  # L2 regularization term (penalty)
                    # 'learning_rate': ['constant', 'adaptive'],  # Learning rate schedule
                    # 'max_iter': [1000,2000,3000]     # Maximum number of iterations
                }

        # Create the MLPRegressor model
        model = MLPClassifier(random_state=random_state)
                
        # Use GridSearchCV to find the best hyperparameters
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                                        cv=3, n_jobs=n_jobs)
                
        # Fit the model using the training data
        grid_search.fit(X_train, y_train_encoded)

        # Get the best parameters and model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        # Print the best parameters
        # print(f"Best hyperparameters: {best_params}")

        #  View cross-validation fold results
        cv_results = grid_search.cv_results_
        # print("Fold-wise test results:")
        for i in range(3):  # Assuming 3-fold cross-validation
            fold_score = cv_results[f'split{i}_test_score'][grid_search.best_index_]
            print(f"Fold {i+1} test score: {fold_score}")

        # Evaluate the model on test data
        Overall = best_model.score(X_test, y_test_encoded)
        # print(f"Test R^2 score: {Overall}")

        model = best_model
        
        # Fit the model using the training data
        # model.fit(X_train, y_train)
        # Save the model to a file
        model_filename = os.path.join(save_base_path, f"NN_{CV_suffix}_{time_window}.joblib")
        joblib.dump(model, model_filename)
        
    
    y_test_fit      = model.predict(X_test)
    accuracy_test   = model.score(X_test, y_test_encoded)
    f1_test         = f1_score(y_test_encoded, y_test_fit, average='macro')
    precision_test  = precision_score(y_test_encoded, y_test_fit, average='macro')
    recall_test     = recall_score(y_test_encoded, y_test_fit, average='macro')

    y_train_fit     = model.predict(X_train)
    accuracy_train  = model.score(X_train, y_train_encoded)
    f1_train         = f1_score(y_train_encoded, y_train_fit, average='macro')
    precision_train  = precision_score(y_train_encoded, y_train_fit, average='macro')
    recall_train     = recall_score(y_train_encoded, y_train_fit, average='macro')
    
    y_test_fit_labels = le.inverse_transform(y_test_fit)
    y_test_labels = le.inverse_transform(y_test_encoded)

    y_train_fit_labels = le.inverse_transform(y_train_fit)
    y_train_labels = le.inverse_transform(y_train_encoded)
        
    test_results    = (y_test_fit_labels, accuracy_test, f1_test, precision_test, recall_test) #, mse, rmse)
    train_results   = (y_train_fit_labels, accuracy_train, f1_train, precision_train, recall_train) #, mse_train, rmse_train)

    # Call the plot function with the necessary parameters
    Title = f"NN_{CV_suffix}_{time_window}"
    #plot_confusion_matrix(y_test_labels, y_test_fit_labels, CV_suffix=Title, class_names=class_names)

    return test_results, train_results
