import logging

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, Flatten, MaxPooling1D, MaxPooling2D, GlobalAveragePooling2D, \
    BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout
from tensorflow.keras.initializers import GlorotNormal as Xavier
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom, RandomFlip, RandomRotation
from tensorflow.keras.optimizers import Adam
from har_baseline.utility.configurator import Configurator, PrivateConfigurator


# Results are 0.5, 1, 2, 3(removed fog of 3s + epochs maybe better results) seconds
class Models(object):

    def __init__(self, time, sf) -> None:
        _time = time
        _sampling_frequency = sf
        _window_length = int(_time * _sampling_frequency)

        _img_shape = (_window_length,
                      PrivateConfigurator().getint('cnn', 'input_shape_y'),
                      3)

        _acc_shape = (PrivateConfigurator().getint('cnn', 'input_shape_x'),
                      PrivateConfigurator().getint('cnn', 'input_shape_y'))

        _output_size = PrivateConfigurator().getint('cnn', 'output')

        _data_augmentation = Sequential([
            RandomZoom(0.1),
            RandomFlip('horizontal', seed=28, input_shape=_img_shape),
            RandomRotation(0.1)
        ])

        _model_1_for_acc = Sequential([
            Conv1D(filters=64,
                   kernel_size=PrivateConfigurator().getint('m1_acc', 'conv-kernel'),
                   activation='relu', input_shape=_acc_shape),

            MaxPooling1D(pool_size=PrivateConfigurator().getint('m1_acc', 'pool-size')),

            # Features Vector
            Flatten(),

            Dense(128, activation="relu", kernel_initializer=Xavier(seed=28)),  # 1024
            Dropout(0.5),

            Dense(128, activation="relu", kernel_initializer=Xavier(seed=28)),  # 1024
            Dropout(0.5),

            # Add an output layer
            Dense(_output_size, activation="softmax")
        ])

        self._models = {
                        'm1_acc': _model_1_for_acc
                        }

    def get_model(self, model_key: str):
        m = self._models[model_key]
        m.compile(
            optimizer=Adam(lr=1e-5),  # TODO: try 1e-4, default is 1e-3
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Get model information
        m.summary(print_fn=logging.info)

        return m


class MlModels(object):
    clf_model_params = {
        'SVC': {
            'model': SVC(),
            'param_grid': {
                'C': [0.01, 0.1, 1, 10],
                'kernel': ['linear', 'poly', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        },
        'WK-NN': {
            'model': KNeighborsClassifier(weights='distance', n_jobs=-1),
            'param_grid': {
                'n_neighbors': list(range(1, 21)),
                'p': [1, 2, 3, 4, 5],
                'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
            }
        },
        'K-NN': {
            'model': KNeighborsClassifier(n_jobs=-1),
            'param_grid': {
                'n_neighbors': list(range(1, 5)),
                'p': [1, 3, 5],
                'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
            }
        },
        'LDA': {
            'model': LinearDiscriminantAnalysis(),
            'param_grid': {
                'solver': ['svd', 'lsqr', 'eigen'],
                'store_covariance': ['True', 'False'],
                'tol': [0.0001, 0.001, 0.01, 0.1, 1.0]
            }
        },
        'QDA': {
            'model': QuadraticDiscriminantAnalysis(),
            'param_grid': {
                'store_covariance': ['True', 'False'],
                'tol': [0.0001, 0.001, 0.01, 0.1, 1.0]
            }
        },
        'RF': {
            'model': RandomForestClassifier(n_jobs=-1),
            'param_grid': {
                'criterion': ['gini', 'entropy'],
                'max_features': ['auto', 'sqrt', 'log2'],
                'min_samples_split': [2, 3, 4, 5]
            }
        },
        'DT': {
            'model': DecisionTreeClassifier(),
            'param_grid': {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'min_samples_split': [2, 3, 4, 5],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        }
    }
