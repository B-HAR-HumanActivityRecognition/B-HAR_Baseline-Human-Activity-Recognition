import os
import random
from itertools import cycle
import tsfel as ts
import tensorflow as tf
import numpy as np
from sklearn import metrics
import sensormotion as sm
import matplotlib.pyplot as plt
import math
from shutil import copy2
import seaborn as sns
import pandas as pd
from imblearn.metrics import sensitivity_specificity_support
from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss, EditedNearestNeighbours
from scipy import stats
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFE
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from pathlib import Path
from datetime import datetime
from sklearn.svm import LinearSVC, SVR
from b_har.utility.configurator import Configurator, PrivateConfigurator
from b_har.models import Models, MlModels
import logging
import progressbar
from timeit import default_timer as timer


class B_HAR:
    _data_delimiters = {
        # if the data are in the last columns, we need to know only the start index
        'dc': (0, -1),
        'cd': (1),
        'tdc': (1, -1),
        'tcd': (2),
        'cdt': (1, -1),
        'dct': (0, -2),
        'ctd': (2),
        'dtc': (0, -2),
        'tpcd': (3),
        'tpdc': (2, -1),
        'tcpd': (3),
        'tcdp': (2, -1),
        'tdcp': (1, -2),
        'tdpc': (1, -2),
        'ptcd': (3),
        'ptdc': (2, -1),
        'pdtc': (1, -2),
        'pdct': (1, -2),
        'pctd': (3),
        'pcdt': (2, -1),
        'ctdp': (2, -1),
        'ctpd': (3),
        'cdtp': (1, -2),
        'cdpt': (1, -2),
        'cptd': (3),
        'cpdt': (2, -1),
        'dtcp': (0, -3),
        'dtpc': (0, -3),
        'dctp': (0, -3),
        'dcpt': (0, -3),
        'dptc': (0, -3),
        'dpct': (0, -3)
    }

    # --- Public Methods ---

    def __init__(self, config_file_path) -> None:
        super().__init__()
        self.__cfg_path = config_file_path

    def stats(self):
        """
        Prints out the statistics of a given dataset
        :return: None
        """
        self._init_log()
        df = self._decode_csv(ds_dir=Configurator(self.__cfg_path).get('dataset', 'path'),
                              separator=Configurator(self.__cfg_path).get('dataset', 'separator', fallback=' '),
                              header_type=Configurator(self.__cfg_path).get('dataset', 'header_type'),
                              has_header=Configurator(self.__cfg_path).getboolean('dataset', 'has_header')
                              )

        header_type = Configurator(self.__cfg_path).get('dataset', 'header_type')
        group_by = Configurator(self.__cfg_path).get('settings', 'group_by')

        # Create stats directory
        stats_dir = os.path.join(Configurator(self.__cfg_path).get('settings', 'log_dir'), 'stats')
        Path(stats_dir).mkdir(parents=True, exist_ok=True)

        # Plot settings
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        values = df[group_by].value_counts()

        # Bar Plot Class
        plt.title('Class Distribution')
        df[group_by].value_counts().plot(kind='bar', color=colors)

        i = 0
        offset = 1000
        for value in values:
            plt.text(i, value + offset, str(np.round(value / np.sum(df[group_by].value_counts()) * 100)) + '%')
            i += 1

        plt.xlabel('Classes')
        plt.ylabel('Instances')
        plt.savefig(os.path.join(Configurator(self.__cfg_path).get('settings', 'log_dir'), 'stats/class_barplot.png'))
        plt.show()
        plt.close()

        # Bar Plot Patients
        if 'p' in header_type:
            plt.title('Data per Patient')
            df['P_ID'].value_counts().plot(kind='bar', color=colors)

            plt.xlabel('Patients')
            plt.ylabel('Instances')
            plt.savefig(
                os.path.join(Configurator(self.__cfg_path).get('settings', 'log_dir'), 'stats/patients_barplot.png'))
            plt.show()
            plt.close()

        # Boxplot Data
        df.boxplot(
            column=list(df.columns[self._data_delimiters[header_type][0]: self._data_delimiters[header_type][1]]))
        plt.savefig(os.path.join(Configurator(self.__cfg_path).get('settings', 'log_dir'), 'stats/data_boxplot.png'))
        plt.show()
        plt.close()

        # Boxplot Class
        df.boxplot(
            column=list(df.columns[self._data_delimiters[header_type][0]: self._data_delimiters[header_type][1]]),
            by='CLASS')
        plt.savefig(os.path.join(Configurator(self.__cfg_path).get('settings', 'log_dir'), 'stats/class_boxplot.png'))
        plt.show()
        plt.close()

    def get_baseline(self, ml_models, dl_models, discard_class: list = None, discard_patients: list = None,
                     ids_test: list = None):
        """
        Evaluates the input dataset with different machine learning and deep learning models in order to get a baseline for
        future analysis and comparisons.

        :param ids_test: list of patient id used as testing set
        :param ml_models: list of machine learning models
        :param dl_models: list of cnn models
        :param discard_class: list of class useless for analysis
        :param discard_patients: list of patient id useless for analysis
        :return:
        """
        self._init_log()
        # Evaluate number of sample per time window
        time_window_size = int(Configurator(self.__cfg_path).getint('settings', 'sampling_frequency') *
                               Configurator(self.__cfg_path).getfloat('settings', 'time'))

        # --- Load Data ---
        try:
            dataset = self._decode_csv(ds_dir=Configurator(self.__cfg_path).get('dataset', 'path'),
                                       separator=Configurator(self.__cfg_path).get('dataset', 'separator', fallback=' '),
                                       header_type=Configurator(self.__cfg_path).get('dataset', 'header_type'),
                                       has_header=Configurator(self.__cfg_path).getboolean('dataset', 'has_header')
                                       )
        except Exception as e:
            print('Failed to load data.')
            print(e.args)
            exit(10)
        # -----------------

        # --- Data Cleaning  ---
        try:
            dataset = self._clean_data(df=dataset,
                                       sampling_frequency=Configurator(self.__cfg_path).getint('settings', 'sampling_frequency'),
                                       high_cutoff=Configurator(self.__cfg_path).getint('cleaning', 'high_cut', fallback=None),
                                       low_cutoff=Configurator(self.__cfg_path).getint('cleaning', 'low_cut', fallback=None),
                                       sub_method=Configurator(self.__cfg_path).get('cleaning', 'sub_method'),
                                       header_type=Configurator(self.__cfg_path).get('dataset', 'header_type')
                                       )
        except Exception as e:
            print('Failed to clean data.')
            print(e.args)
            exit(20)
        # -----------------

        # --- Data Treatment ---
        data_treatment_type = Configurator(self.__cfg_path).get('settings', 'data_treatment')
        if data_treatment_type == 'features_extraction':
            try:
                dt_dataset = self._features_extraction(df=dataset,
                                                       sampling_frequency=Configurator(self.__cfg_path).getint('settings', 'sampling_frequency'),
                                                       time_window=time_window_size,
                                                       overlap=Configurator(self.__cfg_path).getfloat('settings', 'overlap'),
                                                       header_type=Configurator(self.__cfg_path).get('dataset', 'header_type'))
            except Exception as e:
                print('Failed to extract features.')
                print(e.args)
                exit(30)

        elif data_treatment_type == 'segmentation':
            try:
                dt_dataset = self._apply_segmentation(df=dataset,
                                                      sampling_frequency=Configurator(self.__cfg_path).getint('settings', 'sampling_frequency'),
                                                      time_window_size=time_window_size,
                                                      overlap=Configurator(self.__cfg_path).getfloat('settings', 'overlap'))
            except Exception as e:
                print('Failed during data segmentation.')
                print(e.args)
                exit(40)

        elif data_treatment_type == 'raw':
            dt_dataset = dataset

        else:
            print('*** Fallback: not recognised %s, using Raw instead ***' % data_treatment_type)
            logging.info('*** Fallback: not recognised %s, using Raw instead ***' % data_treatment_type)
            dt_dataset = dataset

        # ----------------------

        del dataset

        # --- Preprocessing ---
        try:
            X_train_set, X_validation_set, Y_train_set, Y_validation_set, class_labels = self._data_preprocessing(
                df=dt_dataset,
                drop_class=discard_class,
                drop_patient=discard_patients,
                split_method=Configurator(self.__cfg_path).get('preprocessing', 'split_method'),
                normalisation_method=Configurator(self.__cfg_path).get('preprocessing', 'normalisation_method'),
                selection_method=Configurator(self.__cfg_path).get('preprocessing', 'selection_method'),
                balancing_method=Configurator(self.__cfg_path).get('preprocessing', 'balancing_method'),
                ids_test_set=ids_test
            )
        except Exception as e:
            print('Failed during data preprocessing.')
            print(e.args)
            exit(50)
        # ---------------------

        del dt_dataset

        # --- Start ML Evaluation ---
        if Configurator(self.__cfg_path).getboolean('settings', 'use_ml'):
            self._start_ml_evaluation(X_train_set, Y_train_set, X_validation_set, Y_validation_set, ml_models)
        # ---------------------------

        # --- Start DL Evaluation ---
        if Configurator(self.__cfg_path).getboolean('settings', 'use_dl'):
            self._dl_evaluation(X_train_set, Y_train_set, X_validation_set, Y_validation_set, dl_models, time_window_size)
        # ---------------------------

        # --- Shut down logging ---
        logging.shutdown()
        # ---------------------------

    # --- Private Methods ---

    def _get_cfg_path(self):
        return self.__cfg_path

    def _apply_segmentation(self, df, sampling_frequency, time_window_size, overlap):
        has_patient = 'p' in Configurator(self.__cfg_path).get('dataset', 'header_type')
        x_df = []
        widgets = [
            'Segmentation',
            ' [', progressbar.Timer(), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
        ]

        with progressbar.ProgressBar(widgets=widgets, max_value=len(df)) as bar:
            for d in np.arange(0,len(df)):
                if has_patient:
                    x, y, p = self._get_window(df[d], sampling_frequency, time_window_size, overlap)
                else:
                    x, y = self._get_window(df[d], sampling_frequency, time_window_size, overlap)
                    p = None

                x_df.append(pd.DataFrame(x.reshape(-1, x.shape[1] * x.shape[2])))
                x_df[d]['CLASS'] = y

                if has_patient and p is not None:
                    x_df[d]['P_ID'] = p
                bar.update()

        return x_df

    def _dl_evaluation(self, x_train, y_train, x_val, y_val, dl_models, time_window_size):
        # --- Use Also Features ---
        if Configurator(self.__cfg_path).getboolean('training', 'use_features') and \
                Configurator(self.__cfg_path).get('settings', 'data_treatment') == 'features_extraction':
            logging.info('--- Features as input of CNN ---')
            # Set CNN input size
            n_features = x_train.shape[1]
            PrivateConfigurator().set('cnn', 'input_shape_x', '1')
            PrivateConfigurator().set('cnn', 'input_shape_y', str(n_features))

            # Set CNN kernel sizes
            PrivateConfigurator().set('m1_acc', 'conv-kernel', '1')
            PrivateConfigurator().set('m1_acc', 'pool-size', '1')

            # Reshape to feed CNN
            X_training_features = x_train.reshape((-1, 1, n_features))
            X_testing_features = x_val.reshape((-1, 1, n_features))

            y_training_features = np.asarray(y_train).reshape(-1)
            y_testing_features = np.asarray(y_val).reshape(-1)

            # Evaluation
            self._start_cnn_evaluation(X_training_features,
                                       y_training_features,
                                       X_testing_features,
                                       y_testing_features,
                                       dl_models,
                                       Configurator(self.__cfg_path).getint('training', 'epochs'),
                                       Configurator(self.__cfg_path).getint('training', 'k_fold'),
                                       Configurator(self.__cfg_path).getfloat('training', 'loss_threshold'),
                                       np.unique(y_train),
                                       'Features-as-input')
            # ------------------------
            # --- Use accelerometer data as input to the CNN ---

        logging.info('--- Accelerometer data as input of CNN ---')

        if Configurator(self.__cfg_path).get('settings', 'data_treatment') == 'segmentation':
            # Get windowed data
            n_features = int(x_train.shape[1] / time_window_size)
            x_train = x_train.reshape((-1, time_window_size, n_features))
            x_val = x_val.reshape((-1, time_window_size, n_features))

            # Set CNN input size
            PrivateConfigurator().set('cnn', 'input_shape_x', str(x_train.shape[1]))
            PrivateConfigurator().set('cnn', 'input_shape_y', str(x_train.shape[2]))

            # Set CNN kernel sizes
            if n_features <= 3:
                PrivateConfigurator().set('m1_acc', 'conv-kernel', '1')
                PrivateConfigurator().set('m1_acc', 'pool-size', '1')
            else:
                PrivateConfigurator().set('m1_acc', 'conv-kernel', '3')
                PrivateConfigurator().set('m1_acc', 'pool-size', '2')

            # Evaluation
            logging.info('--> Training set shape: %s' % str(x_train.shape))
            logging.info('--> Validation set shape: %s' % str(x_val.shape))
            self._start_cnn_evaluation(x_train,
                                       y_train,
                                       x_val,
                                       y_val,
                                       dl_models,
                                       Configurator(self.__cfg_path).getint('training', 'epochs'),
                                       Configurator(self.__cfg_path).getint('training', 'k_fold'),
                                       Configurator(self.__cfg_path).getfloat('training', 'loss_threshold'),
                                       np.unique(y_train),
                                       'Accelerometer-data')
            # ---------------------------

        if Configurator(self.__cfg_path).get('settings', 'data_treatment') == 'raw':
            # Set CNN input size
            n_features = x_train.shape[1]
            PrivateConfigurator().set('cnn', 'input_shape_x', '1')
            PrivateConfigurator().set('cnn', 'input_shape_y', str(n_features))

            # Set CNN kernel sizes
            PrivateConfigurator().set('m1_acc', 'conv-kernel', '1')
            PrivateConfigurator().set('m1_acc', 'pool-size', '1')

            # Reshape to feed CNN
            X_training_features = x_train.reshape((-1, 1, n_features))
            X_testing_features = x_val.reshape((-1, 1, n_features))

            y_training_features = np.asarray(y_train).reshape(-1)
            y_testing_features = np.asarray(y_val).reshape(-1)

            logging.info('--> Training set shape: %s' % str(x_train.shape))
            logging.info('--> Validation set shape: %s' % str(x_val.shape))
            # Evaluation
            self._start_cnn_evaluation(X_training_features,
                                       y_training_features,
                                       X_testing_features,
                                       y_testing_features,
                                       dl_models,
                                       Configurator(self.__cfg_path).getint('training', 'epochs'),
                                       Configurator(self.__cfg_path).getint('training', 'k_fold'),
                                       Configurator(self.__cfg_path).getfloat('training', 'loss_threshold'),
                                       np.unique(y_train),
                                       'Raw')

    def _apply_filter(self, df, filter_name, sample_rate, frequency_cutoff, order):
        b, a = sm.signal.build_filter(frequency=frequency_cutoff,
                                      sample_rate=sample_rate,
                                      filter_type=filter_name,
                                      filter_order=order)

        header_type = Configurator(self.__cfg_path).get('dataset', 'header_type')
        for column in list(df.columns)[self._data_delimiters[header_type][0]: self._data_delimiters[header_type][1]]:
            df[column] = sm.signal.filter_signal(b, a, signal=df[column].values)

        return df

    def _init_log(self):
        now = datetime.now()
        timestamp = datetime.fromtimestamp(datetime.timestamp(now))
        log_dir_name = '%s - log' % str(timestamp).split('.')[0]
        new_log_dir_path = os.path.join(Configurator(self.__cfg_path).get('settings', 'log_dir'), log_dir_name)

        try:
            # Create run log directory
            Path(new_log_dir_path).mkdir(parents=True, exist_ok=True)

            # Copy run settings
            copy2(self._get_cfg_path(), new_log_dir_path)

            # Update log dir path for future usage
            Configurator(self._get_cfg_path()).set('settings', 'log_dir', new_log_dir_path)

            # Configure logging
            logging.basicConfig(filename=os.path.join(new_log_dir_path, 'log.rtf'), format='', level=logging.INFO)
        except Exception as e:
            print(e.args)
            logging.info(e.args)
            exit(1)
        else:
            logging.log(msg="Successfully created the directory %s " % new_log_dir_path, level=logging.INFO)

    def _print_val_train_stats(self, history, title, save_to_dl_dir: False):
        # Create training stats directory
        if save_to_dl_dir:
            dl_dir = os.path.join(Configurator(self.__cfg_path).get('settings', 'log_dir'), 'deep_learning')
            training_stats_dir = os.path.join(dl_dir, 'training')
            Path(training_stats_dir).mkdir(parents=True, exist_ok=True)
            save_to = training_stats_dir
        else:
            Path(os.path.join(Configurator(self.__cfg_path).get('settings', 'log_dir'), 'training')).mkdir(parents=True,
                                                                                                           exist_ok=True)
            save_to = os.path.join(Configurator(self.__cfg_path).get('settings', 'log_dir'), 'training')

        fig, axis = plt.subplots(2)
        fig.suptitle('%s' % title)

        axis[0].set_title('Accuracy')
        axis[0].plot(history['accuracy'], label='acc', c='g')
        axis[0].plot(history['val_accuracy'], label='val-acc', c='b')
        axis[0].legend()

        axis[1].set_title('Loss')
        axis[1].plot(history['loss'], label='loss', c='r')
        axis[1].plot(history['val_loss'], label='val-loss', c='m')
        axis[1].legend()

        plt.savefig(os.path.join(save_to, 'training_stats_%s.png' % title))

        plt.show()
        plt.close()

    def _do_k_fold(self, x, y, kf, model, model_name, epochs, x_unseen=None, y_unseen=None):
        # Define supporting variables
        fold = 0
        trained_models = dict()
        initial_weights = model.get_weights()
        accuracy_per_fold = list()
        loss_per_fold = list()

        # Do K-Fold
        # Each fold is used once as a validation while the k - 1 remaining folds form the training set.
        for train, test in kf.split(x):
            fold += 1

            # Define training set
            x_train = x[train]
            y_train = tf.keras.utils.to_categorical(y[train])

            # Define testing set
            x_test = x[test]
            y_test = tf.keras.utils.to_categorical(y[test])

            logging.info('------------------------------------------------------------------------')
            logging.info('Training for fold %s ...' % str(fold))

            # Training and Validation
            history = model.fit(x_train, y_train,
                                validation_data=(x_test, y_test),
                                verbose=0,
                                epochs=epochs)

            # Check if the gap between train and validation is constant
            self._print_val_train_stats(history.history, '%s fold %s' % (model_name, fold),
                                        Configurator(self.__cfg_path).getboolean('settings', 'use_dl'))

            # Predict values of validation
            pred = model.predict(x_test)

            # Evaluate model score (Loss and Accuracy)
            scores = model.evaluate(x_test, y_test, verbose=0)

            logging.info('\nStats for fold %s: %s of %s; %s of %s'
                         % (str(fold), str(model.metrics_names[0]), str(scores[0]), str(model.metrics_names[1]),
                            str(scores[1] * 100))
                         )

            accuracy_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])

            # Test on unseen data
            if x_unseen is not None and y_unseen is not None:
                y_unseen_predicted = model.predict(x_unseen)
                trained_models.update({fold: (model.get_weights(), y_test, pred, y_unseen_predicted, scores[0])})
            else:
                y_unseen_predicted = None

            # Measure this fold's RMSE
            mse = np.sqrt(metrics.mean_squared_error(y_test.argmax(1), pred.argmax(1)))  # lower is better

            # Save trained model with its predictions and ground truth
            trained_models.update(
                {fold: (model.get_weights(), y_test, pred, None, scores[0])})  # None instead of y_unseen_pred

            logging.info(
                'Classification Report on Test Fold\n%s' % str(classification_report(y_test.argmax(1), pred.argmax(1))))

            if x_unseen is not None and y_unseen is not None:
                logging.info('Classification Report on Unseen Data\n%s' %
                             str(classification_report(y_unseen, y_unseen_predicted.argmax(1))))
                logging.info("\nMean Squared Error on Unseen Data: %s" %
                             str(np.sqrt(metrics.mean_squared_error(y_unseen, y_unseen_predicted.argmax(1)))))

            logging.info("\nFold %s score: %s\n" % (str(fold), str(mse)))
            logging.info('------------------------------------------------------------------------')

            # Reset model weights for the next train session
            model.set_weights(initial_weights)

        # Print a final summary
        logging.info('------------------------------------------------------------------------')
        logging.info('Score per fold')
        for i in range(0, len(accuracy_per_fold)):
            logging.info('------------------------------------------------------------------------')
            logging.info('> Fold %s - Loss: %s - Accuracy: %s' % (str(i + 1), loss_per_fold[i], accuracy_per_fold[i]))
        logging.info('------------------------------------------------------------------------')
        logging.info('Average scores for all folds:')
        logging.info('> Accuracy: %s (+- %s)' % (str(np.mean(accuracy_per_fold)), str(np.std(accuracy_per_fold))))
        logging.info('> Loss: %s' % (str(np.mean(loss_per_fold))))
        logging.info('------------------------------------------------------------------------')

        return np.mean(accuracy_per_fold), trained_models  # Accuracy, models

    def _model_ensembles(self, trained_models, model, x_test_unseen, y_test, model_name, class_values, threshold=.40):
        predictions = list()
        for key in trained_models:
            # Use only models with a mean squared error under the threshold for prediction
            if trained_models[key][4] <= threshold:
                model.set_weights(trained_models[key][0])
                y_pred = model.predict(x_test_unseen).argmax(1)
                predictions.append(y_pred)

        if not predictions:
            logging.info('There are no predictions, models were too bad.')
        else:
            n_used_models = len(predictions)
            predictions = stats.mode(np.asarray(predictions))[0][0]

            # Metrics
            metrics1 = sensitivity_specificity_support(y_test, predictions, average='weighted')
            metrics2 = precision_recall_fscore_support(y_test, predictions, average='weighted')
            metrics3 = accuracy_score(y_test, predictions, normalize=True)

            logging.log(msg='--> clf: %s\n'
                            '   • specificity: %s\n'
                            '   • sensitivity: %s\n'
                            '   • precision: %s\n'
                            '   • accuracy: %s\n'
                            '   • recall: %s\n'
                            '   • f1-score: %s\n' % (
                                model_name, metrics1[1], metrics1[0], metrics2[0], metrics3, metrics2[1], metrics2[0]),
                        level=logging.INFO)

            self._cm_analysis(y_test, predictions, class_values, model_name,
                              '%s Model Ensembles (used %s)' % (model_name, str(n_used_models)), 'deep_learning')

    def _cm_analysis(self, y_true, y_pred, labels, model_name, title, path, ymap=None):
        """
        Generate matrix plot of confusion matrix with pretty annotations.
        The plot image is saved to disk.
        args:
          y_true:    true label of the data, with shape (n_samples,)
          y_pred:    prediction of the data, with shape (n_samples,)
          filename:  filename of figure file to save
          labels:    string array, name the order of class labels in the confusion matrix.
                     use `clf.classes_` if using scikit-learn models.
                     with shape (nclass,).
          ymap:      dict: any -> string, length == nclass.
                     if not None, map the labels & ys to more understandable strings.
                     Caution: original y_true, y_pred and labels must align.
          figsize:   the size of the figure plotted.
        """
        # Calc figure size based on number of class, for better visualisation
        tail_size = 2
        n_class = len(np.unique(y_true))
        figsize = (tail_size * n_class, int(tail_size * n_class * 1.2))

        if ymap is not None:
            y_pred = [ymap[yi] for yi in y_pred]
            y_true = [ymap[yi] for yi in y_true]
            labels = [ymap[yi] for yi in labels]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)
        cm = pd.DataFrame(cm, index=labels, columns=labels)
        cm.index.name = 'Actual'
        cm.columns.name = 'Predicted'
        fig, ax = plt.subplots(figsize=figsize)
        plt.title('Confusion Matrix %s' % title)
        sns.heatmap(cm, annot=annot, fmt='', ax=ax)
        plt.savefig(
            os.path.join(Configurator(self.__cfg_path).get('settings', 'log_dir'),
                         '%s/CM %s  %s.png' % (path, model_name, title)))
        plt.show()
        plt.close(fig)
        plt.close()

    def _select_patients_training_test(self, ds, p_ids: list = None):
        patient_ids = ds['P_ID'].unique()
        total_patients = len(patient_ids)
        n_patient_validation = int(
            np.floor(total_patients * Configurator(self.__cfg_path).getfloat('training', 'test_size')))

        test_patients = list()

        if p_ids is not None:
            test_patients = p_ids
            patient_ids = list(patient_ids)
            patient_ids.remove(test_patients)
        else:
            # Select patient for validation
            for _ in range(n_patient_validation):
                # Select random patient
                pos = random.randint(0, len(patient_ids) - 1)
                random_p_id = patient_ids[pos]
                patient_ids = np.delete(patient_ids, pos)

                # Keep track of selected patients
                test_patients.append(random_p_id)

        logging.info('--> Training subjects: %s' % str(patient_ids))
        logging.info('--> Testing subjects: %s' % str(test_patients))

        # Define intra patient training and validation datasets
        training_dataset = ds.loc[ds['P_ID'].isin(patient_ids)].sort_values(by=['P_ID', 'time'])
        validation_dataset = ds.loc[ds['P_ID'].isin(test_patients)].sort_values(by=['P_ID', 'time'])

        return training_dataset, validation_dataset

    def _get_pbp_window_train_and_test(self, ds, training_dataset, validation_dataset, tw_size):
        cnn_training_data = list()
        cnn_training_labels = list()

        cnn_validation_data = list()
        cnn_validation_labels = list()

        for p_id in training_dataset['P_ID'].unique():
            data_i, label_i, _ = self._get_window(ds.loc[ds[(ds['P_ID'] == p_id)].index],
                                                  Configurator(self.__cfg_path).getint('settings',
                                                                                       'sampling_frequency'),
                                                  tw_size,
                                                  Configurator(self.__cfg_path).getfloat('settings', 'overlap'))

            cnn_training_data.append(data_i)
            cnn_training_labels.append(label_i)

        for val_p_id in validation_dataset['P_ID'].unique():
            data_i, label_i, _ = self._get_window(ds.loc[ds[(ds['P_ID'] == val_p_id)].index],
                                                  Configurator(self.__cfg_path).getint('settings',
                                                                                       'sampling_frequency'),
                                                  tw_size,
                                                  Configurator(self.__cfg_path).getfloat('settings', 'overlap'))

            cnn_validation_data.append(data_i)
            cnn_validation_labels.append(label_i)

        # Create CNN format training dataset and labels
        cnn_training_data = np.concatenate(cnn_training_data)
        cnn_training_labels = np.concatenate(cnn_training_labels)

        # Create CNN format training dataset and labels
        cnn_validation_data = np.concatenate(cnn_validation_data)
        cnn_validation_labels = np.concatenate(cnn_validation_labels)

        return cnn_training_data, cnn_validation_data, cnn_training_labels, cnn_validation_labels

    def _data_preprocessing(self, df, drop_class, drop_patient, split_method, normalisation_method, selection_method,
                            balancing_method, ids_test_set: list = None):
        widgets = [
            'Data Preprocessing',
            ' [', progressbar.Timer(), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
        ]
        logging.info('--- Data Preprocessing ---')

        with progressbar.ProgressBar(widgets=widgets, max_value=8) as bar:
            # Drop unnecessary classes
            if drop_class is not None:
                for class_name in drop_class:
                    indexes = df[(df['CLASS'] == class_name)].index
                    df.drop(indexes, inplace=True)

            # Drop unnecessary patients
            if drop_patient is not None:
                for patient_id in drop_patient:
                    indexes = df[(df['P_ID'] == patient_id)].index
                    df.drop(indexes, inplace=True)

            bar.update()

            # Class renaming
            if df['CLASS'].unique().dtype == int:
                old_labels = sorted(df['CLASS'].unique())
            else:
                old_labels = df['CLASS'].unique()
            original_name_labels = list()
            logging.info('--> New labels:')
            for new, old in zip(range(len(old_labels)), old_labels):
                logging.info('• %s --> %s' % (str(old), str(new)))
                original_name_labels.append(str(old))
                df['CLASS'] = df['CLASS'].replace({old: new})

            bar.update()

            # Train/Test split
            header_type = Configurator(self.__cfg_path).get('dataset', 'header_type')
            if split_method == 'inter' and 'p' in header_type:
                # Define inter patient training and validation datasets
                training_dataset, validation_dataset = self._select_patients_training_test(df, ids_test_set)
            elif split_method == 'intra' and 'p' in header_type or split_method == 'holdout':
                training_dataset, validation_dataset = train_test_split(df,
                                                                        shuffle=True,
                                                                        test_size=Configurator(self.__cfg_path).getfloat(
                                                                            'training',
                                                                            'test_size'),
                                                                        random_state=28)
            else:
                training_dataset = None
                validation_dataset = None
                logging.info('*** Error: %s not a valid train/test split method ***' % split_method)
                print('*** Error: %s not a valid train/test split method ***' % split_method)
                exit(12)

            bar.update()

            # Get training data and labels
            if 'P_ID' in training_dataset.columns:
                # Drop P_ID if exists
                training_data = training_dataset.drop(['CLASS', 'P_ID'], axis=1)
                validation_data = validation_dataset.drop(['CLASS', 'P_ID'], axis=1)

            else:
                training_data = training_dataset.drop(['CLASS'], axis=1)
                validation_data = validation_dataset.drop(['CLASS'], axis=1)

            training_labels = training_dataset['CLASS']
            validation_labels = validation_dataset['CLASS']

            del training_dataset, validation_dataset

            bar.update()

            # Normalisation
            if normalisation_method == 'minmax':
                scaler = preprocessing.MinMaxScaler()
            elif normalisation_method == 'robust':
                scaler = preprocessing.RobustScaler()
            else:
                scaler = preprocessing.StandardScaler()

            training_data = scaler.fit_transform(training_data)
            validation_data = scaler.transform(validation_data)

            bar.update()

            if Configurator(self.__cfg_path).get('settings', 'data_treatment') == 'features_extraction':
                # Create a directory in order to save extracted features names
                features_path = os.path.join(Configurator(self.__cfg_path).get('settings', 'log_dir'), 'features')
                Path(features_path).mkdir(parents=True, exist_ok=True)

                with open(os.path.join(features_path, 'extracted_features.rtf'), 'a+') as exf:
                    exf.write('%s' % '\n'.join(list(df.columns)))
            else:
                features_path = None

            bar.update()

            if Configurator(self.__cfg_path).getboolean('settings', 'features_selection') and Configurator(
                    self.__cfg_path).get('settings', 'data_treatment') == 'features_extraction':
                # Highly correlated features are removed
                # corr_features = ts.correlated_features(training_data)
                # training_data.drop(corr_features, axis=1, inplace=True)
                # validation_data.drop(corr_features, axis=1, inplace=True)

                if selection_method == 'variance':
                    # Remove low variance features
                    selector = VarianceThreshold()
                    training_data = selector.fit_transform(training_data)
                    validation_data = selector.transform(validation_data)

                elif selection_method == 'l1':
                    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(training_data, training_labels)
                    model = SelectFromModel(lsvc, prefit=True)
                    training_data = model.fit_transform(training_data)
                    validation_data = model.transform(validation_data)

                elif selection_method == 'tree-based':
                    clf = ExtraTreesClassifier(n_estimators=50)
                    clf = clf.fit(training_data, training_labels)
                    model = SelectFromModel(clf, prefit=True)
                    training_data = model.fit_transform(training_data)
                    validation_data = model.transform(validation_data)

                elif selection_method == 'recursive':
                    estimator = SVR(kernel="linear")
                    selector = RFE(estimator, n_features_to_select=5, step=1)
                    training_data = selector.fit_transform(training_data, training_labels)
                    validation_data = selector.transform(validation_data)

                else:
                    logging.info('*** Error: %s not implemented features selection technique ***' % str(selection_method))
                    print('*** Error: %s not implemented features selection technique ***' % str(selection_method))
                    exit(5)

                logging.info('--> Features selected: %s' % str(validation_data.shape[1]))

            bar.update()

            # Balancing
            resampling_technique = Configurator(self.__cfg_path).get('settings', 'resampling')
            if resampling_technique == 'under':
                if balancing_method == 'random_under':
                    sampler = RandomUnderSampler()
                elif balancing_method == 'near_miss':
                    sampler = NearMiss()
                elif balancing_method == 'edited_nn':
                    sampler = EditedNearestNeighbours()
                else:
                    logging.info(
                        '*** Fallback: not recognised %s, using RandomUnderSampler instead ***' % balancing_method)
                    sampler = RandomUnderSampler()
            elif resampling_technique == 'over':
                if balancing_method == 'smote':
                    sampler = SMOTE()
                elif balancing_method == 'adasyn':
                    sampler = ADASYN()
                elif balancing_method == 'kmeans_smote':
                    sampler = KMeansSMOTE(k_neighbors=3)
                elif balancing_method == 'random_over':
                    sampler = RandomOverSampler()
                else:
                    logging.info(
                        '*** Fallback: not recognised %s, using RandomOverSampler instead ***' % balancing_method)
                    sampler = RandomOverSampler()
            else:
                balancing_method = ''
                sampler = None

            if sampler is not None:
                training_data, training_labels = sampler.fit_resample(training_data, training_labels)
            else:
                training_data, training_labels = training_data, training_labels

            bar.update()

            logging.info('--> Applied %s %s' % (resampling_technique, balancing_method))
            logging.info('--> Train test shape: %s' % str(training_data.shape))
            logging.info('--> Validation test shape: %s' % str(validation_data.shape))

        return training_data, validation_data, training_labels.values, validation_labels.values, original_name_labels

    def _pbp_features_extraction(self, df, tw_size):
        patient_ids = df['P_ID'].unique()
        features = list()

        for p_id in patient_ids:
            features_dataset_i = self._features_extraction(df.drop(df[(df['P_ID'] != p_id)].index),
                                                           Configurator(self.__cfg_path).getint('settings',
                                                                                                'sampling_frequency'),
                                                           tw_size,
                                                           Configurator(self.__cfg_path).getfloat('settings',
                                                                                                  'overlap'),
                                                           Configurator(self.__cfg_path).getfloat('dataset',
                                                                                                  'header_type'))
            features.append(features_dataset_i)

        # Features dataset of all patients
        features_dataset = pd.concat(features, ignore_index=True)
        return features_dataset

    def _derive_headers(self, columns, header_type: str = None):

        # if the data are in the last columns, we need to calculate n in a different way
        if header_type[len(header_type)-1] == "d":
            n = columns - self._data_delimiters[header_type]
        else:
            n = columns + self._data_delimiters[header_type][1] - self._data_delimiters[header_type][0]
        header = []

        for i in header_type:
            if i == 't':
                header.append('time')
            if i == 'c':
                header.append('CLASS')
            if i == 'p':
                header.append('P_ID')
            if i == 'd':
                axis = ['x', 'y', 'z']
                column_signature = 'A'
                accelerometer_name = 0

                for column_index, ax in zip(range(n), cycle(axis)):
                    if (column_index % 3) == 0:
                        accelerometer_name += 1
                    header.append('%s%s%s' % (column_signature, str(accelerometer_name), ax))

        return header

    def _decode_csv(self, ds_dir: str, header_type, separator: str = ' ', has_header: bool = False):
        frames = list()
        is_first = True
        header = None
        widgets = [
            'Loading Dataset',
            ' [', progressbar.Timer(), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
        ]

        with progressbar.ProgressBar(widgets=widgets, max_value=len(os.listdir(ds_dir))) as bar:
            for file in os.listdir(ds_dir):
                if (file.endswith('.csv') or file.endswith('.txt')) and not file.startswith('.'):
                    # Derive header if not exists
                    if is_first and not has_header:
                        is_first = False
                        header = self._derive_headers(pd.read_csv(os.path.join(ds_dir, file), sep=separator).shape[1], header_type)

                    # Append dataframes
                    if has_header:
                        frames.append(pd.read_csv(os.path.join(ds_dir, file), sep=separator))
                    else:
                        df_i = pd.read_csv(os.path.join(ds_dir, file), sep=separator, header=None)
                        df_i.columns = header
                        frames.append(df_i)
                bar.update()
        df = pd.concat(frames, ignore_index=True)

        array_df = []

        # if there are more than 1 class, I need to create an array containing a subdataset for each class for each patient,
        # otherwise I just need to group the data by the P_ID
        if df['CLASS'].nunique() != 1:
            multiclass = True
            idx = np.where(np.roll(df['CLASS'], 1) != df['CLASS'])[0]
            for st,en in zip(idx , idx[1:]):
                array_df.append(df[st:en])
        else:
            multiclass = False
            array_df = df.groupby(['P_ID', 'CLASS'])

        return array_df

    def _clean_data(self, df, sampling_frequency, high_cutoff, low_cutoff, sub_method, header_type):


        widgets = [
            'Cleaning Data',
            ' [', progressbar.Timer(), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
        ]

        logging.info('--- Cleaning data ---')
        df_filtered = []
        with progressbar.ProgressBar(widgets=widgets, max_value=2) as bar:

            for d in df:

                # Handle Data Errors: clean from NaN and infinite values
                d.replace([math.inf, -math.inf], np.nan, inplace=True)
                if sub_method == 'mean':
                    d.fillna(d.mean(), inplace=True)
                elif sub_method == 'forward':
                    d.fillna(method='ffill', inplace=True)
                elif sub_method == 'backward':
                    d.fillna(method='bfill', inplace=True)
                elif sub_method == 'constant':
                    d = d.fillna(Configurator(self.__cfg_path).getfloat('cleaning', 'constant_value'))
                else:
                    logging.info('*** Error: %s is not a valid substitution method, skipped ***' % sub_method)

                df_filtered.append(d.reset_index(drop=True))
                bar.update()

        # Apply filter
        filter_name = Configurator(self.__cfg_path).get('cleaning', 'filter')
        filter_order = Configurator(self.__cfg_path).getint('cleaning', 'filter_order')

        if 'low' in filter_name:
            cutoff = low_cutoff
        elif 'high' in filter_name:
            cutoff = high_cutoff
        elif filter_name == 'bandpass':
            cutoff = (high_cutoff, low_cutoff)
        elif filter_name == 'no':
            cutoff = None
        else:
            cutoff = None
            logging.info('*** Error: invalid filter name %s ***' % filter_name)
            print('*** Error: invalid filter name %s ***' % filter_name)
            exit(11)

        for d in np.arange(0,len(df)):
            if cutoff is not None:
                df_filtered[d] = pd.DataFrame(
                    self._apply_filter(
                        df=df_filtered[d],
                        filter_name=filter_name,
                        sample_rate=sampling_frequency,
                        frequency_cutoff=cutoff,
                        order=filter_order
                    ),
                    columns=list(df[d].columns[self._data_delimiters[header_type][0]: self._data_delimiters[header_type][1]])
                )

                # Add class and other to the filtered dataset
                df_filtered[d]['CLASS'] = df[d]['CLASS']
                if 'p' in header_type:
                    df_filtered[d]['P_ID'] = df[d]['P_ID']
                if 't' in header_type:
                    df_filtered[d].insert(0, 'time', df[d]['time'])
        bar.update()

        return df_filtered

    def _features_extraction(self, df, sampling_frequency, time_window, overlap, header_type):

        features_df = []
        for d in df:
            x = d.drop(['CLASS'], axis=1)
            if 'p' in header_type:
                x = x.drop(['P_ID'], axis=1)
            if 't' in header_type:
                x = x.drop(['time'], axis=1)

            y = d['CLASS']

            # Available domains: "statistical"; "spectral"; "temporal"
            domain = Configurator(self.__cfg_path).get('settings', 'features_domain')
            if domain == 'all':
                cfg = ts.get_features_by_domain()
            else:
                cfg = ts.get_features_by_domain(domain)

            tsfel_overlap = round(overlap/Configurator(self.__cfg_path).getfloat('settings', 'time'), 3)
            # Features Extraction
            X_features = ts.time_series_features_extractor(cfg,
                                                           x,
                                                           fs=sampling_frequency,
                                                           window_size=time_window,
                                                           overlap=tsfel_overlap,
                                                           verbose=1)
            Y_features = self._labels_windowing(y, sampling_frequency, time_window, overlap)

            # Handling eventual missing values from the feature extraction
            X_features = self._fill_missing_values(X_features)

            logging.info('--> Features extracted: %s' % str(X_features.shape[1]))

            # Attach labels
            features_df.append(pd.DataFrame(X_features))
            features_df['CLASS'] = Y_features

        print ("STOP")

        return features_df

    def _start_ml_evaluation(self, x_train, y_train, x_test, y_test, models_and_params):
        widgets = [
            'ML Evaluation',
            ' [', progressbar.Timer(), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
        ]

        # Create ml directory if not exists
        Path(os.path.join(Configurator(self.__cfg_path).get('settings', 'log_dir'), 'machine_learning')).mkdir(
            parents=True,
            exist_ok=True)

        report_raw = list()
        logging.info('--- Starting ML evaluations ---')

        # Get selected models and params
        mp = dict()
        for m in models_and_params:
            mp.update({m: MlModels().clf_model_params[m]})

        with progressbar.ProgressBar(widgets=widgets, max_value=len(mp.keys())) as bar:
            for name, param in mp.items():
                logging.info(
                    '--> Model %s: %ss time window' % (name, Configurator(self.__cfg_path).get('settings', 'time')))
                clf = GridSearchCV(param['model'], param['param_grid'], scoring='accuracy',
                                   n_jobs=-1)  # scoring='recall_macro'

                # Raw data
                start_training = timer()
                clf.fit(x_train, y_train)
                stop_training = timer()
                logging.info('--> Elapsed time training: %s' % self._to_readable_time(stop_training - start_training))

                start_prediction = timer()
                y_pred = clf.predict(x_test)
                stop_prediction = timer()
                logging.info('--> Elapsed time prediction: %s' % self._to_readable_time(stop_prediction - start_prediction))

                # Metrics
                metrics1 = sensitivity_specificity_support(y_test, y_pred, average='weighted')
                metrics2 = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                metrics3 = accuracy_score(y_test, y_pred, normalize=True)

                # Create Report
                report_raw.append({
                    'clf': name,
                    'y_pred': y_pred,
                    'sensitivity': metrics1[0],
                    'specificity': metrics1[1],
                    'precision': metrics2[0],
                    'recall': metrics2[1],
                    'f1-score': metrics2[2],
                    'accuracy': metrics3
                })

                self._cm_analysis(y_test, y_pred, np.unique(y_train), '', name, 'machine_learning')

                logging.log(msg=classification_report(y_test, y_pred), level=logging.INFO)
                logging.log(msg='--> clf: %s\n'
                                '   • specificity: %s\n'
                                '   • sensitivity: %s\n'
                                '   • precision: %s\n'
                                '   • accuracy: %s\n'
                                '   • recall: %s\n'
                                '   • f1-score: %s\n' % (
                                    name, metrics1[1], metrics1[0], metrics2[0], metrics3, metrics2[1], metrics2[0]),
                            level=logging.INFO)
                bar.update()

    @staticmethod
    def _to_readable_time(t):
        unit = 's'
        if t > 60:
            unit = 'm'
            t = t / 60
        if t > 60:
            unit = 'h'
            t = t / 60
        return '%s%s' % (str(round(t, 3)), unit)

    def _start_cnn_evaluation(self, x_train, y_train, x_test, y_test, models_name: list, epochs, n_fold: int,
                              me_loss_threshold, class_values, title):
        widgets = [
            'DL Evaluation',
            ' [', progressbar.Timer(), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
        ]

        # Create ml directory if not exists
        Path(os.path.join(Configurator(self.__cfg_path).get('settings', 'log_dir'), 'deep_learning')).mkdir(
            parents=True,
            exist_ok=True)

        logging.log(msg='--- Starting CNN evaluations ---', level=logging.INFO)
        kf = KFold(n_fold, shuffle=True, random_state=0)

        with progressbar.ProgressBar(widgets=widgets, max_value=len(models_name)) as bar:
            for model_name in models_name:
                # Set output size of CNN
                PrivateConfigurator().set('cnn', 'output', str(len(np.unique(y_train))))

                logging.log(msg='--> Model %s: %s time window, %s epochs, %s me threshold' %
                                (model_name, x_train[0].shape, epochs, me_loss_threshold), level=logging.INFO)
                # Get the model
                model = Models(Configurator(self.__cfg_path).getfloat('settings', 'time'),
                               Configurator(self.__cfg_path).getfloat('settings', 'sampling_frequency')).get_model(
                    model_name)

                # Train the model
                start_training = timer()
                score, trained_models = self._do_k_fold(x_train, y_train, kf, model, '%s %s' % (model_name, title),
                                                        epochs)
                stop_training = timer()
                logging.info('--> Elapsed time training: %s' % self._to_readable_time(stop_training - start_training))

                # Model Ensembles
                start_prediction = timer()
                self._model_ensembles(trained_models, model, x_test, y_test, '%s %s' % (model_name, title),
                                      threshold=me_loss_threshold,
                                      class_values=class_values)
                stop_prediction = timer()
                logging.info(
                    '--> Elapsed time prediction: %s' % self._to_readable_time(stop_prediction - start_prediction))
                bar.update()

    def _get_window(self, df: pd.DataFrame, sampling_frequency: int, window_size: int, overlap: float):
        hop_size = window_size - int(sampling_frequency * overlap)

        data = list()
        labels = list()
        patients = list()
        has_patient = 'p' in Configurator(self.__cfg_path).get('dataset', 'header_type')

        columns = df.columns[self._data_delimiters['tdcp'][0]: self._data_delimiters['tdcp'][1]]
        for i in range(0, len(df) - window_size + 1, hop_size):
            window = list()
            for column in columns:
                x_i = df[column].values[i: i + window_size]
                window.append(x_i)

            # Associate a label for the current window based on mode
            label = stats.mode(df['CLASS'].values[i: i + window_size])[0][0]

            data.append(np.array(window).T)
            labels.append(label)

            if has_patient:
                patients.append(stats.mode(df['P_ID'].values[i: i + window_size])[0][0])

        if has_patient:
            return np.asarray(data), np.asarray(labels), np.asarray(patients)
        else:
            return np.asarray(data), np.asarray(labels)

    @staticmethod
    def _get_pbp_train(data, labels, balancing, n_features, time_window):
        # Return normal or balanced data
        if balancing == 'normal':
            # Return normal data
            return data, labels

        elif balancing == 'over':
            # Oversample accelerometer data
            oversample = SMOTE()
            X_over, y_over = oversample.fit_resample(data.reshape(-1, n_features),
                                                     np.concatenate([[x] * time_window for x in labels]))

            # Restore original window format
            X_training_over = X_over.reshape(-1, time_window, n_features)
            y_training_over = np.asarray([y_over][0])[::time_window]

            return X_training_over, y_training_over

        elif balancing == 'under':
            # Undersample accelerometer data
            undersample = RandomUnderSampler()
            X_under, y_under = undersample.fit_resample(data.reshape(-1, n_features),
                                                        np.concatenate([[x] * time_window for x in labels]))

            # Restore original window format
            X_training_under = X_under.reshape(-1, time_window, n_features)
            y_training_under = np.asarray([y_under][0])[::time_window]

            return X_training_under, y_training_under

    @staticmethod
    def _get_train_and_test(data, labels, balancing, time_window, n_features, test_size):
        # Split data into training and test set
        X_training, X_testing, y_training, y_testing = train_test_split(data, labels,
                                                                        test_size=test_size,
                                                                        shuffle=True,
                                                                        random_state=0)

        # Return normal or balanced data
        if balancing == 'normal':
            # Return normal data
            return X_training, X_testing, y_training, y_testing

        elif balancing == 'over':
            # Oversample accelerometer data
            oversample = SMOTE()
            X_over, y_over = oversample.fit_resample(X_training.reshape(-1, n_features),
                                                     np.concatenate([[x] * time_window for x in y_training]))

            # Restore original window format
            X_training_over = X_over.reshape(-1, time_window, n_features)
            y_training_over = np.asarray([y_over][0])[::time_window]

            return X_training_over, X_testing, y_training_over, y_testing

        elif balancing == 'under':
            # Training set balancing using Random Under sampling

            # Get indexes of each class
            indexes = list()
            for cls in np.unique(labels):
                indexes.append(np.where(y_training == cls)[0])

            # Get the min class
            min_cls = math.inf
            for cls in indexes:
                if len(cls) <= min_cls:
                    min_cls = len(cls)

            # Shuffle indexes
            for cls in indexes:
                random.Random(82).shuffle(cls)

            # Build up
            X_training_under = np.vstack(X_training[x][:min_cls] for x in indexes)
            y_training_under = np.hstack([cls_val] * min_cls for cls_val in np.unique(labels))

            # Zero Mean
            training_mean = X_training_under.reshape((-1, n_features)).mean(axis=0)
            training_std = X_training_under.reshape((-1, n_features)).std(axis=0)

            X_training_under = (X_training_under - training_mean) / training_std
            X_testing = (X_testing - training_mean) / training_std

            return X_training_under, X_testing, y_training_under, y_testing

    @staticmethod
    def _labels_windowing(labels, sampling_frequency, time_window, overlap):
        hop_size = time_window - int(sampling_frequency * overlap)
        windowed_labels = list()

        for i in range(0, len(labels) - time_window + 1, hop_size):
            new_label = stats.mode(labels.values[i: i + time_window])[0][0]
            windowed_labels.append(new_label)

        return np.asarray(windowed_labels)

    @staticmethod
    def _fill_missing_values(df):
        """ Handle eventual missing data. Strategy: replace with mean.

          Parameters
          ----------
          df pandas DataFrame
          Returns
          -------
            Data Frame without missing values.
        """
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.mean(), inplace=True)
        return df
