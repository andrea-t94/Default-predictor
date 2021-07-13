import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.models import load_model
import warnings

class BaselineModel:
    '''BaselineModel class, based on a shallow NN.
    Contains all the methods for model generation and management'''
    def __init__(self, name, uuid = None):
        self.name = name
        self.uuid = str(uuid)
        if uuid:
            try:
                os.makedirs("models/" + self.name + "/" + self.uuid)
            except:
                pass

    def build(self, input_shape, config):
        #define the parameters
        self.config = config
        self.input_layer_shape = input_shape

        #build the shallow model
        input_layer = Input(shape=self.input_layer_shape)
        x = Dense(self.config["dense"], activation='relu')(input_layer)
        x = Dropout(self.config["dropout"])(x)
        out = Dense(1, activation='sigmoid')(x)
        self.model = Model(name='default-predictor', inputs=input_layer, outputs=out)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config["learning_rate"]
        )
        self.model.compile(
            optimizer=self.optimizer,
            loss="binary_crossentropy",
            metrics=['accuracy']
        )
        self.model.summary()

    def train(self, x_train, y_train, validation):
        self.model.fit(
            x_train,
            y_train,
            validation_data=validation,
            batch_size=self.config["batch_size"],
            epochs=self.config["epochs"],
            class_weight=self.config["class_weight"],
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    filepath= "models/" + self.name + "/" + self.uuid + "/{val_accuracy:.2f}.h5",
                    monitor='val_loss', save_best_only=True, verbose=1)
            ]
        )

    def test(self, x_test, y_test, return_score=True):
        # predict crisp classes for test set
        yhat_classes = np.argmax(self.model.predict(x_test, verbose=0), axis=1)
        #Y_test = np.argmax(y_test, axis=-1)
        acc = round(accuracy_score(y_test, yhat_classes), 4) * 100
        precision = round(precision_score(y_test, yhat_classes, average='weighted'), 4) * 100
        recall = round(recall_score(y_test, yhat_classes, average='weighted'), 4) * 100
        f1 = round(f1_score(y_test, yhat_classes), 4) * 100
        # area under roc curve
        auroc = round(roc_auc_score(y_test, yhat_classes), 4) * 100
        tn, fp, fn, tp = confusion_matrix(y_test, yhat_classes).ravel()
        confusion_matrix_abs_nums = {"FP": fp, "FN": fn, "TN": tn, "TP": tp}

        score = {
            "Accuracy %": acc,
            "Precision %": precision,
            "Recall %": recall,
            "F1-Score %": f1,
            "AUROC %": auroc,
            "ConfusionMatrixCounts": confusion_matrix_abs_nums
        }
        if return_score:
            return score
        else:
            print("Scores on test data: " + score)

    def load_best_model(self):
        #directory in which store models with unique uuid
        models_dir =  "models/" + self.name + "/"
        self.best_acc = 0
        self.best_model_path = None
        for dirs in os.listdir(models_dir):
            tmp_models_subdir = models_dir + dirs + "/"
            print(tmp_models_subdir)
            try:
                for model_file in os.listdir(tmp_models_subdir):
                    model_acc = float(os.path.splitext(model_file)[0])
                    if model_acc > self.best_acc:
                        self.best_acc = model_acc
                        self.best_model_path = tmp_models_subdir + model_file
            except:
                warnings.warn("Problems with directory {}".format(tmp_models_subdir))
        if self.best_model_path:
            self.model = load_model(self.best_model_path)
        else:
            raise FileNotFoundError("No trained model found in {}".format(models_dir))

    def class_prediction(self, input):
        if hasattr(self, 'model'):
            return self.model.predict(input, verbose=0) #ndarray not JSON serializable
        else:
            raise ValueError("No loaded model found, please use load_best_model method before")

