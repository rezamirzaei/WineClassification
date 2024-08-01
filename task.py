#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 17:46:17 2023

@author: rezami
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.datasets import load_wine
from tensorflow import keras
import pandas as pd
import numpy as np

class Controller:

    def import_data(self):
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df

    def separate_data(self, df):
        X = df.drop('target', axis=1)
        y = df['target']
        return X, y

    def create_train_test_sets(self, X, y):
        return train_test_split(X, y, test_size=0.2)

    def standardize_data(self, X_train, X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    def create_model(self):
        model = keras.Sequential([
            keras.layers.Dense(256, activation='elu'),
            keras.layers.Dense(128, activation='elu'),
            keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    

    def train_ai(self, model, X_train, X_test, y_train, y_test):
        history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
        return model, history
    
    def visualize_results(self, history,title):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
   

    def plot_confusion_matrix(self, y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.show()

    def compare_models(self, svm_scores, nn_scores, title):
        plt.plot(svm_scores, label='SVM')
        plt.plot(nn_scores, label='Neural Network')
        plt.title(title)
        plt.xlabel('Epoch (for Neural Network)')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
    def visualize_data(self, df):
    # Class Distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x='target')
        plt.title('Class Distribution')
        plt.show()

    # Feature Distribution
        df.drop('target', axis=1).hist(bins=15, figsize=(15, 10), layout=(5, 3))
        plt.suptitle('Feature Distribution')
        plt.show()

    # Correlation Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.show()

    # Pairplot
        sns.pairplot(df, hue='target', diag_kind='hist')
        plt.title('Pair Plots')
        plt.show()   
        
    def run(self):
        df = self.import_data()
        self.visualize_data(df)
        X, y = self.separate_data(df)
        X_train, X_test, y_train, y_test = self.create_train_test_sets(X, y)
        X_train, X_test = self.standardize_data(X_train, X_test)

        # Neural Network
        model_nn = self.create_model()
        model_nn, history_nn = self.train_ai(model_nn, X_train, X_test, y_train, y_test)

        y_pred_nn_train = np.argmax(model_nn.predict(X_train), axis=-1)
        y_pred_nn_test = np.argmax(model_nn.predict(X_test), axis=-1)
        self.visualize_results(history_nn, 'Neural Network Accuracy')
        self.plot_confusion_matrix(y_train, y_pred_nn_train, 'Confusion Matrix for Neural Network (Training)')
        self.plot_confusion_matrix(y_test, y_pred_nn_test, 'Confusion Matrix for Neural Network (Validation)')

        # SVM
        model_svm = LinearSVC(penalty='l1', dual=False, C=1.0)
        model_svm.fit(X_train, y_train)
        y_pred_svm_train = model_svm.predict(X_train)
        y_pred_svm_test = model_svm.predict(X_test)
        self.visualize_results(history_nn, 'Neural Network Accuracy')
        self.plot_confusion_matrix(y_train, y_pred_svm_train, 'Confusion Matrix for SVM (Training)')
        self.plot_confusion_matrix(y_test, y_pred_svm_test, 'Confusion Matrix for SVM (Validation)')

        # Comparison
        svm_train_acc = accuracy_score(y_train, y_pred_svm_train)
        svm_test_acc = accuracy_score(y_test, y_pred_svm_test)
        nn_train_accs = history_nn.history['accuracy']
        nn_test_accs = history_nn.history['val_accuracy']
        
        self.compare_models([svm_train_acc]*len(nn_train_accs), nn_train_accs, 'Training Accuracy Comparison')
        self.compare_models([svm_test_acc]*len(nn_test_accs), nn_test_accs, 'Validation Accuracy Comparison')

if __name__ == '__main__':
    controller = Controller()
    controller.run()