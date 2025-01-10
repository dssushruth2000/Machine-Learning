import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
import pandas as pd
import random


def load_data():
    training_set = image_dataset_from_directory(
        "Monkey Species Data/Training Data",
        label_mode="categorical",
        image_size=(100, 100),
        batch_size=32,
    )
    test_set = image_dataset_from_directory(
        "Monkey Species Data/Prediction Data",
        label_mode="categorical",
        image_size=(100, 100),
        batch_size=32,
        shuffle=False
    )
    return training_set, test_set


def build_model_1():
    model = Sequential([
        Input((100, 100, 3)),
        Rescaling(1./255),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_model_2():
    model = Sequential([
        Input((100, 100, 3)),
        Rescaling(1./255),
        Conv2D(16, (5, 5), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D((3, 3)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def fine_tune_model():
    base_model = EfficientNetV2S(include_top=False, input_shape=(100, 100, 3))
    base_model.trainable = False  

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)


    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model




def train_and_evaluate(model, training_set, test_set, epochs=30):   
    history = model.fit(
        training_set, 
        epochs=epochs, 
        validation_data=test_set, 
        verbose=1,
    )
    
    test_loss, test_acc = model.evaluate(test_set)
    return model, history, test_loss, test_acc


def save_history(history, filename):
    pd.DataFrame(history.history).to_csv(f"{filename}_history.csv")




def plot_confusion_matrix(predictions, true_labels, title='Confusion Matrix'):
    cm = confusion_matrix(true_labels, predictions) 
    print(title)
    print("Confusion Matrix:")
    print(cm)


def main():
    training_set, test_set = load_data()
    
    model1, history1, loss1, acc1 = train_and_evaluate(build_model_1(), training_set, test_set)
    save_history(history1, "model1")
    
    model2, history2, loss2, acc2 = train_and_evaluate(build_model_2(), training_set, test_set)
    save_history(history2, "model2")
    
    model3, history3, loss3, acc3 = train_and_evaluate(fine_tune_model(), training_set, test_set)
    save_history(history3, "model3")


    plot_learning_curves(history1, 'Model 1')
    plot_learning_curves(history2, 'Model 2')
    plot_learning_curves(history3, 'Fine-tuned Model')

    print(f"Accuracy of Model 1: {acc1}\nAccuracy of Model 2: {acc2}\nAccuracy of Fine-tuned Model: {acc3}")


    if acc1 >= acc2:
        best_model = model1
        best_acc = acc1
        best_model.save("best_model1.keras")
        print("Model 1 is saved as the best model.")
    else:
        best_model = model2
        best_acc = acc2
        best_model.save("best_model2.keras")
        print("Model 2 is saved as the best model.")


    model3.save("fine_tuned_model.keras")
    print("Fine-tuned Model saved.")



    predictions1 = np.argmax(best_model.predict(test_set), axis=1)
    predictions3 = np.argmax(model3.predict(test_set), axis=1)


    true_labels = np.concatenate([y for x, y in test_set], axis=0)
    true_labels = np.argmax(true_labels, axis=1)

    plot_confusion_matrix(predictions1, true_labels, title='Confusion Matrix for Best Model')
    plot_confusion_matrix(predictions3, true_labels, title='Confusion Matrix for Fine-tuned Model')


    misclassified_indices = np.where(predictions1 != true_labels)[0]
    print(f"Number of misclassified images: {len(misclassified_indices)}")

    random_indices = random.sample(range(len(misclassified_indices)), min(10, len(misclassified_indices)))

    print(f"Displaying up to {min(10, len(misclassified_indices))} misclassified images:")
    for i in random_indices:
        img_index = misclassified_indices[i]
        img_path = test_set.file_paths[img_index]
        img = image.load_img(img_path, target_size=(100, 100))
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f'Best Model Misclassified: Predicted {predictions1[img_index]}, True {true_labels[img_index]}')
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.title(f'Fine-tuned Model Prediction: Predicted {predictions3[img_index]}')
        plt.show()

def plot_learning_curves(history, title='Model Learning Curves'):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(1, len(accuracy) + 1)
    
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, accuracy, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r-', label='Validation Accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()



