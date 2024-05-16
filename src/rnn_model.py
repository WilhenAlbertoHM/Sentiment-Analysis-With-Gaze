import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, SimpleRNN, LSTM
import numpy as np
import matplotlib.pyplot as plt 


"""
load_data: Loads organized time series data; features.npy and labels.npy.
Param: offset - The offset or gap between time steps.
Returns: Data regarding features and labels as a pair.
"""
def load_data(offset=1):
    file_features = "features_60_steps.npy"
    file_labels = "labels_60_steps.npy"
    features = np.load(file_features)
    labels = np.load(file_labels)

    # Display shapes
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    return features[::offset], labels[::offset]

"""
main: The main function.
"""
def main():
    # Organize data into time series by creating two .npy files 
    # for data and labels. After, load data into x and y.
    # organize_into_time_series(n_steps=60)
    x, y = load_data(offset=5)
    
    # Print the shapes.
    print(f"X shape: {x.shape}")
    print(f"Y shape: {y.shape}")

    # Split the data into training and testing sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42, shuffle=True)

    # Further split the training set into training and validation sets.
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42, shuffle=True)

    # Scale the data.
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
    x_val = scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape)

    # Get the timesteps (30), number of features (13), and number of labels (5).
    n_timesteps, n_features = x.shape[1], x.shape[2]
    n_labels = y.shape[1]

    # Create the model.
    model = Sequential()
    model.add(Input(shape=(n_timesteps, n_features)))
    model.add(SimpleRNN(units=128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(units=n_labels, activation="relu"))
    model.summary()

    # Early stopping callback.
    early_stopping = EarlyStopping(monitor="val_loss", patience=2)
    
    # Compile and train the model.
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])    
    history = model.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_val, y_val), callbacks=[early_stopping])

    # Evaluate the model.
    mse, mae = model.evaluate(x_test, y_test)
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    

    # Predict the labels.
    y_pred = model.predict(x_test)
    y_diff = np.abs(np.subtract(y_pred, y_test))
    y_diff_avg = np.mean(y_diff, axis=0)
    print(f"y_pred:\n {y_pred}")
    print(f"y_test:\n {y_test}")
    print(f"y_pred - y_test:\n {y_diff}")
    print(f"y_diff_avg:\n {y_diff_avg}")

    # Plot training and testing loss
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.plot(history.history["val_mae"], label="MAE Valid", color="red")
    plt.plot(history.history["mae"], label="MAE Train", color="blue")
    plt.plot(history.history["val_loss"], label="MSE Valid", color="orange")
    plt.plot(history.history["loss"], label="MSE Train", color="green")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()