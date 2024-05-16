import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
import numpy as np
import matplotlib.pyplot as plt 

"""
organize_into_time_series: Organizes the data into time series, displayed in two .npy files, 
                           for features and labels.
Param: n_steps - The number of time steps (30 fps from a video).
Returns: None.
"""
def organize_into_time_series(n_steps=30):
    # Load the data.
    data = np.genfromtxt("csv_files/cleaned_data_use_this.csv", delimiter=",")
    
    # For NaN values, replace them with the average of the column.
    mean_values = np.nanmean(data, axis=0)
    nan_indices = np.isnan(data)
    data[nan_indices] = np.take(mean_values, np.where(nan_indices)[1])

    # Get the window size by the start and end indices
    start_index = n_steps
    end_index = data.shape[0]
    
    # Append features and labels in specific time steps.
    features = []
    labels = []
    feature_size = data.shape[1] - 5
    for i in range(start_index, end_index):
        indices = range(i - n_steps, i)
        
        # Features are the first 13 columns and labels are the last 5 columns.
        labels.append(data[indices[0]][feature_size:])
        features.append(np.delete(data[indices], range(feature_size, data.shape[1]), axis=1))

    # Save the data and labels in .npy files.
    np.save("features_" + str(n_steps) + "_steps" + ".npy", np.array(features))
    np.save("labels_" + str(n_steps) + "_steps" + ".npy", np.array(labels))

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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)

    # Split the training data into training and validation sets.
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42, shuffle=True)

    # Scale the data.
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
    x_val = scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape)
    
    # Create the train and test datasets.
    buffer_size = 60_000
    batch_size = 16
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size).batch(batch_size).repeat()
    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).shuffle(buffer_size).batch(batch_size).repeat()

    # Get the timesteps (30), number of features (13), and number of labels (5).
    n_timesteps, n_features = x.shape[1], x.shape[2]
    n_labels = y.shape[1]

    # Create the model.
    model = Sequential()
    model.add(Input(shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=128, kernel_size=3, activation="relu"))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(units=n_labels, activation="relu"))
    model.summary()
    
    # Compile and train the model.
    optimizer = Adam(learning_rate=0.0001)
    es = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])    
    history = model.fit(train_data, 
                        steps_per_epoch=len(x_train) // batch_size, 
                        epochs=10, 
                        validation_data=val_data,
                        validation_steps=len(x_val) // batch_size,
                        callbacks=[es])

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
    plt.title("Training and Testing Loss - MAE and MSE")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.plot(history.history["val_mae"], label="MAE Valid", color="red")
    plt.plot(history.history["mae"], label="MAE Train", color="blue")
    plt.plot(history.history["val_loss"], label="MSE Valid", color="orange")
    plt.plot(history.history["loss"], label="MSE Train", color="green")
    plt.legend()
    plt.show()

    # Save the model.
    model.save("cnn_model_old")

if __name__ == "__main__":
    main()
