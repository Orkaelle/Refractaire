import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import mlflow

car = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']


def load_az_dataset(datasetPath):

	# initialize the list of data and labels
	cdata = []
	clabels = []

	# loop over the rows of the A-Z handwritten digit dataset
	for row in open(datasetPath):

		# parse the label and image from the row
		row = row.split(",")
		label = int(row[0])
		image = np.array([int(x) for x in row[1:]], dtype="uint8")

		# update the list of data and labels
		cdata.append(image)
		clabels.append(label)

	# convert the data and labels to NumPy arrays
	cdata = np.array(cdata, dtype="float32")
	clabels = np.array(clabels, dtype="int")

	# Change depth of image to 1
	cdata = cdata.reshape(cdata.shape[0], 28, 28, 1)

	# return a 2-tuple of the A-Z data and labels
	return (cdata, clabels)


def train_model (datasetPath, modelName, conv1_layers, conv1_kernelsize, conv1_activation, conv2_layers, conv2_kernelsize, conv2_activation, dropout1, dense_units, dense_activation, dropout2):

	np.random.seed(40)

	# Split to train and test
	cdata, clabels = load_az_dataset(datasetPath)

	x = cdata
	y = clabels
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=5)

	# Change type from int to float and normalize to [0, 1]
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	# Optionally check the number of samples
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# Convert class vectors to binary class matrices (transform the problem to multi-class classification)
	num_classes = len(car)
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	with mlflow.start_run():
		# Create a neural network with 2 convolutional layers and 2 dense layers
		model = Sequential()
		model.add(Convolution2D(conv1_layers, conv1_kernelsize, conv1_kernelsize, activation=conv1_activation, input_shape=(28,28,1)))
		model.add(Convolution2D(conv2_layers, conv2_kernelsize, conv2_kernelsize, activation=conv2_activation))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(dropout1))
		
		model.add(Flatten())
		model.add(Dense(dense_units, activation=dense_activation))
		model.add(Dropout(dropout2))
		model.add(Dense(num_classes, activation='softmax'))

		model.summary()
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		# Train the model
		model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(x_test, y_test))

		# Get loss and accuracy on validation set
		score = model.evaluate(x_test, y_test, verbose=0)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])

		# Save the data in MLFlow
		mlflow.log_param("Conv1_layers", conv1_layers)
		mlflow.log_param("Conv1_kernelsize", conv1_kernelsize)
		mlflow.log_param("Conv1_activation", conv1_activation)
		mlflow.log_param("Conv2_layers", conv2_layers)
		mlflow.log_param("Conv2_kernelsize", conv2_kernelsize)
		mlflow.log_param("Conv2_activation", conv2_activation)
		mlflow.log_param("Dropout1", dropout1)
		mlflow.log_param("Dense_units", dense_units)
		mlflow.log_param("Dense_activation", dense_activation)
		mlflow.log_param("Dropout2", dropout2)

		mlflow.log_metric("TestLoss", score[0])
		mlflow.log_metric("TestAccuracy", score[1])

		mlflow.keras.log_model(model, modelName)

	return score