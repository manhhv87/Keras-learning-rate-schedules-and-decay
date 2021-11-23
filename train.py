# importing the necessary packages
from pyimagesearch.learning_rate_schedulers import StepDecay
from pyimagesearch.learning_rate_schedulers import PolynomialDecay
from pyimagesearch.resnet import ResNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

# setting the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# constructing the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--schedule", type=str, default="",
                help="learning rate schedule method (standard, step, linear, poly)")
ap.add_argument("-e", "--epochs", type=int, default=100,
                help="# of epochs to train for")
ap.add_argument("-l", "--lr-plot", type=str, default="lr.png",
                help="path to output learning rate plot")
ap.add_argument("-t", "--train-plot", type=str, default="training.png",
                help="path to output training plot")
args = vars(ap.parse_args())

# storing number of epochs to train for in a convenience variable,
# then initialize the list of callbacks and learning rate scheduler
# to be used
epochs = args["epochs"]
callbacks = []
schedule = None

# check to see if step-based learning rate decay should be used
if args["schedule"] == "step":
    print("[INFO] using 'step-based' learning rate decay...")
    schedule = StepDecay(initAlpha=1e-1, factor=0.25, dropEvery=15)

# checking to see if linear learning rate decay should be used
elif args["schedule"] == "linear":
    print("[INFO] using 'linear' learning rate decay...")
    schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=1e-1, power=1)

# check to see if a polynomial learning rate decay should be used
elif args["schedule"] == "poly":
    print("[INFO] using 'polynomial' learning rate decay...")
    schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=1e-1, power=5)

# if the learning rate schedule is not empty, add it to the list of
# callbacks
if schedule is not None:
    callbacks = [LearningRateScheduler(schedule)]

# loading the training and testing data, then scaling it into
# range [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# converting the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initializing the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

# initializing the decay for the optimizer
decay = 0.0

# When using Keras' "standard" decay, then we need to set the
# decay parameter
if args["schedule"] == "standard":
    print("[INFO] using 'keras standard' learning rate decay...")
    decay = 1e-1 / epochs

# otherwise, no learning rate schedule is being used
elif schedule is None:
    print("[INFO] no learning rate schedule being used")

# initialize our optimizer and model, then compile it
opt = SGD(lr=1e-1, momentum=0.9, decay=decay)
model = ResNet.build(32, 32, 3, 10, (9, 9, 9),
                     (64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=128, epochs=epochs, callbacks=callbacks, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=labelNames))

# plotting the training loss and accuracy
N = np.arange(0, args["epochs"])
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["train_plot"])

# if the learning rate schedule is not empty, then save the learning
# rate plot
if schedule is not None:
    schedule.plot(N)
    plt.savefig(args["lr_plot"])
