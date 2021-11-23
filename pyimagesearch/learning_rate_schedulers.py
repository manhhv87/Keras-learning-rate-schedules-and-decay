# importing the necessary packages
import matplotlib.pyplot as plt
import numpy as np

class LearningRateDecay:
	def plot(self, epochs, title="Learning Rate Schedule"):
		# computing the set of learning rates for each corresponding
		# epoch
		lrs = [self(i) for i in epochs]

		# the learning rate schedule
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(epochs, lrs)
		plt.title(title)
		plt.xlabel("Epoch #")
		plt.ylabel("Learning Rate")

class StepDecay(LearningRateDecay):
	def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
		# storing the base initial learning rate, drop factor, and epochs to drop every
		self.initAlpha = initAlpha
		self.factor = factor
		self.dropEvery = dropEvery

	def __call__(self, epoch):
		# computing the learning rate for the current epoch
		exp = np.floor((1 + epoch) / self.dropEvery)
		alpha = self.initAlpha * (self.factor ** exp)

		# returning the learning rate
		return float(alpha)

class PolynomialDecay(LearningRateDecay):
	def __init__(self, maxEpochs=100, initAlpha=0.01, power=1.0):
		# storing the maximum number of epochs, base learning rate, and power of the polynomial
		self.maxEpochs = maxEpochs
		self.initAlpha = initAlpha
		self.power = power

	def __call__(self, epoch):
		# computing the new learning rate based on polynomial decay
		decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
		alpha = self.initAlpha * decay

		# returning the new learning rate
		return float(alpha)

if __name__ == "__main__":
	# plotting a step-based decay which drops by a factor of 0.5 every 25 epochs
	sd = StepDecay(initAlpha=0.01, factor=0.5, dropEvery=25)
	sd.plot(range(0, 100), title="Step-based Decay")
	plt.show()

	# plotting a linear decay by using a power of 1
	pd = PolynomialDecay(power=1)
	pd.plot(range(0, 100), title="Linear Decay (p=1)")
	plt.show()

	# showing a polynomial decay with a steeper drop by increasing the power value
	pd = PolynomialDecay(power=5)
	pd.plot(range(0, 100), title="Polynomial Decay (p=5)")
	plt.show()
