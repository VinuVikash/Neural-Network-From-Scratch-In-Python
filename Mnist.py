import mnist_loader,network
import matplotlib.pyplot as plt
training_data, validation_data, test_data =mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
test_data=list(test_data)
training_data=list(training_data)
#plt.imshow(training_data[0][0].reshape(28,28),cmap='gray')
#plt.show()
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
