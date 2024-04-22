from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

xor_data = [
    ([0,0], [0]),
    ([1,0], [1]),
    ([0,1], [1]),
    ([1,1], [0])
]

xor_nn = NeuralNet(2, 2, 1)
xor_nn.train(xor_data, print_interval=100, iters=1000)

print(xor_nn.test_with_expected(xor_data))
print(xor_nn.evaluate([1, 1]))