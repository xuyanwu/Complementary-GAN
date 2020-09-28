# Complementary-GAN

The code will be updated to the paper version in few days.

To run the code for the MNIST, do the following:

1. cd mnist_cifar
2. modify the run.sh to 'python main.py --dataset MNIST --num_epoches 40 --nc 1 --p1 0.02 --p2 1.0' ### p1: labeled data proportion, p2: proportion of complementary labeled data.###
3. sh run.sh
