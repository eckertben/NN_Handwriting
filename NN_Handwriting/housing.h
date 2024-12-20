#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 3  // Example: 3 input features (e.g., number of bedrooms, size, location)
#define NUM_NEURONS 5 // Number of neurons in the hidden layer
#define OUTPUT_SIZE 1 // Output: predicted house price
#define LEARNING_RATE 0.01 // Constant that represents the step size for gradient descent
#define EPOCHS 1000  // Number of training epochs

// Structure to represent the neural network

typedef struct NeuralNetwork {
    double weights_input_hidden[INPUT_SIZE][NUM_NEURONS];  // Weights between input and hidden layer
    double weights_hidden_output[NUM_NEURONS][OUTPUT_SIZE]; // Weights between hidden and output layer
    double bias_hidden[NUM_NEURONS];  // Bias for hidden layer, essentially acts as y-intercept (data not centered at 0)
    double bias_output[OUTPUT_SIZE];  // Bias for output layer, acts as y-int (for data not centered at 0)
} NeuralNetwork;

/*double relu(double x);

double relu_derivative(double x);*/

double leaky_relu(double x);

double leaky_relu_derivative(double x);

double linear(double x);

double xavier_initializer(int input_size, int output_size);

void initialize_network(NeuralNetwork *nn);

void forward(NeuralNetwork *nn, double *input, double *hidden_output, double *output);

// Takes in the network, the dataset, the array of neurons, the output price, and the 
void backward(NeuralNetwork *nn, double *input, double *hidden_output, double output, double target);

//Takes in the neural network, the given data, the target prices, and the length of the data
// Adjusts the weights of the neurons based on the MSE (Mean Squared Error)
void train(NeuralNetwork *nn, double data[][INPUT_SIZE], double targets[], int num_samples);

double predict(NeuralNetwork *nn, double *input);
