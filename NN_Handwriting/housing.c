#include "housing.h"

// ReLU activation function
/*double relu(double x) {
    // Graph with y=0 when x<0, y=x when x>0
    return fmax(0, x);
}

// Derivative of ReLU (for backpropagation)
double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
    : Can also be written as :
    if (x > 0) {
        return 1;
    }
    return 0;:
}*/

double leaky_relu(double x) {
    return (x > 0) ? x : 0.01 * x;  // small slope for negative inputs
}

double leaky_relu_derivative(double x) {
    return (x > 0) ? 1.0 : 0.01;  // small slope for negative inputs
}

// Linear activation function (for output layer)
double linear(double x) {
    return x; // Linear activation for regression
}

double xavier_initializer(int input_size, int output_size) {
    double range = sqrt(2.0 / (input_size + output_size));  // Xavier range
    return ((rand() / (double)RAND_MAX) * 2 - 1) * range;  // Random value in range [-range, range]
}

// Initialize the network with random weights and biases
void initialize_network(NeuralNetwork *nn) {
    for (int i = 0; i < NUM_NEURONS; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            nn->weights_input_hidden[j][i] = xavier_initializer(INPUT_SIZE, NUM_NEURONS);
        }
    }

    for (int i = 0; i < NUM_NEURONS; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            nn->weights_hidden_output[i][j] = xavier_initializer(OUTPUT_SIZE, NUM_NEURONS);
            nn->bias_output[j] = 0.0; // Initial bias for output layer
        }
        nn->bias_hidden[i] = 0.0;  // Initial bias for hidden layer
    } // Creates the network leading from the neurons to the output
}

// Forward pass to predict house price, essentially just input->neurons->output
void forward(NeuralNetwork *nn, double *input, double *hidden_output, double *output) {
    // Calculate neuron layer
    for (int i = 0; i < NUM_NEURONS; i++) {
        hidden_output[i] = nn->bias_hidden[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden_output[i] += input[j] * nn->weights_input_hidden[j][i]; // w1*d1 + w2*d2 +w3*d3 + bias (input -> neurons)
        }
        hidden_output[i] = leaky_relu(hidden_output[i]); // Apply ReLU activation (negative values at neurons are set to zero)
    }

    // Calculate output layer
    *output = nn->bias_output[0]; // Since we have only 1 output
    for (int i = 0; i < NUM_NEURONS; i++) {
        *output += hidden_output[i] * nn->weights_hidden_output[i][0]; // Essentially adding the dot product (matrix has a height of 1)
    }
}

void backward(NeuralNetwork *nn, double *input, double *hidden_output, double output, double target) {
    // Compute the error at the output layer (MSE derivative)
    double output_error = target - output;
    double output_gradient = output_error;  // MSE gradient
    output_gradient = fmax(fmin(output_gradient, 1), -1);

    // Update weights and bias for output layer
    for (int i = 0; i < NUM_NEURONS; i++) {
        nn->weights_hidden_output[i][0] += LEARNING_RATE * output_gradient * hidden_output[i];  // Update weights
    }
    nn->bias_output[0] += LEARNING_RATE * output_gradient;  // Update bias

    // Propagate error back to the hidden layer
    double neuron_error[NUM_NEURONS];
    for (int i = 0; i < NUM_NEURONS; i++) {
        neuron_error[i] = output_gradient * nn->weights_hidden_output[i][0];
    }

    // Gradient for the hidden layer
    for (int i = 0; i < NUM_NEURONS; i++) {
        // Essentially undoes the operation done in the forward direction
        double hidden_gradient = neuron_error[i] * leaky_relu_derivative(hidden_output[i]);  // Use Leaky ReLU derivative
        hidden_gradient = fmax(fmin(hidden_gradient, 1), -1);

        // Update weights and biases for hidden layer
        for (int j = 0; j < INPUT_SIZE; j++) {
            nn->weights_input_hidden[j][i] += LEARNING_RATE * hidden_gradient * input[j];  // Update weights
        }
        nn->bias_hidden[i] += LEARNING_RATE * hidden_gradient;  // Update bias
    }
}

// Training the neural network
void train(NeuralNetwork *nn, double data[][INPUT_SIZE], double targets[], int num_samples) {
    double neuron_val[NUM_NEURONS]; // Array of the neurons (in this there are 5)
    double output; // Single variable representing the outputted price

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        //double total_loss = 0.0;

        for (int i = 0; i < num_samples; i++) {
            forward(nn, data[i], neuron_val, &output); // Changes the output to the current predicted price for the data given
            //total_loss += (targets[i] - output)*(targets[i] - output);  // Calculates the error for this epoch using Mean Squared (MSE)
            backward(nn, data[i], neuron_val, output, targets[i]); // Passes the data of i, neurons, output, and the target price of i
        }
    }
}

// Use the trained network to make predictions
double predict(NeuralNetwork *nn, double *input) {
    double hidden_output[NUM_NEURONS];
    double output;
    forward(nn, input, hidden_output, &output);
    return output;
}