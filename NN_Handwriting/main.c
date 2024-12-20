#include "housing.h"

int main() {
    // Example data: [bedrooms, size, location] and house prices

    double data[7][INPUT_SIZE] = {
        {3, 1.5, 1},  // Example: 3 bedrooms, 1500 sq. ft, location code 1
        {2, 0.8, 2},
        {4, 2.0, 1},
        {3, 1.2, 3},
        {2, 0.6, 3},
        {5, 2.4, 2},
        {2, 1.0, 1}
    };
    double targets[7] = {3.9, 2.0, 5.4, 2.8, 1.8, 6.4, 2.8}; // Target: House prices

    // Initialize neural network
    NeuralNetwork nn;
    initialize_network(&nn);

    // Train the network
    train(&nn, data, targets, 7);

    // Predict a new house price
    double new_input[INPUT_SIZE] = {3, 1.5, 3};
    double predicted_price = predict(&nn, new_input);
    printf("Predicted House Price: $%.2fK\n", predicted_price * 100);

    double new_input2[INPUT_SIZE] = {2, 0.8, 2};
    predicted_price = predict(&nn, new_input2);
    printf("Predicted House Price: $%.2fK\n", predicted_price * 100);

    double new_input3[INPUT_SIZE] = {4, 0.8, 1};
    predicted_price = predict(&nn, new_input3);
    printf("Predicted House Price: $%.2fK\n", predicted_price * 100);

    double new_input4[INPUT_SIZE] = {20, 5.0, 3};
    predicted_price = predict(&nn, new_input4);
    printf("Predicted House Price: $%.2fK\n", predicted_price * 100);

    double new_input5[INPUT_SIZE] = {1, 5.0, 2};
    predicted_price = predict(&nn, new_input5);
    printf("Predicted House Price: $%.2fK\n", predicted_price * 100);

    return 0;
}