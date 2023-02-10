#include <iostream>

#include "neurolib.h"

int main() {

    int layerSizes[] = {2, 20, 40, 60, 40, 20, 2};
    int numOfLayers = sizeof(layerSizes) / sizeof(int);

    std::cout << "Hello" << std::endl;
    neurolib neuronNet(layerSizes, numOfLayers);

    float inputs[] = {0, 1};
    float* outputs = new float[layerSizes[numOfLayers - 1]];
    outputs[0] = 0.0;
    outputs[1] = 0.0;

    // for (int i = 0; i < 1; i++) {
    //     for (int x = 0; x < 1; x++) {
    //         inputs[0] = 0;
    //         inputs[1] = 1;
    //         neuronNet.trainModel(inputs, 1);
    //         inputs[0] = 1;
    //         inputs[1] = 1;
    //         neuronNet.trainModel(inputs, 1);
    //         inputs[0] = 1;
    //         inputs[1] = 0;
    //         neuronNet.trainModel(inputs, 0);
    //         inputs[0] = 0;
    //         inputs[1] = 0;
    //         neuronNet.trainModel(inputs, 0);
    //     }
    //     neuronNet.applyBatch();
    // }

    inputs[0] = 0.0;
    inputs[1] = 0.0;
    neuronNet.runModel(inputs, outputs);

    printf("Output of raw model in main:\n");
    for (int i = 0; i < layerSizes[numOfLayers - 1]; i++) {
        printf("  %d: %f\n", i, outputs[i]);
    }

    neuronNet.softMax(outputs, 2);

    printf("\nOutput of SoftMax in main:\n");
    for (int i = 0; i < layerSizes[numOfLayers - 1]; i++) {
        printf("  %d: %f\n", i, outputs[i]);
    }

    // printf("\nWeights of second layer:\n");
    // for (int i = 0; i < layerSizes[1]; i++) {
    //     printf("  Neuron %d:\n", i);
    //     for (int x = 0; x < layerSizes[2]; x++) {
    //         printf("    %d : %f\n", x, neuronNet.layers[1].neurons[i].weights[x]);
    //     }
    // }

    // printf("\nBiases of second layer:\n");
    // for (int i = 0; i < layerSizes[1]; i++) {
    //     printf("    %d : %f\n", i, neuronNet.layers[1].neurons[i].bias);
    // }

    float a = 0.2;
    // neuronNet.trainModel(&a);

    delete[] outputs;
}