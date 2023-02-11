#include <iostream>

#include "neurolib.h"

int main() {
    // Todo: Fix weird output layer return in "runModel"
    // Todo: Fix ugly code

    int layerSizes[] = {2, 3, 2};
    int numOfLayers = sizeof(layerSizes) / sizeof(int);

    std::cout << "Hello" << std::endl;
    neurolib neuronNet(layerSizes, numOfLayers);

    float inputs[] = {0, 1};
    float* outputs = new float[layerSizes[numOfLayers - 1]];
    outputs[0] = 0.0;
    outputs[1] = 0.0;

    for (int i = 0; i < 1000; i++) {

        inputs[0] = 0;
        inputs[1] = 1;
        neuronNet.trainModel(inputs, 1);
        inputs[0] = 1;
        inputs[1] = 1;
        neuronNet.trainModel(inputs, 1);
        inputs[0] = 1;
        inputs[1] = 0;
        neuronNet.trainModel(inputs, 0);
        inputs[0] = 0;
        inputs[1] = 0;
        neuronNet.trainModel(inputs, 0);

        neuronNet.printWeightInfo();

        neuronNet.applyBatch();
    }

    neuronNet.printWeightInfo();

    inputs[0] = 1.0;
    inputs[1] = 1.0;
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

    delete[] outputs;
}