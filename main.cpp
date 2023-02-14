#include "neurolib.h"

int main() {

    int layerSizes[] = {2, 3, 2};
    int numOfLayers = sizeof(layerSizes) / sizeof(int);
    float* outputs = new float[layerSizes[numOfLayers - 1]];

    printf("Hello!");
    float inputs[] = {0, 1};

    // neurolib neuronNet("testModel.o"); //Models can be loaded like this

    neurolib neuronNet(layerSizes, numOfLayers);

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

        // neuronNet.printWeightInfo();

        neuronNet.applyBatch();
    }

    neuronNet.printWeightInfo();
    neuronNet.saveModel("testModel.o");

    inputs[0] = 1.0;
    inputs[1] = 0.0;
    neuronNet.runModel(inputs, outputs);

    printf("Output of raw model:\n");
    for (int i = 0; i < layerSizes[numOfLayers - 1]; i++) {
        printf("  %d: %f\n", i, outputs[i]);
    }

    neuronNet.softMax(outputs, 2);

    printf("\nOutput of SoftMax:\n");
    for (int i = 0; i < layerSizes[numOfLayers - 1]; i++) {
        printf("  %d: %f\n", i, outputs[i]);
    }

    delete[] outputs;
}