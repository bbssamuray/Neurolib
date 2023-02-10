#include "neurolib.h"

#include <bits/stdc++.h>
#include <stdio.h>

#include <cmath>

neurolib::neurolib(int layerSizes[], int numOfLayers) {

    this->numOfLayers = numOfLayers;

    trainingSinceLastBatch = 0;

    srand(0);  // Change the seed later

    layers = new layer[numOfLayers];  // Initialize layers
    for (int layerId = 0; layerId < numOfLayers; layerId++) {
        layers[layerId].size = layerSizes[layerId];

        layers[layerId].neurons = new neuron[layerSizes[layerId]];  // Initialize neurons in the said layer
        for (int neuronId = 0; neuronId < layerSizes[layerId]; neuronId++) {

            // randomize biases
            layers[layerId].neurons[neuronId].bias = randF(-1.0, 1.0);
            // Set bias batch to 0
            layers[layerId].neurons[neuronId].biasBatchSum = 0.0;

            if (layerId == numOfLayers - 1) {
                // output layer doesn't need any weights
                continue;
            }

            layers[layerId].neurons[neuronId].weights = new float[layerSizes[layerId + 1]];         // Initialize weights connecting to the next layer
            layers[layerId].neurons[neuronId].weightBatchSum = new float[layerSizes[layerId + 1]];  // Init weight batches

            for (int weightId = 0; weightId < layerSizes[layerId + 1]; weightId++) {
                // Randomize weights
                layers[layerId].neurons[neuronId].weights[weightId] = randF(-1.0, 1.0);
                // Set weight batch to 0
                layers[layerId].neurons[neuronId].weightBatchSum[weightId] = 0.0;
            }
        }
    }
}

neurolib::~neurolib() {
    for (int layerId = 0; layerId < numOfLayers; layerId++) {
        for (int neuronId = 0; neuronId < layers[layerId].size; neuronId++) {
            delete[] layers[layerId].neurons[neuronId].weights;
            delete[] layers[layerId].neurons[neuronId].weightBatchSum;
            // delete[] layers[layerId].neurons[neuronId].biasBatchSum;
        }
        delete[] layers[layerId].neurons;
    }
    delete[] layers;
}

float neurolib::randF(float min, float max) {
    // Gives a random float value between a range
    return (float)(rand()) / (float)(RAND_MAX) * (max - min) + (min);
}

inline float neurolib::actFunc(float x) {
    // Leaky RelU
    if (x > 0) {
        return x;
    } else {
        return x * 0.1;
    }
}

inline float neurolib::actFuncDer(float x) {
    // Derivative of the activation function
    // Leaky RelU derivative
    if (x >= 0) {
        return 1.0;
    } else {
        return 0.1;
    }
}

void neurolib::softMax(float* inputs, int inputSize) {

    if (inputSize <= 0) {
        // Use output layer's size if no size is given
        // Overload this with no int argument?
        inputSize = layers[numOfLayers - 1].size;
    }

    float sum = 0.0;

    for (int i = 0; i < inputSize; i++) {
        sum += expf(inputs[i]);
    }

    for (int i = 0; i < inputSize; i++) {
        inputs[i] = expf(inputs[i]) / sum;
    }
}

void neurolib::runModel(float inputs[], float outputs[]) {

    // Set input layer's values
    for (int inputId = 0; inputId < layers[0].size; inputId++) {
        layers[0].neurons[inputId].value = inputs[inputId];
    }

    // Reset all of other neuron values to their bias
    for (int layerId = 1; layerId < numOfLayers; layerId++) {
        for (int neuronId = 0; neuronId < layers[layerId].size; neuronId++) {
            layers[layerId].neurons[neuronId].value = layers[layerId].neurons[neuronId].bias;
        }
    }

    for (int layerId = 0; layerId < numOfLayers - 1; layerId++) {
        // multiply current neuron's value with the corresponding weight and add it to the neuron that weight is connected to
        for (int neuronId = 0; neuronId < layers[layerId].size; neuronId++) {
            for (int weightId = 0; weightId < layers[layerId + 1].size; weightId++) {
                layers[layerId + 1].neurons[weightId].value += layers[layerId].neurons[neuronId].weights[weightId] * layers[layerId].neurons[neuronId].value;
            }
        }
        // Run every neuron of the next layer through the activation function
        for (int neuronId = 0; neuronId < layers[layerId + 1].size; neuronId++) {
            layers[layerId + 1].neurons[neuronId].value = actFunc(layers[layerId + 1].neurons[neuronId].value);
        }
    }

    // Run last layer's neurons through the activation function
    for (int neuronId = 0; neuronId < layers[numOfLayers - 1].size; neuronId++) {
        layers[numOfLayers - 1].neurons[neuronId].value = actFunc(layers[numOfLayers - 1].neurons[neuronId].value);
    }

    // Return the output layer's values as a dynamic array
    float* outputLayer = new float[layers[numOfLayers - 1].size];

    for (int i = 0; i < layers[numOfLayers - 1].size; i++) {
        outputs[i] = layers[numOfLayers - 1].neurons[i].value;
    }
}

void neurolib::trainModel(float inputs[], int truth) {
    // truth is the ID of output neuron that should be 1.0

    trainingSinceLastBatch++;

    float* smaxResults = new float[layers[numOfLayers - 1].size];

    runModel(inputs, smaxResults);
    softMax(smaxResults, 0);

    // Set up output layer's derivatives and biases
    for (int neuronId = 0; neuronId < layers[numOfLayers - 1].size; neuronId++) {
        if (neuronId == truth) {
            // Need to pass values before the activation function
            // Doesn't really matter for RelU though
            const float derivative = (smaxResults[neuronId] - 1) * actFuncDer(layers[numOfLayers - 1].neurons[neuronId].value);
            layers[numOfLayers - 1].neurons[neuronId].derivative = derivative;
            layers[numOfLayers - 1].neurons[neuronId].biasBatchSum += derivative;
        } else {
            // Same as above
            const float derivative = (smaxResults[neuronId]) * actFuncDer(layers[numOfLayers - 1].neurons[neuronId].value);
            layers[numOfLayers - 1].neurons[neuronId].derivative = derivative;
            layers[numOfLayers - 1].neurons[neuronId].biasBatchSum += derivative;
        }
    }

    for (int layerId = numOfLayers - 2; layerId >= 0; layerId--) {
        layer* currentLayer = &(layers[layerId]);
        layer* nextLayer = &(layers[layerId + 1]);

        for (int neuronId = 0; neuronId < currentLayer->size; neuronId++) {
            neuron* currentNeuron = &(currentLayer->neurons[neuronId]);

            // Calculate the derivative
            float derivative = 0;
            for (int nextNeuronId = 0; nextNeuronId < nextLayer->size; nextNeuronId++) {
                // Sum all of the next layer's neurons multiplied by the corresponding weight
                neuron* nextNeuron = &(nextLayer->neurons[nextNeuronId]);
                derivative += nextNeuron->derivative * currentNeuron->weights[nextNeuronId];
            }
            derivative *= actFuncDer(currentNeuron->value);
            currentNeuron->derivative = derivative;

            // Tweak the bias batch
            currentNeuron->biasBatchSum += derivative;

            // Tweak the weight batch
            for (int weightId = 0; weightId < layers[layerId + 1].size; weightId++) {
                currentNeuron->weightBatchSum[weightId] += derivative * currentNeuron->value;
            }
        }
    }

    delete[] smaxResults;
}

void neurolib::applyBatch() {

    if (trainingSinceLastBatch == 0) {
        return;
    }
    debugCounter++;

    for (int layerId = 0; layerId < numOfLayers; layerId++) {
        layer* currentLayer = &(layers[layerId]);
        const int weightCount = layers[layerId + 1].size;

        for (int neuronId = 0; neuronId < currentLayer->size; neuronId++) {
            neuron* currentNeuron = &(currentLayer->neurons[neuronId]);

            // Apply bias batch
            currentNeuron->bias -= currentNeuron->biasBatchSum / trainingSinceLastBatch * stepSize;

            currentNeuron->biasBatchSum = 0.0;

            // Last layer Doesn't have any weights
            if (layerId == numOfLayers - 1) {
                continue;
            }

            // Apply weight batch
            for (int weightId = 0; weightId < weightCount; weightId++) {
                currentNeuron->weights[weightId] -= currentNeuron->weightBatchSum[weightId] / (float)trainingSinceLastBatch * stepSize;
                currentNeuron->weightBatchSum[weightId] = 0.0;
            }
        }
    }

    trainingSinceLastBatch = 0;
}