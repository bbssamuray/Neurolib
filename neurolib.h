class neurolib {

   public:
    struct neuron {
        float* weights;  // Array of weights connecting to the next layer
        float bias;
        float value;
        float derivative;       // For training only
        float* weightBatchSum;  // For training only, array
        float biasBatchSum;     // For training only
    };

    struct layer {
        int size;         // Number of neurons in this layer
        neuron* neurons;  // Array of neurons
    };

    const float stepSize = 0.1;

    int debugCounter = 0;  // temp
    int numOfLayers;
    int trainingSinceLastBatch;
    layer* layers;  // Array of all of the layers

    neurolib(int layerSizes[], int numOfLayers);
    ~neurolib();
    float randF(float min, float max);
    float actFunc(float x);
    float actFuncDer(float x);
    void softMax(float* inputs, int inputSize);
    void runModel(float inputs[], float outputs[]);
    void trainModel(float* inputs, int truth);
    void applyBatch();
    void printWeightInfo();
};
