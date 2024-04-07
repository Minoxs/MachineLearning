namespace Neural;

// TODO CREATE A TRAINER CLASS
public class Network {
    private readonly List<Layer> _layers;

    public Network(int inputSize, int outputSize, params int[] hiddenLayer) {
        _layers = new List<Layer> { new(inputSize, 0) };

        foreach (var layerSize in hiddenLayer) _layers.Add(new Layer(layerSize, _layers.Last().NodeCount()));

        _layers.Add(new Layer(outputSize, _layers.Last().NodeCount()));
    }

    public float[] Output => _layers.Last().Activation();

    public static float LearningRate => 0.2f;

    public float[] Cost(float[] expected) {
        return Output.Select((x, i) => x - expected[i]).ToArray();
    }

    // TODO SHOULD RECEIVE INPUT IF IT'S PUBLIC
    // TODO RETURNING ARRAY MIGHT BE DUMB ?
    public float[] ForwardPass() {
        return _layers.Aggregate(Array.Empty<float>(), (a, layer) => layer.Activate(a)).ToArray();
    }

    // TODO SHOULD PROBABLY BE A METHOD OF NODE
    private static float _activationRate(float a) {
        return a * (1 - a);
    }

    private float[][] _backPropagation(float[] expected) {
        var deltas = new List<float[]> {
            // Delta of the last layer is the cost function
            Cost(expected)
        };

        for (var i = _layers.Count - 2; i >= 1; i--) {
            // Error contributed to the next layer in the network
            var contributedError = deltas.Last().Sum(err => err * LearningRate);

            // Deltas of this layer
            var layerDelta = _layers[i].Activation().Select(a => contributedError * _activationRate(a)).ToArray();
            deltas.Add(layerDelta);
        }

        deltas.Reverse();
        return deltas.ToArray();
    }

    private void _updateParams(float[][] deltaLayers) {
        for (var i = 1; i < _layers.Count; i++)
            _layers[i].UpdateParams(LearningRate, deltaLayers[i - 1], _layers[i - 1].Activation());
    }

    public void Train(float[] input, float[] expectedOutput) {
        // TODO CREATE A WAY TO SET INPUT
        // _inputLayer.Set(input);
        ForwardPass();
        var deltaLayers = _backPropagation(expectedOutput);
        _updateParams(deltaLayers);
    }
}
