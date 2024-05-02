using Neural.Builder;
using Neural.Layers;

namespace Neural;

// TODO CREATE A TRAINER CLASS
public class Network {
    private readonly IPerceptronLayer[] _layers;

    public Network(int inputSize, int outputSize, params int[] hiddenLayer) {
        var builder = new LayerBuilder();
        var layers = new List<IPerceptronLayer> {
            new InputLayer(inputSize)
        };

        var size = inputSize;
        foreach (var layer in hiddenLayer) {
            layers.Add(builder.Build(layer, size));
            size = layer;
        }

        layers.Add(builder.Build(outputSize, size));

        _layers = layers.ToArray();
    }

    public float[] Output => _layers.Last().Activation();

    public static float LearningRate => 0.2f;

    public float[] Cost(float[] expected) {
        return Output.Select((x, i) => x - expected[i]).ToArray();
    }

    public float[] ForwardPass(float[] input) {
        (_layers.First() as InputLayer)?.SetInput(input);
        return _layers.Aggregate(Array.Empty<float>(), (a, layer) => layer.ForwardPass(a)).ToArray();
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

        for (var i = _layers.Length - 2; i >= 1; i--) {
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
        for (var i = 1; i < _layers.Length; i++)
            (_layers[i] as ITrainableLayer)?.UpdateParams(LearningRate, deltaLayers[i - 1], _layers[i - 1].Activation());
    }

    public void Train(float[] input, float[] expectedOutput) {
        ForwardPass(input);
        var deltaLayers = _backPropagation(expectedOutput);
        _updateParams(deltaLayers);
    }
}
