using Neural.Layers;

namespace Neural;

public class InputLayer(int size) : IPerceptronLayer {
    private readonly float[] _values = new float[size];

    public float[] Activation() {
        return _values;
    }

    public float[] ForwardPass(float[] activation) {
        return Activation();
    }

    public void SetInput(float[] values) {
        if (values.Length != _values.Length) {
            throw new ArgumentException(
                $"Invalid input size {values.Length} expected {_values.Length}",
                nameof(values)
            );
        }

        for (var i = 0; i < _values.Length; i++) {
            _values[i] = values[i];
        }
    }
}
