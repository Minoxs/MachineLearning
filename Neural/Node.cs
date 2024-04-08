namespace Neural;

public struct Node {
    private float _value;
    private float _bias;
    private readonly float[] _weights;

    private static float ActivationFunction(float value) {
        return 1 / (1 + MathF.Exp(-value));
    }

    public Node(float bias, IEnumerable<float> weights) {
        _value = 0.0f;
        _bias = bias;
        _weights = weights.ToArray();
    }

    public float ForwardPass(float[] activation) {
        // INPUT NODE RECEIVES NO ACTIVATION
        // RETURN VALUE DIRECTLY
        // TODO THIS WOULD PROBABLY LOOK BETTER IN A CLASS BASED SYSTEM
        if (activation.Length == 0) return _value;

        var z = 0.0f;
        for (var i = 0; i < activation.Length; i++) z += activation[i] * _weights[i];

        _value = ActivationFunction(z + _bias);
        return _value;
    }

    public float Value() {
        return _value;
    }

    public void UpdateParams(float alpha, float delta, float[] input) {
        for (var i = 0; i < _weights.Length; i++) _weights[i] -= alpha * delta * input[i];
        _bias -= alpha * delta;
    }
}
