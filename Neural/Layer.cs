namespace Neural;

// TODO USE IEnumerable
// TODO CREATE A LAYER BUILDER
// TODO TURN INTO AN ABSTRACT CLASS
public struct Layer {
    private readonly List<Node> _nodes;

    private Node _initNode(int weightCount) {
        var rng = new Random();

        return new Node(
            rng.NextSingle() / 10,
            _getWeights(rng, weightCount)
        );
    }

    private List<float> _getWeights(Random rng, int count) {
        var weights = new List<float>();
        weights.Capacity = count;

        for (var i = 0; i < count; i++) weights.Add(rng.NextSingle() / 10);

        return weights;
    }

    public Layer(int size, int weightCount) {
        _nodes = new List<Node>();

        for (var i = 0; i < size; i++) _nodes.Add(_initNode(weightCount));
    }

    public float[] Activate(float[] activation) {
        return _nodes.Select(node => node.Activate(activation)).ToArray();
    }

    public float[] Activation() {
        return _nodes.Select(node => node.Value()).ToArray();
    }

    public void UpdateParams(float alpha, float[] delta, float[] input) {
        for (var i = 0; i < _nodes.Count; i++) _nodes[i].UpdateParams(alpha, delta[i], input);
    }

    public int NodeCount() {
        return _nodes.Count;
    }
}
