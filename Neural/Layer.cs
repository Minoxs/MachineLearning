namespace Neural;

// TODO USE IEnumerable
// TODO TURN INTO AN ABSTRACT CLASS
public class Layer {
    private readonly Node[] _nodes;

    public Layer(IEnumerable<Node> nodes) {
        _nodes = nodes.ToArray();
    }

    public float[] Activation => _nodes.Select(node => node.Value()).ToArray();

    public int NodeCount => _nodes.Length;

    public void UpdateParams(float alpha, float[] delta, float[] input) {
        for (var i = 0; i < _nodes.Length; i++) _nodes[i].UpdateParams(alpha, delta[i], input);
    }

    public float[] ForwardPass(float[] activation) {
        return _nodes.Select(node => node.ForwardPass(activation)).ToArray();
    }
}
