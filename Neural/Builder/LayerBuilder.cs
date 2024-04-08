namespace Neural.Builder;

public class LayerBuilder {
    private readonly Random _rng = new();

    public Layer Build(int size, int weightCount) {
        var nodes = new Node[size];
        for (var i = 0; i < size; i++) nodes[i] = _initNode(_rng, weightCount);

        return new Layer(nodes);
    }

    private static Node _initNode(Random rng, int weightCount) {
        return new Node(
            rng.NextSingle() / 10,
            _getWeights(rng, weightCount)
        );
    }

    private static IEnumerable<float> _getWeights(Random rng, int count) {
        var weights = new float[count];
        for (var i = 0; i < count; i++) weights[i] = rng.NextSingle() / 10;
        return weights;
    }
}
