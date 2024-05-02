namespace Neural.Layers;

public interface ILayer<in TInput, out TOutput> {
    public TOutput Activation();
    public TOutput ForwardPass(TInput activation);
}

public interface IPerceptronLayer : ILayer<float[], float[]>;

public interface ITrainableLayer : IPerceptronLayer {
    public void UpdateParams(float alpha, float[] delta, float[] input);
}
