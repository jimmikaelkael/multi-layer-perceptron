import java.util.ArrayList;


public class Layer {

	// main constructor
	public Layer(int prev_n_neurons, int n_neurons, java.util.Random rand)
	{
		// all the layers/neurons must use the same random number generator
		_n_neurons = n_neurons + 1;
		_prev_n_neurons = prev_n_neurons + 1;

		// allocate everything
		_neurons = new ArrayList<Neuron>();
		_outputs = new float[_n_neurons];

		for (int i = 0; i < _n_neurons; ++i)
			_neurons.add(new Neuron(_prev_n_neurons, rand));
	}

	// add 1 in front of the out vector
	public static float[] add_bias(float[] in)
	{
		float out[] = new float[in.length + 1];
		for (int i = 0; i < in.length; ++i)
			out[i + 1] = in[i];
		out[0] = 1.0f;
		return out;
	}

	// compute the output of the layer
	public float[] evaluate(float in[])
	{
		float inputs[];

		// add an input (bias) if necessary
		if (in.length != getWeights(0).length)
			inputs = add_bias(in);
		else
			inputs = in;

		assert(getWeights(0).length == inputs.length);

		// stimulate each neuron of the layer and get its output
		for (int i = 1; i < _n_neurons; ++i)
			_outputs[i] = _neurons.get(i).activate(inputs);

		// bias treatment
		_outputs[0] = 1.0f;

		return _outputs;
	}

	public int size() { return _n_neurons; }
	public float getOutput(int i) { return _outputs[i]; }
	public float getActivationDerivative(int i) { return _neurons.get(i).getActivationDerivative(); }
	public float[] getWeights(int i) { return _neurons.get(i).getSynapticWeights(); }
	public float getWeight(int i, int j) { return _neurons.get(i).getSynapticWeight(j); }
	public void setWeight(int i, int j, float v) { _neurons.get(i).setSynapticWeight(j, v); }

	// --------
	private int _n_neurons, _prev_n_neurons;
	private ArrayList<Neuron> _neurons;
	private float _outputs[];
}
