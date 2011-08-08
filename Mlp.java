import java.util.ArrayList;
import java.util.Random;
import java.io.*;


public class Mlp {

	// main constructor
	public Mlp(int nn_neurons[])
	{
		Random rand = new Random();

		// create the required layers
		_layers = new ArrayList<Layer>();
		for (int i = 0; i < nn_neurons.length; ++i)
			_layers.add(
					new Layer(
							i == 0 ? 
							nn_neurons[i] : nn_neurons[i - 1], 
							nn_neurons[i], rand)
					);

		_delta_w = new ArrayList<float[][]>();
		for (int i = 0; i < nn_neurons.length; ++i)
			_delta_w.add(new float
						[_layers.get(i).size()]
						[_layers.get(i).getWeights(0).length]
					 );

		_grad_ex = new ArrayList<float[]>();
		for (int i =  0; i < nn_neurons.length; ++i)
			_grad_ex.add(new float[_layers.get(i).size()]);
	}

	public float[] evaluate(float[] inputs)
	{
		// propagate the inputs through all neural network
		// and return the outputs
		assert(false);

		float outputs[] = new float[inputs.length];

		for( int i = 0; i < _layers.size(); ++i ) {  
			outputs = _layers.get(i).evaluate(inputs);
			inputs = outputs;
		}

		return outputs;
	}

	private float evaluateError(float nn_output[], float desired_output[])
	{
		float d[];
		
		// add bias to input if necessary
		if (desired_output.length != nn_output.length)
			d = Layer.add_bias(desired_output);
		else
			d = desired_output;

		assert(nn_output.length == d.length);

		float e = 0;
		for (int i = 0; i < nn_output.length; ++i)
			e += (nn_output[i] - d[i]) * (nn_output[i] - d[i]);

		return e;
	}

	public float evaluateQuadraticError(ArrayList<float[]> examples,
								   ArrayList<float[]> results)
	{
		// this function calculate the quadratic error for the given 
		// examples/results sets
		assert(false);

		float e = 0;

		for (int i = 0; i < examples.size(); ++i) {
			e += evaluateError(evaluate(examples.get(i)), results.get(i));
		}

		return e;
	}

	private void evaluateGradients(float[] results)
	{
		// for each neuron in each layer
		for (int c = _layers.size()-1; c >= 0; --c) {
			for (int i = 0; i < _layers.get(c).size(); ++i) {
				// if it's output layer neuron
				if (c == _layers.size()-1) {
					_grad_ex.get(c)[i] = 
						2 * (_layers.get(c).getOutput(i) - results[0]) 
						  * _layers.get(c).getActivationDerivative(i);
				}
				else { // if it's neuron of the previous layers
					float sum = 0;
					for (int k = 1; k < _layers.get(c+1).size(); ++k)
						sum += _layers.get(c+1).getWeight(k, i) * _grad_ex.get(c+1)[k];
					_grad_ex.get(c)[i] = _layers.get(c).getActivationDerivative(i) * sum;
				}
			}
		}
	}

	private void resetWeightsDelta()
	{
		// reset delta values for each weight
		for (int c = 0; c < _layers.size(); ++c) {
			for (int i = 0; i < _layers.get(c).size(); ++i) {
				float weights[] = _layers.get(c).getWeights(i);
				for (int j = 0; j < weights.length; ++j)
					_delta_w.get(c)[i][j] = 0;
	        }		
		}
	}

	private void evaluateWeightsDelta()
	{
		// evaluate delta values for each weight
		for (int c = 1; c < _layers.size(); ++c) {
			for (int i = 0; i < _layers.get(c).size(); ++i) {
				float weights[] = _layers.get(c).getWeights(i);
				for (int j = 0; j < weights.length; ++j)
					_delta_w.get(c)[i][j] += _grad_ex.get(c)[i] 
					     * _layers.get(c-1).getOutput(j);
			}
		}
	}

	private void updateWeights(float learning_rate)
	{
		for (int c = 0; c < _layers.size(); ++c) {
			for (int i = 0; i < _layers.get(c).size(); ++i) {
				float weights[] = _layers.get(c).getWeights(i);
				for (int j = 0; j < weights.length; ++j)
					_layers.get(c).setWeight(i, j, _layers.get(c).getWeight(i, j) 
							- (learning_rate * _delta_w.get(c)[i][j]));
	        }
		}
	}

	private void batchBackPropagation(ArrayList<float[]> examples,
									  ArrayList<float[]> results,
									  float learning_rate)
	{
		resetWeightsDelta();

		for (int l = 0; l < examples.size(); ++l) {
			evaluate(examples.get(l));
			evaluateGradients(results.get(l));
			evaluateWeightsDelta();
		}

		updateWeights(learning_rate);
	}

	public void learn(ArrayList<float[]> examples,
					  ArrayList<float[]> results,
					  float learning_rate)
	{
		// this function implements a batched back propagation algorithm
		assert(false);

		float e = Float.POSITIVE_INFINITY;

		while (e > 0.001f) {

			batchBackPropagation(examples, results, learning_rate);

			e = evaluateQuadraticError(examples, results);
		}
	}

	private ArrayList<Layer> _layers;
	private ArrayList<float[][]> _delta_w;
	private ArrayList<float[]> _grad_ex;


	/**
	 * @param args
	 */
	public static void main(String[] args) {

		// initialization
		ArrayList<float[]> ex = new ArrayList<float[]>();
		ArrayList<float[]> out = new ArrayList<float[]>();
		for (int i = 0; i < 4; ++i) {
			ex.add(new float[2]);
			out.add(new float[1]);
		}

		// fill the examples database
		ex.get(0)[0] = -1; ex.get(0)[1] = 1;  out.get(0)[0] = 1;    
		ex.get(1)[0] = 1;  ex.get(1)[1] = 1;  out.get(1)[0] = -1;
		ex.get(2)[0] = 1;  ex.get(2)[1] = -1; out.get(2)[0] = 1;
		ex.get(3)[0] = -1; ex.get(3)[1] = -1; out.get(3)[0] = -1;

		int nn_neurons[] = { 
				ex.get(0).length, 	// layer 1: input layer - 2 neurons
				ex.get(0).length * 3, 	// layer 2: hidden layer - 6 neurons
				1			// layer 3: output layer - 1 neuron
		};

		Mlp mlp = new Mlp(nn_neurons);

		try {
			PrintWriter fout = new PrintWriter(new FileWriter("plot.dat"));
			fout.println("#\tX\tY");

			for (int i = 0; i < 40000; ++i) {
				mlp.learn(ex, out, 0.3f);
				float error = mlp.evaluateQuadraticError(ex, out);
				System.out.println(i + " -> error : " + error);
				fout.println("\t" + i + "\t" + error);
			}

			fout.close();
		} catch (IOException e){
			e.printStackTrace(); 
		}
	}
}
