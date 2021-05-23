class NeuralNet{
  Matrix weights_ih, weights_ho, bias_h, bias_o;
  double l_rate = 0.01d;
  Matrix ref;
  
  NeuralNet(int i, int h, int o){
    weights_ih = new Matrix(h, i);
    weights_ho = new Matrix(o, h);
    
    bias_h = new Matrix(h, 1);
    bias_o = new Matrix(o, 1);
    
    ref = new Matrix(1, 1);
  }
  
  ArrayList<Double> predict(double[] X){
    Matrix input = ref.fromArray(X);
    Matrix hidden = ref.multiply(weights_ih, input);
    hidden.add(bias_h);
    hidden.sigmoid();
    
    Matrix output = ref.multiply(weights_ho, hidden);
    output.add(bias_o);
    output.sigmoid();
    
    return output.toArray();
  }
  
  void train(double[] X, double[] Y){
    Matrix input = ref.fromArray(X);
    Matrix hidden = ref.multiply(weights_ih, input);
    hidden.add(bias_h);
    hidden.sigmoid();
    
    Matrix output = ref.multiply(weights_ho, hidden);
    output.add(bias_o);
    output.sigmoid();
    
    Matrix target = ref.fromArray(Y);
    Matrix error = ref.sub(target, output);
    Matrix gradient = output.dsigmoid();
    gradient.multiply(error); // error
    gradient.multiply(l_rate);
    
    Matrix hidden_T = hidden.transpose();
    Matrix who_delta = ref.multiply(gradient, hidden_T);
    
    weights_ho.add(who_delta);
    bias_o.add(gradient);
    Matrix who_T = weights_ho.transpose();
    Matrix hidden_errors = ref.multiply(who_T, error);
    
    Matrix h_gradient = hidden.dsigmoid();
    h_gradient.multiply(hidden_errors);
    h_gradient.multiply(l_rate);
    
    Matrix i_T = input.transpose();
    Matrix wih_delta = ref.multiply(h_gradient, i_T);
    
    weights_ih.add(wih_delta);
    bias_h.add(h_gradient);
  }
}
