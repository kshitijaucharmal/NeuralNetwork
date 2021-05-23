class Matrix {
  double[][] _matrix;
  int rows, cols;

  Matrix(int r, int c) {
    rows = r;
    cols = c;
    _matrix = new double[r][c];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        _matrix[i][j] = random(-1f, 1f);
      }
    }
  }
  
  void add(Matrix other){
    for(int i = 0; i < rows; i++){
      for(int j = 0; j < cols; j++){
        _matrix[i][j] += other._matrix[i][j];
      }
    }
  }
  
  void add(double scalar){
    for(int i = 0; i < rows; i++){
      for(int j = 0; j < cols; j++){
        _matrix[i][j] *= scalar;
      }
    }
  }
  
  Matrix sub(Matrix a, Matrix b){
    Matrix sum = new Matrix(a.rows, a.cols);
    for(int i = 0; i < rows; i++){
      for(int j = 0; j < cols; j++){
        sum._matrix[i][j] = a._matrix[i][j] - b._matrix[i][j];
      }
    }
    return sum;
  }
  
  Matrix transpose(){
    Matrix temp = new Matrix(cols, rows);
    for(int i = 0; i < rows; i++){
      for(int j = 0; j < cols; j++){
        temp._matrix[j][i] = _matrix[i][j];
      }
    }
    return temp;
  }
  
  Matrix multiply(Matrix a, Matrix b){
    Matrix temp = new Matrix(a.rows, b.cols);
    for(int i = 0; i < temp.rows; i++){
      for(int j = 0; j < temp.cols; j++){
        double sum = 0;
        for(int k = 0; k < a.cols; k++){
          sum += a._matrix[i][k] * b._matrix[k][j];
        }
        temp._matrix[i][j] = sum;
      }
    }
    return temp;
  }
  
  void multiply(Matrix a){
    for(int i = 0; i < a.rows; i++){
      for(int j = 0; j < a.cols; j++){
        _matrix[i][j] *= a._matrix[i][j];
      }
    }
  }
  
  void multiply(double a){
    for(int i = 0; i < rows; i++){
      for(int j = 0; j < cols; j++){
        _matrix[i][j] *= a;
      }
    }
  }
  
  void sigmoid(){
    for(int i = 0; i < rows; i++){
      for(int j = 0; j < cols; j++){
        _matrix[i][j] = 1 / (1 + Math.exp(-_matrix[i][j]));
      }
    }
  }
  
  Matrix dsigmoid(){
    Matrix temp = new Matrix(rows, cols);
    for(int i = 0; i < rows; i++){
      for(int j = 0; j < cols; j++){
        temp._matrix[i][j] = _matrix[i][j] * (1f - _matrix[i][j]);
      }
    }
    return temp;
  }
  
  Matrix fromArray(double[] x){
    Matrix temp = new Matrix(x.length, 1);
    for(int i = 0; i < x.length; i++) temp._matrix[i][0] = x[i];
    return temp;
  }
  
  ArrayList<Double> toArray(){
    ArrayList<Double> temp = new ArrayList<Double>();
    
    for(int i = 0; i < rows; i++){
      for(int j = 0; j < cols; j++){
        temp.add(_matrix[i][j]);
      }
    }
    
    return temp;
  }
}
