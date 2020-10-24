fn main() {

    use rustlearn::prelude::*;
    use rustlearn::datasets::iris;
    use rustlearn::metrics::accuracy_score;    
    use rustlearn::cross_validation::CrossValidation;
    
    // Logistic regression
    use rustlearn::linear_models::sgdclassifier::Hyperparameters;

    let (data, target) = iris::load_data();
    
    let num_splits = 10;
    let num_epochs = 5;
    
    let mut _accuracy = 0.0;
    
    for (train_idx, test_idx) in CrossValidation::new(data.rows(), num_splits) {
    
        let x_train = data.get_rows(&train_idx);
        let y_train = target.get_rows(&train_idx);
        let x_test = data.get_rows(&test_idx);
        let y_test = target.get_rows(&test_idx);
    
        let mut model = Hyperparameters::new(data.cols())
                                        .learning_rate(0.5)
                                        .l2_penalty(0.0)
                                        .l1_penalty(0.0)
                                        .one_vs_rest();
    
        for _ in 0..num_epochs {
            model.fit(&x_train, &y_train).unwrap();
        }
    
        let prediction = model.predict(&x_test).unwrap();
        _accuracy += accuracy_score(&y_test, &prediction);
    }
    
    _accuracy /= num_splits as f32;
    println!("resultado: {}", _accuracy);
}
