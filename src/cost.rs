#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cost {
    MeanSquareError,
}

impl Cost {
    pub fn cost(&self, predicted_outputs: &Vec<f64>, expected_outputs: &Vec<f64>) -> f64 {
        let mut cost = 0.;
        for i in 0..predicted_outputs.len() {
            let error = predicted_outputs[i] - expected_outputs[i];
            cost += error * error;
        }
        return 0.5 * cost;
    }

    pub fn derivative(&self, predicted_output: f64, expected_output: f64) -> f64 {
        predicted_output - expected_output
    }
}
