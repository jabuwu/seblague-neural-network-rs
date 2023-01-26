#[derive(Debug, Clone)]
pub struct DataPoint {
    pub inputs: Vec<f64>,
    pub expected_outputs: Vec<f64>,
}
