#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Sigmoid,
}

impl Activation {
    pub fn activate(&self, inputs: &Vec<f64>, index: usize) -> f64 {
        1.0 / (1. + (-inputs[index]).exp())
    }

    pub fn derivative(&self, inputs: &Vec<f64>, index: usize) -> f64 {
        let a = self.activate(inputs, index);
        a * (1. - a)
    }
}
