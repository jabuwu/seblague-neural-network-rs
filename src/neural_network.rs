use rand::prelude::*;

use crate::{Cost, DataPoint, Layer, LayerLearnData, NetworkLearnData};

pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub layer_sizes: Vec<usize>,
    pub cost: Cost,
    pub rng: Box<dyn RngCore>,
}

impl NeuralNetwork {
    // Create the neural network
    pub fn new(layer_sizes: &Vec<usize>) -> NeuralNetwork {
        let mut rng = Box::new(thread_rng());

        let mut layers = vec![];
        for i in 0..(layer_sizes.len() - 1) {
            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1], rng.as_mut()));
        }

        NeuralNetwork {
            layers,
            layer_sizes: layer_sizes.clone(),
            cost: Cost::MeanSquareError,
            rng,
        }
    }

    // Run the inputs through the network to predict which class they belong to.
    // Also returns the activations from the output layer.
    pub fn classify(&self, inputs: &Vec<f64>) -> (usize, Vec<f64>) {
        let outputs = self.calculate_outputs(inputs);
        let predicted_class = self.max_value_index(&outputs);
        (predicted_class, outputs)
    }

    // Run the inputs through the network to calculate the outputs
    pub fn calculate_outputs(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut inputs = inputs.clone();
        for layer in self.layers.iter() {
            inputs = layer.calculate_outputs(inputs);
        }
        return inputs;
    }

    pub fn learn(
        &mut self,
        training_data: &Vec<DataPoint>,
        learn_rate: f64,
        regularization: f64,
        momentum: f64,
    ) {
        let mut learn_data = NetworkLearnData::new(&self.layers);
        for data in training_data.iter() {
            self.update_gradients(data, &mut learn_data);
        }
        for layer in self.layers.iter_mut() {
            layer.apply_gradients(
                learn_rate / training_data.len() as f64,
                regularization,
                momentum,
            );
        }
    }

    pub fn update_gradients(&mut self, data: &DataPoint, learn_data: &mut NetworkLearnData) {
        // Feed data through the network to calculate outputs.
        // Save all inputs/weightedinputs/activations along the way to use for backpropagation.
        let mut inputs_to_next_layer = data.inputs.clone();

        for i in 0..self.layers.len() {
            inputs_to_next_layer = self.layers[i]
                .calculate_outputs_learn_data(&inputs_to_next_layer, &mut learn_data.layer_data[i]);
        }

        // -- Backpropagation --
        let output_layer_index = self.layers.len() - 1;
        let output_layer = &mut self.layers[output_layer_index];
        let output_learn_data = &mut learn_data.layer_data[output_layer_index];

        // Update output layer gradients
        output_layer.calculate_output_layer_node_values(
            output_learn_data,
            &data.expected_outputs,
            self.cost,
        );
        output_layer.update_gradients(output_learn_data);

        // Update all hidden layer gradients
        let mut i = output_layer_index - 1;
        loop {
            let layer_learn_data = unsafe {
                &mut *(&learn_data.layer_data[i] as *const LayerLearnData as *mut LayerLearnData)
            };
            let hidden_layer = unsafe { &mut *(&self.layers[i] as *const Layer as *mut Layer) };

            hidden_layer.calculate_hidden_layer_node_values(
                layer_learn_data,
                &self.layers[i + 1],
                &learn_data.layer_data[i + 1].node_values,
            );
            hidden_layer.update_gradients(layer_learn_data);
            if i == 0 {
                break;
            } else {
                i -= 1;
            }
        }
    }

    pub fn set_cost_function(&mut self, cost: Cost) {
        self.cost = cost;
    }

    pub fn max_value_index(&self, values: &Vec<f64>) -> usize {
        let mut max_value = -999999.; // TODO
        let mut index = 0;
        for i in 0..values.len() {
            if values[i] > max_value {
                max_value = values[i];
                index = i;
            }
        }
        index
    }

    pub fn cost(&self, data_points: &Vec<DataPoint>) -> f64 {
        let mut cost = 0.;
        for data in data_points.iter() {
            let (_, outputs) = self.classify(&data.inputs);
            cost += self.cost.cost(&outputs, &data.expected_outputs);
        }
        cost /= data_points.len() as f64;
        cost
    }
}
