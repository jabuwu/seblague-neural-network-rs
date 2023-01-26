use std::f64::consts::{E, PI};

use rand::prelude::*;

use crate::{Activation, Cost, LayerLearnData};

pub struct Layer {
    pub num_nodes_in: usize,
    pub num_nodes_out: usize,
    pub weights: Vec<f64>,
    pub biases: Vec<f64>,
    pub cost_gradient_w: Vec<f64>,
    pub cost_gradient_b: Vec<f64>,
    pub weight_velocities: Vec<f64>,
    pub bias_velocities: Vec<f64>,
    pub activation: Activation,
}

impl Layer {
    // Create the layer
    pub fn new(num_nodes_in: usize, num_nodes_out: usize, rng: &mut dyn RngCore) -> Layer {
        let weights = vec![0.; num_nodes_in * num_nodes_out];
        let cost_gradient_w = vec![0.; weights.len()];
        let biases = vec![0.; num_nodes_out];
        let cost_gradient_b = vec![0.; biases.len()];
        let weight_velocities = vec![0.; weights.len()];
        let bias_velocities = vec![0.; biases.len()];
        let activation = Activation::Sigmoid;
        let mut layer = Layer {
            num_nodes_in,
            num_nodes_out,
            weights,
            biases,
            cost_gradient_w,
            cost_gradient_b,
            weight_velocities,
            bias_velocities,
            activation,
        };
        layer.initialize_random_weights(rng);
        layer
    }

    // Calculate layer output activations
    pub fn calculate_outputs(&self, inputs: Vec<f64>) -> Vec<f64> {
        let mut weighted_inputs = vec![0.; self.num_nodes_out];
        for node_out in 0..self.num_nodes_out {
            let mut weighted_input = self.biases[node_out];
            for node_in in 0..self.num_nodes_in {
                weighted_input += inputs[node_in] * self.get_weight(node_in, node_out);
            }
            weighted_inputs[node_out] = weighted_input;
        }

        // Apply activation function
        let mut activations = vec![0.; self.num_nodes_out];
        for output_node in 0..self.num_nodes_out {
            activations[output_node] = self.activation.activate(&weighted_inputs, output_node);
        }

        return activations;
    }

    // Calculate layer output activations and store inputs/weightedInputs/activations in the given learnData object
    pub fn calculate_outputs_learn_data(
        &self,
        inputs: &Vec<f64>,
        learn_data: &mut LayerLearnData,
    ) -> Vec<f64> {
        learn_data.inputs = inputs.clone();

        for node_out in 0..self.num_nodes_out {
            let mut weighted_input = self.biases[node_out];
            for node_in in 0..self.num_nodes_in {
                weighted_input += inputs[node_in] * self.get_weight(node_in, node_out);
            }
            learn_data.weighted_inputs[node_out] = weighted_input;
        }

        // Apply activation function
        for i in 0..learn_data.activations.len() {
            learn_data.activations[i] = self.activation.activate(&learn_data.weighted_inputs, i);
        }

        learn_data.activations.clone()
    }

    // Update weights and biases based on previously calculated gradients.
    // Also resets the gradients to zero.
    pub fn apply_gradients(&mut self, learn_rate: f64, regularization: f64, momentum: f64) {
        let weight_decay = 1. - regularization * learn_rate;

        for i in 0..self.weights.len() {
            let weight = self.weights[i];
            let velocity =
                self.weight_velocities[i] * momentum - self.cost_gradient_w[i] * learn_rate;
            self.weight_velocities[i] = velocity;
            self.weights[i] = weight * weight_decay + velocity;
            self.cost_gradient_w[i] = 0.;
        }

        for i in 0..self.biases.len() {
            let velocity =
                self.bias_velocities[i] * momentum - self.cost_gradient_b[i] * learn_rate;
            self.bias_velocities[i] = velocity;
            self.biases[i] += velocity;
            self.cost_gradient_b[i] = 0.;
        }
    }

    // Calculate the "node values" for the output layer. This is an array containing for each node:
    // the partial derivative of the cost with respect to the weighted input
    pub fn calculate_output_layer_node_values(
        &self,
        layer_learn_data: &mut LayerLearnData,
        expected_outputs: &Vec<f64>,
        cost: Cost,
    ) {
        for i in 0..layer_learn_data.node_values.len() {
            // Evaluate partial derivatives for current node: cost/activation & activation/weightedInput
            let cost_derivative =
                cost.derivative(layer_learn_data.activations[i], expected_outputs[i]);
            let activation_derivative = self
                .activation
                .derivative(&layer_learn_data.weighted_inputs, i);
            layer_learn_data.node_values[i] = cost_derivative * activation_derivative;
        }
    }

    // Calculate the "node values" for a hidden layer. This is an array containing for each node:
    // the partial derivative of the cost with respect to the weighted input
    pub fn calculate_hidden_layer_node_values(
        &self,
        layer_learn_data: &mut LayerLearnData,
        old_layer: &Layer,
        old_node_values: &Vec<f64>,
    ) {
        for new_node_index in 0..self.num_nodes_out {
            let mut new_node_value = 0.;
            for old_node_index in 0..old_node_values.len() {
                // Partial derivative of the weighted input with respect to the input
                let weighted_input_derivative =
                    old_layer.get_weight(new_node_index, old_node_index);
                new_node_value += weighted_input_derivative * old_node_values[old_node_index];
            }
            new_node_value *= self
                .activation
                .derivative(&layer_learn_data.weighted_inputs, new_node_index);
            layer_learn_data.node_values[new_node_index] = new_node_value;
        }
    }

    pub fn update_gradients(&mut self, layer_learn_data: &LayerLearnData) {
        // Update cost gradient with respect to weights
        for node_out in 0..self.num_nodes_out {
            let node_value = layer_learn_data.node_values[node_out];
            for node_in in 0..self.num_nodes_in {
                // Evaluate the partial derivative: cost / weight of current connection
                let derivative_cost_wrt_weight = layer_learn_data.inputs[node_in] * node_value;
                // The costGradientW array stores these partial derivatives for each weight.
                // Note: the derivative is being added to the array here because ultimately we want
                // to calculate the average gradient across all the data in the training batch
                let flat_weight_index = self.get_flat_weight_index(node_in, node_out);
                self.cost_gradient_w[flat_weight_index] += derivative_cost_wrt_weight;
            }
        }

        // Update cost gradient with respect to biases
        for node_out in 0..self.num_nodes_out {
            // Evaluate partial derivative: cost / bias
            let derivative_cost_wrt_bias = 1. * layer_learn_data.node_values[node_out];
            self.cost_gradient_b[node_out] += derivative_cost_wrt_bias;
        }
    }

    pub fn get_weight(&self, node_in: usize, node_out: usize) -> f64 {
        let flat_index = node_out * self.num_nodes_in + node_in;
        return self.weights[flat_index];
    }

    pub fn get_flat_weight_index(
        &self,
        input_neuron_index: usize,
        output_neuron_index: usize,
    ) -> usize {
        output_neuron_index * self.num_nodes_in + input_neuron_index
    }

    pub fn set_activation_function(&mut self, activation: Activation) {
        self.activation = activation;
    }

    pub fn initialize_random_weights(&mut self, rng: &mut dyn RngCore) {
        fn random_in_normal_distribution(
            rng: &mut dyn RngCore,
            mean: f64,
            standard_deviation: f64,
        ) -> f64 {
            let x1: f64 = 1. - rng.gen_range(0.0..1.0);
            let x2: f64 = 1. - rng.gen_range(0.0..1.0);

            let y1 = (-2.0 * x1.log(E)).sqrt() * (2.0 * PI * x2).cos();
            y1 * standard_deviation + mean
        }
        for i in 0..self.weights.len() {
            self.weights[i] =
                random_in_normal_distribution(rng, 0., 1.) / (self.num_nodes_in as f64).sqrt();
        }
    }
}
