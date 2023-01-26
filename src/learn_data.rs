use crate::Layer;

pub struct NetworkLearnData {
    pub layer_data: Vec<LayerLearnData>,
}

impl NetworkLearnData {
    pub fn new(layers: &Vec<Layer>) -> NetworkLearnData {
        let mut layer_data = vec![];
        for layer in layers.iter() {
            layer_data.push(LayerLearnData::new(layer));
        }
        NetworkLearnData { layer_data }
    }
}

pub struct LayerLearnData {
    pub inputs: Vec<f64>,
    pub weighted_inputs: Vec<f64>,
    pub activations: Vec<f64>,
    pub node_values: Vec<f64>,
}

impl LayerLearnData {
    pub fn new(layer: &Layer) -> LayerLearnData {
        LayerLearnData {
            inputs: vec![],
            weighted_inputs: vec![0.; layer.num_nodes_out],
            activations: vec![0.; layer.num_nodes_out],
            node_values: vec![0.; layer.num_nodes_out],
        }
    }
}
