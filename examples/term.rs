use glam::Vec2;
use rand::prelude::*;
use seblague_neural_network::{DataPoint, NeuralNetwork};

fn training_data() -> Vec<DataPoint> {
    let mut rng = thread_rng();
    let mut points = vec![];
    for i in 0..100 {
        for j in 0..100 {
            if rng.gen_bool(0.02) {
                let position = Vec2::new(i as f32 / 100., j as f32 / 100.);
                let poison = position.distance(Vec2::splat(0.5)) < 0.25
                    || (j > 50 && i > 70)
                    || (j < 50 && i < 20)
                    || j < 10;
                points.push(DataPoint {
                    inputs: vec![position.x as f64, position.y as f64],
                    expected_outputs: if poison { vec![0., 1.] } else { vec![1., 0.] },
                });
            }
        }
    }
    points
}

fn main() {
    let training_data = training_data();
    let mut network = NeuralNetwork::new(&vec![2, 30, 30, 30, 2]);
    let mut rng = thread_rng();
    loop {
        for _ in 0..1000 {
            let mut subset = vec![];
            for _ in 0..20 {
                subset.push(training_data[rng.gen_range(0..training_data.len())].clone());
            }
            network.learn(&subset, 0.1, 0., 0.9);
        }
        let (term_width, term_height) = if let Some((w, h)) = term_size::dimensions() {
            (w as i32, h as i32 - 2)
        } else {
            (100, 50)
        };
        let mut plotted_points: Vec<(char, i32, i32)> = vec![];
        for data in training_data.iter() {
            let poison = data.expected_outputs[0] == 1.;
            plotted_points.push((
                if poison { '.' } else { '#' },
                (data.inputs[0] * term_width as f64) as i32,
                (data.inputs[1] * term_height as f64) as i32,
            ));
        }
        for yy in 0..term_height {
            let y = yy as f32 / term_height as f32;
            for xx in 0..term_width {
                let x = xx as f32 / term_width as f32;
                let (classification, _) = network.classify(&vec![x as f64, y as f64]);
                let mut draw = true;
                for point in plotted_points.iter() {
                    if xx == point.1 && yy == point.2 {
                        print!("{}", point.0);
                        draw = false;
                        break;
                    }
                }
                if draw {
                    if classification == 0 {
                        print!(" ");
                    } else {
                        print!("-");
                    }
                }
            }
            println!();
        }
        println!("Cost: {}", network.cost(&training_data));
    }
}
