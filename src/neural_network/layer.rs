use crate::utils::transpose;

use super::neuron;
pub struct Layer {
    neurons: Vec<neuron::Neuron>,
    pub neuron_results: Vec<Vec<f32>>,
    pub error_delta: Vec<Vec<f32>>
}

impl Layer {
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exponents = vec![];
        let mut exp_sum: f32 = 0.0;

        for logit in logits {
            let exponent = (logit - max).exp(); //Prevents overflow by subtracting max logit from the current one.
            exponents.push(exponent);
            exp_sum += exponent;
        }

        for exp in &mut exponents {
            *exp /= exp_sum;
        }
        return exponents;
    }

    pub fn run_layer(&mut self, inputs:&[Vec<f32>], softmax:bool) {
        let mut results: Vec<Vec<f32>> = vec![];
        for neuron in &mut self.neurons {
            let z = neuron.preactivate(inputs);
            if !softmax { results.push(neuron.relu(&z)) }
            else { results.push(z); }
        }
        
        let mut  temp_results = transpose(&results);
        if softmax {
            for sample_logits in &mut temp_results {
                *sample_logits = self.softmax(sample_logits);
            }
        }
        self.neuron_results = temp_results;
    } 
}