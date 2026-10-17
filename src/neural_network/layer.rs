use super::neuron;

pub struct Layer {
    neurons: Vec<neuron::Neuron>
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
}