pub struct Neuron {
    weights: Vec<f32>,
    bias: f32,
    error: f32,
    weighted_sum: f32,
}

impl Neuron {
    fn relu(&self, z: f32) -> f32 {
        z.max(0.0)
    }

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

    fn preactivation(&self, inputs: Vec<f32>) -> f32 {
        let mut sum = self.bias;
        for (weight, input) in self.weights.iter().zip(inputs) {
            sum += weight * input;
        }
        return sum;
    }
}
