pub struct Neuron {
    weights: Vec<f32>,
    bias: f32
}

impl Neuron {
    pub fn relu(&self, z: &[f32]) -> Vec<f32> {
        let mut activated: Vec<f32> = vec![];
        for sum in z {
            activated.push(sum.max(0.0));
        }
        return activated;
    }

    pub fn preactivate(&mut self, inputs: &[Vec<f32>]) -> Vec<f32> {
        let mut product_sum: Vec<f32> = vec![];
        for column in inputs {
            let mut sum = self.bias;
            for (weight, input) in self.weights.iter().zip(column) {
                sum += weight * input;
            }
            product_sum.push(sum);
        }
        return product_sum;
    }
}
