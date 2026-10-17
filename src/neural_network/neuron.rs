pub struct Neuron {
    weights: Vec<f32>,
    bias: f32,
    errors: Vec<f32>,
    weighted_sums: Vec<f32>,
}

impl Neuron {
    fn relu(&self, z: &[f32]) -> Vec<f32> {
        let mut activated: Vec<f32> = vec![];
        for sum in z {
            activated.push(sum.max(0.0));
        }
        return activated;
    }

    fn preactivation(&mut self, inputs: &[Vec<f32>]) {
        self.weighted_sums.clear();
        for column in inputs {
            let mut sum = self.bias;
            for (weight, input) in self.weights.iter().zip(column) {
                sum += weight * input;
            }
            self.weighted_sums.push(sum);
        }
    }

    pub fn activation(&mut self, inputs: &[Vec<f32>]) -> Vec<f32> {
        self.preactivation(inputs);
        self.relu(&self.weighted_sums)
    }
}
