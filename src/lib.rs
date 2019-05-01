extern crate rand;

#[macro_export]
macro_rules! sample {
    ($($x:expr => $y:expr),*) => {
        vec![$(($x.to_vec(), $y.to_vec())),*]
    }
}

pub fn diff(predicted: &[f64], actual: &[f64]) -> f64 {
    assert!(predicted.len() == actual.len(), "predicted length ({}) and actual length ({}) mismatch", predicted.len(), actual.len());

    let mut error = 0.0;
    for i in 0..actual.len() {
        let z = predicted[i] - actual[i];
        error += z * z;
    }

    error / actual.len() as f64
}

#[derive(Debug, Clone)]
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    fn new(inputs: usize) -> Neuron {
        Neuron {
            weights: vec![0f64; inputs],
            bias: 0f64
        }
    }

    fn calc(&self, input: &[f64]) -> f64 {
        assert!(self.weights.len() == input.len(), "weight length ({}) and neuron length ({}) mismatch", self.weights.len(), input.len());

        let mut sum = self.bias;
        for (i, val) in input.iter().enumerate() {
            sum += val * self.weights[i];
        }
        
        sum.tanh()
    }

    fn mutate(&mut self, multiplier: f64) {
        for weight in self.weights.iter_mut() {
            let random = 2.0 * rand::random::<f64>() - 1.0;
            *weight += multiplier * random;
        }

        let random = 2.0 * rand::random::<f64>() - 1.0;
        self.bias += multiplier * random;
    }
}

#[derive(Debug, Clone)]
struct Layer {
    neurons: Vec<Neuron>
}

impl Layer {
    fn new(inputs: usize, current: usize) -> Layer {
        Layer {
            neurons: vec![Neuron::new(inputs); current]
        }
    }

    // todo use `&mut [f64]` pass-in return instead of `Vec<f64>`
    fn calc(&self, input: &[f64]) -> Vec<f64> {
        let mut r = Vec::with_capacity(self.neurons.len());

        for neuron in self.neurons.iter() {
            r.push(neuron.calc(input));
        }

        r
    }

    fn mutate(&mut self, multiplier: f64) {
        for neuron in self.neurons.iter_mut() {
            neuron.mutate(multiplier);
        }
    }

    fn mutate_sparsly(&mut self, multiplier: f64, percent: f64) {
        for neuron in self.neurons.iter_mut() {
            if rand::random::<f64>() <= percent {
                neuron.mutate(multiplier);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Net {
    layers: Vec<Layer>
}

impl Net {
    pub fn new(config: &[usize]) -> Net {
        assert!(config.len() >= 2, "layout must have at least an input and output");
        let mut last = config[0];

        let mut layers = Vec::with_capacity(config.len());
        for n in config.iter().skip(0) {
            layers.push(Layer::new(last, *n));
            last = *n;
        }

        Net {
            layers: layers
        }
    }

    pub fn calc(&self, input: &[f64]) -> Vec<f64> {
        let mut r = Vec::new();
        let mut input = input;

        for layer in self.layers.iter() {
            r = layer.calc(input);
            input = &r;
        }

        r
    }

    pub fn mutate(&mut self, multiplier: f64) {
        for layer in self.layers.iter_mut() {
            layer.mutate(multiplier);
        }
    }

    pub fn mutate_sparsly(&mut self, multiplier: f64, percent: f64) {
        for layer in self.layers.iter_mut() {
            layer.mutate_sparsly(multiplier, percent);
        }
    }

    pub fn train_once<F>(&mut self, multiplier: f64, error_fun: F) -> f64 where F: Fn(&Net) -> f64 {
        let error = error_fun(self);
        //println!("{}", error);
        let mut new_error;

        loop {
            let mut new_net = self.clone();
            new_net.mutate_sparsly(multiplier, 0.25);
            new_error = error_fun(&new_net);
            //println!("{}", new_error);
            if new_error < error {
                *self = new_net;
                break;
            }
        }

        new_error
    }
}

