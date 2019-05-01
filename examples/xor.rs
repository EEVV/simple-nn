#[macro_use]
extern crate simple_nn;

use simple_nn::Net;

const EPOCHS: usize = 10;

const T: f64 = 1.0;
const F: f64 = -1.0;

fn main() {
    let training_set = sample![
        [F, F] => [F],
        [F, T] => [T],
        [T, F] => [T],
        [T, T] => [F]
    ];

    let mut net = Net::new(&[2, 3, 1]);
    let mut multiplier = 1.0;
    println!("training");
    for i in 0..EPOCHS {
        let mut error = 0.0;
        error += net.train_once(multiplier, |net| {
            let mut error = 0.0;
            for (input, output) in training_set.iter() {
                error += simple_nn::diff(&net.calc(input), output);
            }
            error

        });
        println!("{} {} {}", i, error, multiplier);
        multiplier /= 1.05;
    }

    println!("testing");
    for (input, output) in training_set.iter() {
        let predicted = &net.calc(input);
        println!("{:?} {:?}", output, predicted);
    }
}