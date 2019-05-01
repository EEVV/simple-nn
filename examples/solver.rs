#[macro_use]
extern crate simple_nn;

use simple_nn::Net;

const XOR_NET_EPOCHS: usize = 10;
const SOLVER_NET_EPOCHS: usize = 10;

const T: f64 = 1.0;
const F: f64 = -1.0;

fn main() {
    let training_set = sample![
        [F, F] => [F],
        [F, T] => [T],
        [T, F] => [T],
        [T, T] => [F]
    ];

    let mut xor_net = Net::new(&[2, 3, 1]);
    let mut multiplier = 1.0;
    println!("xor_net training");
    for i in 0..XOR_NET_EPOCHS {
        let error = xor_net.train_once(multiplier, |net| {
            let mut error = 0.0;
            for (input, output) in training_set.iter() {
                error += simple_nn::diff(&net.calc(input), output);
            }
            error

        });
        println!("{} {} {}", i, error, multiplier);
        multiplier /= 1.05;
    }

    println!("xor_net testing");
    for (input, output) in training_set.iter() {
        let predicted = &xor_net.calc(input);
        println!("{:?} {:?}", output, predicted);
    }

    let mut solver_net = Net::new(&[0, 3, 2]);
    solver_net.mutate(1.0);
    multiplier = 1.0;
    println!("solver_net training");
    for i in 0..SOLVER_NET_EPOCHS {
        let error = solver_net.train_once(multiplier, |net| {
            xor_net.calc(&net.calc(&[]))[0]
        });
        println!("{} {} {}", i, error, multiplier);
        multiplier /= 1.05;
    }

    println!("solver_net testing");
    {
        let predicted = &solver_net.calc(&[]);
        println!("{:?}", predicted);
    }
}