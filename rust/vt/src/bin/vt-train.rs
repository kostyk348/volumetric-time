//! `vt-train` — the deciding experiment, in plain Rust.
//!
//! ```text
//! vt-train --arm base        --epochs 1200 --seed 1
//! vt-train --arm randfreeze  --epochs 1200 --seed 1
//! vt-train --arm v2          --epochs 1200 --seed 1 --verbose
//! ```

use std::process::ExitCode;
use vt::train::{run, Arm, Config};

fn main() -> ExitCode {
    let mut config = Config::default();
    let mut verbose = false;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        let value = args.get(index + 1).cloned().unwrap_or_default();
        match args[index].as_str() {
            "--arm" => match Arm::parse(&value) {
                Some(arm) => config.arm = arm,
                None => {
                    eprintln!("--arm must be base | randfreeze | v2");
                    return ExitCode::FAILURE;
                }
            },
            "--epochs" => config.epochs = value.parse().unwrap_or(config.epochs),
            "--seed" => config.seed = value.parse().unwrap_or(config.seed),
            "--dim" => config.dim = value.parse().unwrap_or(config.dim),
            "--hidden" => config.hidden = value.parse().unwrap_or(config.hidden),
            "--batch" => config.batch = value.parse().unwrap_or(config.batch),
            "--lr" => config.lr = value.parse().unwrap_or(config.lr),
            "--wd" => config.weight_decay = value.parse().unwrap_or(config.weight_decay),
            "--log" => config.log_every = value.parse().unwrap_or(config.log_every),
            "--p" => config.p = value.parse().unwrap_or(config.p),
            "--verbose" => {
                verbose = true;
                index += 1;
                continue;
            }
            "--help" | "-h" => {
                println!("vt-train [--arm base|randfreeze|v2] [--epochs N] [--seed N] [--verbose]");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown flag {other}");
                return ExitCode::FAILURE;
            }
        }
        index += 2;
    }

    let started = std::time::Instant::now();
    let report = run(&config, verbose);
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "{} seed={} epochs={} | to95 {} | best acc {:.3} | final te {:.4} | mem@ {:?} | saved {:.1} % | freezes {} | {:.1} s",
        report.arm,
        report.seed,
        config.epochs,
        report
            .epoch_to_95
            .map(|e| e.to_string())
            .unwrap_or_else(|| "never".into()),
        report.best_test_acc,
        report.final_test_loss,
        if report.train_loss_first_mem.is_nan() { None } else { Some(report.train_loss_first_mem as usize) },
        report.backward_saved_pct,
        report.freezes,
        elapsed
    );
    ExitCode::SUCCESS
}
