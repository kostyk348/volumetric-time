//! `vt` — check the mechanisms of volumetric-time without training anything.
//!
//! ```text
//! vt selftest                          # every invariant, with the measured numbers printed
//! vt geometry before.csv after.csv      # one update: rotation share, radial part, rank collapse
//! vt budget costs.csv BUDGET            # hard budget: spend, gain, dual price
//! vt lead log.csv THRESHOLD             # which signal peaks before the loss crosses
//! ```
//!
//! CSV matrices are one row per line, values comma separated.

use std::process::ExitCode;
use vt::{fusion, geometry, plan};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first() else {
        println!("vt <selftest|geometry|budget|lead> ...");
        return ExitCode::SUCCESS;
    };
    match command.as_str() {
        "selftest" => selftest(),
        "geometry" => {
            let (Some(before), Some(after)) = (args.get(1), args.get(2)) else {
                eprintln!("vt geometry before.csv after.csv");
                return ExitCode::FAILURE;
            };
            let (Ok(w0), Ok(w1)) = (read_matrix(before), read_matrix(after)) else {
                eprintln!("could not read both matrices");
                return ExitCode::FAILURE;
            };
            if w0.len() != w1.len() || w0[0].len() != w1[0].len() {
                eprintln!("the matrices have different shapes");
                return ExitCode::FAILURE;
            }
            let update = geometry::analyse(&w0, &w1);
            println!("v_ang            {:.6}", update.v_ang);
            println!("v_rad            {:.6}", update.v_rad);
            println!("rotation share   {:.4}", update.rotation_share);
            println!("rank before      {:.3}", update.rank_before);
            println!("rank after       {:.3}", update.rank_after);
            println!("rank collapse    {:.2} %", update.rank_collapse_pct);
            println!("spectral norm    {:.4}", geometry::spectral_norm(&w0));
            ExitCode::SUCCESS
        }
        "budget" => {
            let (Some(path), Some(budget)) = (args.get(1), args.get(2)) else {
                eprintln!("vt budget costs.csv BUDGET");
                return ExitCode::FAILURE;
            };
            let Ok(budget) = budget.parse::<f64>() else {
                eprintln!("budget must be a number");
                return ExitCode::FAILURE;
            };
            let Ok(rows) = read_pairs(path) else {
                eprintln!("could not read {path}");
                return ExitCode::FAILURE;
            };
            let costs: Vec<f64> = rows.iter().map(|(c, _)| *c).collect();
            let gains: Vec<f64> = rows.iter().map(|(_, g)| *g).collect();
            let allocation = plan::allocate(&costs, &gains, budget);
            let (price, spend) = plan::dual_ascent(&costs, &gains, budget, 4000);
            let taken = allocation.selected.iter().filter(|s| **s).count();
            println!("units            {}", costs.len());
            println!("taken            {taken}");
            println!("spent            {:.3} of {budget:.3} ({:.1} %)", allocation.spent, allocation.spent / budget.max(1e-12) * 100.0);
            println!("gain             {:.3}", allocation.gain);
            println!("dual price       {price:.4}");
            println!("dual spend       {spend:.3}");
            ExitCode::SUCCESS
        }
        "lead" => {
            let (Some(path), Some(threshold)) = (args.get(1), args.get(2)) else {
                eprintln!("vt lead log.csv THRESHOLD");
                return ExitCode::FAILURE;
            };
            let Ok(threshold) = threshold.parse::<f64>() else {
                eprintln!("threshold must be a number");
                return ExitCode::FAILURE;
            };
            let Ok((header, columns)) = read_table(path) else {
                eprintln!("could not read {path}");
                return ExitCode::FAILURE;
            };
            if columns.is_empty() {
                eprintln!("no columns in {path}");
                return ExitCode::FAILURE;
            }
            let loss = columns[0].clone();
            let signals: Vec<(String, Vec<f64>)> = header
                .iter()
                .skip(1)
                .cloned()
                .zip(columns.iter().skip(1).cloned())
                .collect();
            for lead in plan::lead_times(&loss, threshold, &signals) {
                let verdict = if lead.lead == i64::MIN {
                    "no crossing".to_string()
                } else if lead.lead > 0 {
                    format!("{} samples earlier", lead.lead)
                } else {
                    format!("{} samples late", -lead.lead)
                };
                println!("{:<20} peak at {:<6} {}", lead.signal, lead.peak, verdict);
            }
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown command {other}");
            ExitCode::FAILURE
        }
    }
}

fn selftest() -> ExitCode {
    println!("== geometry: rotation, scaling, basis invariance ==");
    let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let angle = 0.01_f64;
    let rotation = vec![vec![angle.cos(), -angle.sin()], vec![angle.sin(), angle.cos()]];
    let rotated = geometry::matmul(&rotation, &identity);
    let spin = geometry::analyse(&identity, &rotated);
    println!("rotated by 0.01 rad: v_ang {:.6}, rotation share {:.4}", spin.v_ang, spin.rotation_share);

    let mut scaled = identity.clone();
    scaled[0][0] = 1.01;
    scaled[1][1] = 1.01;
    let stretch = geometry::analyse(&identity, &scaled);
    println!("scaled by 1.01     : v_ang {:.6}, rotation share {:.4}", stretch.v_ang, stretch.rotation_share);

    let w0 = vec![vec![0.9, 0.3, -0.2], vec![0.1, 1.2, 0.4], vec![0.0, -0.5, 0.8]];
    let w1 = vec![vec![0.85, 0.33, -0.25], vec![0.15, 1.1, 0.5], vec![0.05, -0.45, 0.9]];
    let mut q = vec![vec![0.0; 3]; 3];
    for r in 0..2 {
        for c in 0..2 {
            q[r][c] = rotation[r][c];
        }
    }
    q[2][2] = 1.0;
    let direct = geometry::analyse(&w0, &w1);
    let rebased = geometry::analyse(&geometry::matmul(&q, &w0), &geometry::matmul(&q, &w1));
    println!("basis invariance   : |Δv_ang| = {:.2e} (must be ~0)", (direct.v_ang - rebased.v_ang).abs());

    println!("== fusion: a merged pair must compute the same function ==");
    let w_prev = vec![vec![0.2, -0.5, 0.7], vec![0.1, 0.9, -0.3]];
    let w_next = vec![vec![1.1, -0.4], vec![0.3, 0.8], vec![-0.6, 0.2]];
    let (w, b) = fusion::merge(&w_next, &[0.01, -0.02, 0.03], &w_prev, &[0.05, -0.1]);
    let x = vec![0.3, -0.7, 1.4];
    let reference = fusion::apply_two(&w_next, &[0.01, -0.02, 0.03], &w_prev, &[0.05, -0.1], &x);
    let merged = fusion::apply_merged(&w, &b, &x);
    let worst = reference.iter().zip(&merged).map(|(a, c)| (a - c).abs()).fold(0.0_f64, f64::max);
    println!("merged vs pair     : max |Δ| = {:.2e}", worst);
    let params_before = 3 * 2 + 2 + 2 * 3 + 3;
    let params_after = w.len() * w[0].len() + b.len();
    println!("parameters         : {params_before} -> {params_after}");

    println!("== budget: the constraint must hold, and the price must find it ==");
    let costs: Vec<f64> = (1..=200).map(|i| 0.5 + (i % 9) as f64 * 0.25).collect();
    let gains: Vec<f64> = (1..=200).map(|i| ((i * 31) % 97) as f64 / 97.0).collect();
    let budget = 300.0;
    let allocation = plan::allocate(&costs, &gains, budget);
    let (price, spend) = plan::dual_ascent(&costs, &gains, budget, 4000);
    println!("greedy spend       : {:.2} of {:.0} (over: {})", allocation.spent, budget, allocation.spent > budget + 1e-9);
    println!("dual price         : {price:.4}, spend {:.2} ({:.2} % of budget)", spend, spend / budget * 100.0);

    println!("== lead time: a signal that peaks after the crossing is useless ==");
    let loss = vec![9.0, 8.0, 6.0, 4.0, 1.0, 0.2];
    let early: Vec<f64> = vec![0.0, 1.0, 0.1, 0.1, 0.1, 0.1];
    let late: Vec<f64> = vec![0.0, 0.1, 0.1, 0.1, 0.1, 1.0];
    for lead in plan::lead_times(&loss, 2.0, &[("v_ang".into(), early), ("weight norm".into(), late)]) {
        println!("{:<20} peak {} lead {}", lead.signal, lead.peak, lead.lead);
    }

    let ok = spin.rotation_share > 0.95
        && stretch.rotation_share < 1e-9
        && (direct.v_ang - rebased.v_ang).abs() < 1e-12
        && worst < 1e-12
        && allocation.spent <= budget + 1e-9;
    println!("\nselftest: {}", if ok { "all invariants hold" } else { "AN INVARIANT FAILED" });
    if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

fn read_matrix(path: &str) -> Result<Vec<Vec<f64>>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: Result<Vec<f64>, _> = line.split(',').map(|part| part.trim().parse::<f64>()).collect();
        rows.push(row.map_err(|_| format!("bad number in {line:?}"))?);
    }
    if rows.is_empty() {
        return Err("empty matrix".into());
    }
    Ok(rows)
}

fn read_pairs(path: &str) -> Result<Vec<(f64, f64)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            continue;
        }
        let cost = parts[0].trim().parse::<f64>().map_err(|_| format!("bad cost in {line:?}"))?;
        let gain = parts[1].trim().parse::<f64>().map_err(|_| format!("bad gain in {line:?}"))?;
        out.push((cost, gain));
    }
    Ok(out)
}

fn read_table(path: &str) -> Result<(Vec<String>, Vec<Vec<f64>>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header_line = lines.next().ok_or("empty file")?;
    let header: Vec<String> = header_line.split(',').map(|s| s.trim().to_string()).collect();
    let mut columns = vec![Vec::new(); header.len()];
    for line in lines {
        for (index, part) in line.split(',').enumerate() {
            if index >= columns.len() {
                break;
            }
            if let Ok(value) = part.trim().parse::<f64>() {
                columns[index].push(value);
            }
        }
    }
    Ok((header, columns))
}
