//! Compute as a budget with a price, and the measurement that decides whether the whole idea holds.
//!
//! Two mechanisms from `docs/architecture-v2.md`:
//!
//! 1. `allocate` / `dual_ascent` — hard budget instead of a soft penalty. A candidate unit of work
//!    (a token in a layer) has a cost and an expected gain; we spend the budget on the best gain per
//!    unit cost, and the dual variable `λ` prices the budget so that the constraint is tight. The
//!    budget is then *respected*, not hoped for — which is what a test can check.
//! 2. `lead_times` — does a signal peak before generalisation? That is the question the notebook
//!    asked with plots, and it has a number: the lead, in samples.

#[derive(Debug, Clone, PartialEq)]
pub struct Allocation {
    pub selected: Vec<bool>,
    pub spent: f64,
    pub gain: f64,
    /// Index of the last unit considered, for tracing why something was skipped.
    pub skipped_expensive: usize,
}

/// Greedy knapsack: sort by gain per unit cost, take while it fits. Not optimal in general, but
/// optimal for the fractional relaxation, and the units here are small and many.
pub fn allocate(cost: &[f64], gain: &[f64], budget: f64) -> Allocation {
    assert_eq!(cost.len(), gain.len(), "cost and gain describe the same units");
    let mut order: Vec<usize> = (0..cost.len()).collect();
    order.sort_by(|a, b| {
        let ra = if cost[*a] > 0.0 { gain[*a] / cost[*a] } else { f64::INFINITY };
        let rb = if cost[*b] > 0.0 { gain[*b] / cost[*b] } else { f64::INFINITY };
        rb.partial_cmp(&ra).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut selected = vec![false; cost.len()];
    let mut spent = 0.0;
    let mut taken = 0.0;
    let mut skipped = 0;
    for index in order {
        if gain[index] <= 0.0 {
            continue;
        }
        if spent + cost[index] <= budget + 1e-12 {
            selected[index] = true;
            spent += cost[index];
            taken += gain[index];
        } else {
            skipped += 1;
        }
    }
    Allocation { selected, spent, gain: taken, skipped_expensive: skipped }
}

/// Dual ascent on the price of compute. `λ` rises while the budget is exceeded and falls while it
/// is not: the allocation is then `gain_i > λ · cost_i`. Returns the price and the resulting spend.
pub fn dual_ascent(cost: &[f64], gain: &[f64], budget: f64, iterations: usize) -> (f64, f64) {
    assert_eq!(cost.len(), gain.len());
    let mut price = 0.0_f64;
    let mut rate = 0.5_f64;
    let mut best_price = 0.0;
    let mut best_spend = spend_at(cost, gain, budget, price);
    for _ in 0..iterations {
        let spend = spend_at(cost, gain, budget, price);
        if (spend - budget).abs() < (budget * 0.01).max(1e-12) {
            return (price, spend);
        }
        if spend > budget {
            price += rate;
        } else {
            price = (price - rate).max(0.0);
            rate *= 0.95;
        }
        if (spend - budget).abs() < (best_spend - budget).abs() {
            best_price = price;
            best_spend = spend;
        }
    }
    (best_price, best_spend)
}

/// Spend when units are taken while their gain exceeds `λ · cost`, in order of gain per cost.
fn spend_at(cost: &[f64], gain: &[f64], budget: f64, price: f64) -> f64 {
    let mut order: Vec<usize> = (0..cost.len()).collect();
    order.sort_by(|a, b| {
        let ra = if cost[*a] > 0.0 { gain[*a] / cost[*a] } else { f64::INFINITY };
        let rb = if cost[*b] > 0.0 { gain[*b] / cost[*b] } else { f64::INFINITY };
        rb.partial_cmp(&ra).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut spent = 0.0;
    for index in order {
        if gain[index] <= price * cost[index] {
            continue;
        }
        if spent + cost[index] <= budget + 1e-12 {
            spent += cost[index];
        }
    }
    spent
}

#[derive(Debug, Clone, PartialEq)]
pub struct Lead {
    pub signal: String,
    /// Position of the peak of the absolute first difference.
    pub peak: usize,
    /// Samples between the peak and the moment the loss crossed the threshold. Positive = earlier.
    pub lead: i64,
}

/// When did each signal peak, and how long before the loss crossed the threshold?
///
/// This is the metric that decides the whole programme. A signal that peaks *after* the crossing
/// cannot be used to act on the crossing, no matter how well it correlates.
pub fn lead_times(loss: &[f64], threshold: f64, signals: &[(String, Vec<f64>)]) -> Vec<Lead> {
    let crossing = loss.iter().position(|value| *value < threshold);
    let Some(crossing) = crossing else {
        return signals
            .iter()
            .map(|(name, _)| Lead { signal: name.clone(), peak: 0, lead: i64::MIN })
            .collect();
    };
    signals
        .iter()
        .map(|(name, values)| {
            let mut peak = 0usize;
            let mut best = 0.0f64;
            for index in 1..values.len() {
                let step = (values[index] - values[index - 1]).abs();
                if step > best {
                    best = step;
                    peak = index;
                }
            }
            Lead { signal: name.clone(), peak, lead: crossing as i64 - peak as i64 }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_budget_is_respected_and_not_overspent() {
        let cost: Vec<f64> = (1..=50).map(|i| (i % 7) as f64 + 0.5).collect();
        let gain: Vec<f64> = (1..=50).map(|i| ((i * 13) % 17) as f64).collect();
        for budget in [0.0, 5.0, 25.0, 60.0, 1e6] {
            let allocation = allocate(&cost, &gain, budget);
            assert!(allocation.spent <= budget + 1e-9, "budget {budget}: spent {}", allocation.spent);
        }
    }

    #[test]
    fn more_budget_never_buys_less_gain() {
        let cost: Vec<f64> = (1..=40).map(|i| (i % 5) as f64 + 1.0).collect();
        let gain: Vec<f64> = (1..=40).map(|i| ((i * 7) % 11) as f64).collect();
        let mut previous = -1.0;
        for budget in (0..=200).step_by(10) {
            let allocation = allocate(&cost, &gain, budget as f64);
            assert!(allocation.gain >= previous - 1e-9, "gain fell at budget {budget}");
            previous = allocation.gain;
        }
    }

    #[test]
    fn the_dual_price_tracks_the_constraint_to_one_percent() {
        let cost: Vec<f64> = (1..=200).map(|i| 0.5 + (i % 9) as f64 * 0.25).collect();
        let gain: Vec<f64> = (1..=200).map(|i| ((i * 31) % 97) as f64 / 97.0).collect();
        let budget: f64 = 300.0;
        let (price, spend) = dual_ascent(&cost, &gain, budget, 4000);
        assert!(spend <= budget * 1.01 + 1e-9, "spend {spend} over budget {budget} at price {price}");
    }

    #[test]
    fn the_ordering_of_units_wins_before_a_worse_signal_can() {
        // One unit pays ten times better than the other and fits the budget alone.
        let cost = vec![1.0, 1.0];
        let gain = vec![10.0, 1.0];
        let allocation = allocate(&cost, &gain, 1.0);
        assert!(allocation.selected[0]);
        assert!(!allocation.selected[1]);
    }

    #[test]
    fn a_signal_that_peaks_early_has_a_positive_lead() {
        let loss = vec![9.0, 5.0, 1.0, 0.1];
        let early: Vec<f64> = vec![0.0, 1.0, 0.2, 0.2];
        let late: Vec<f64> = vec![0.0, 0.1, 0.2, 1.0];
        let leads = lead_times(&loss, 2.0, &[("early".into(), early), ("late".into(), late)]);
        assert_eq!(leads[0].peak, 1);
        assert_eq!(leads[0].lead, 1, "crossing at 2, peak at 1");
        assert!(leads[1].lead < 0, "the late signal cannot be acted upon");
    }

    #[test]
    fn no_crossing_is_reported_as_no_answer() {
        let loss = vec![9.0, 9.5, 9.0];
        let leads = lead_times(&loss, 1.0, &[("x".into(), vec![0.0, 1.0, 0.5])]);
        assert_eq!(leads[0].lead, i64::MIN);
    }
}
