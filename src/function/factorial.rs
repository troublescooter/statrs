use std::{f32, f64};
use std::sync::{Once, ONCE_INIT};
use function::gamma;

pub trait Factorial {
    /// Computes the factorial function `x -> x!`.
    ///
    /// # Remarks
    ///
    /// Returns infinity if `x > max_arg()`
    fn factorial(x: usize) -> Self;

    /// Computes the logarithmic factorial fucntion `x -> ln(x!)`
    /// for `x >= 0`
    ///
    /// # Remarks
    ///
    /// Returns `0.0` if `x <= 1`
    fn ln_factorial(x: usize) -> Self;

    /// The maximum argument size for `factorial(x)`
    /// before the return value will overflow
    fn max_arg() -> usize;
}

impl Factorial for f64 {
    fn factorial(x: usize) -> f64 {
        if x > Self::max_arg() {
            f64::INFINITY
        } else {
            get_fcache()[x]
        }
    }

    fn ln_factorial(x: usize) -> f64 {
        if x <= 1 {
            0.0
        } else if x > Self::max_arg() {
            gamma::ln_gamma::<f64>(x as f64 + 1.0)
        } else {
            get_fcache()[x].ln()
        }
    }

    fn max_arg() -> usize {
        170
    }
}

impl Factorial for f32 {
    fn factorial(x: usize) -> f32 {
        if x > Self::max_arg() {
            f32::INFINITY
        } else {
            get_fcache()[x] as f32
        }
    }

    fn ln_factorial(x: usize) -> f32 {
        if x <= 1 {
            0.0
        } else if x > Self::max_arg() {
            gamma::ln_gamma::<f32>(x as f32 + 1.0)
        } else {
            get_fcache()[x].ln() as f32
        }
    }

    fn max_arg() -> usize {
        34
    }
}

pub trait Binomial {
    /// Computes the binomial coefficient `n choose k`
    /// where `k` and `n` are non-negative values.
    ///
    /// # Remarks
    ///
    /// Returns `0.0` if `k > n`
    fn binomial(n: usize, k: usize) -> Self;

    /// Computes the natural logarithm of the binomial coefficient
    /// `ln(n choose k)` where `k` and `n` are non-negative values
    ///
    /// # Remarks
    ///
    /// Returns negative infinity if `k > n`
    fn ln_binomial(n: usize, k: usize) -> Self;
}

macro_rules! impl_binomial_for {
    ($T:ty, $ninf:expr) => (
        impl Binomial for $T {
            fn binomial(n: usize, k: usize) -> $T {
                if k > n {
                    0.0
                } else {
                    (0.5 +
                    (Self::ln_factorial(n) - Self::ln_factorial(k) - Self::ln_factorial(n - k)).exp())
                        .floor()
                }
            }

            fn ln_binomial(n: usize, k: usize) -> $T {
                if k > n {
                    $ninf
                } else {
                    Self::ln_factorial(n) - Self::ln_factorial(k) - Self::ln_factorial(n - k)
                }
            }
        }
    );
}
impl_binomial_for!(f64, f64::NEG_INFINITY);
impl_binomial_for!(f32, f32::NEG_INFINITY);

// Initialization for pre-computed cache of 171 factorial
// values 0!...170! for f64
const CACHE_SIZE: usize = 171;

static mut FCACHE: &'static mut [f64; CACHE_SIZE] = &mut [1.0; CACHE_SIZE];
static START: Once = ONCE_INIT;

fn get_fcache() -> &'static [f64; CACHE_SIZE] {
    unsafe {
        START.call_once(|| {
            (1..CACHE_SIZE).fold(FCACHE[0], |acc, i| {
                let fac = acc * i as f64;
                FCACHE[i] = fac;
                fac
            });
        });
        FCACHE
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::{f64, u64};

    #[test]
    fn test_factorial_and_ln_factorial() {
        let mut factorial = 1.0;
        for i in 1..171 {
            factorial *= i as f64;
            assert_eq!(f64::factorial(i), factorial);
            assert_eq!(f64::ln_factorial(i), factorial.ln());
        }
    }

    #[test]
    fn test_factorial_overflow() {
        assert_eq!(f64::factorial(172), f64::INFINITY);
        assert_eq!(f64::factorial(u64::MAX), f64::INFINITY);
    }

    #[test]
    fn test_ln_factorial_does_not_overflow() {
        assert_eq!(f64::ln_factorial(1 << 10), 6078.2118847500501140);
        assert_almost_eq!(f64::ln_factorial(1 << 12), 29978.648060844048236, 1e-11);
        assert_eq!(f64::ln_factorial(1 << 15), 307933.81973375485425);
        assert_eq!(f64::ln_factorial(1 << 17), 1413421.9939462073242);
    }

    #[test]
    fn test_binomial() {
        assert_eq!(f64::binomial(1, 1), 1.0);
        assert_eq!(f64::binomial(5, 2), 10.0);
        assert_eq!(f64::binomial(7, 3), 35.0);
        assert_eq!(f64::binomial(1, 0), 1.0);
        assert_eq!(f64::binomial(0, 1), 0.0);
        assert_eq!(f64::binomial(5, 7), 0.0);
    }

    #[test]
    fn test_ln_binomial() {
        assert_eq!(f64::ln_binomial(1, 1), 1f64.ln());
        assert_almost_eq!(f64::ln_binomial(5, 2), 10f64.ln(), 1e-14);
        assert_almost_eq!(f64::ln_binomial(7, 3), 35f64.ln(), 1e-14);
        assert_eq!(f64::ln_binomial(1, 0), 1f64.ln());
        assert_eq!(f64::ln_binomial(0, 1), 0f64.ln());
        assert_eq!(f64::ln_binomial(5, 7), 0f64.ln());
    }
}
