use std::f64;
use num;
use rand::{Rng, Rand};
use rand::distributions::{Sample, IndependentSample};
use function::{beta, factorial};
use statistics::*;
use distribution::{Univariate, Discrete, Distribution};
use result::Result;
use error::StatsError;
use {Float, Integer};

/// Implements the [Binomial](https://en.wikipedia.org/wiki/Binomial_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Binomial, Discrete};
/// use statrs::statistics::Mean;
///
/// let n = Binomial::new(0.5, 5).unwrap();
/// assert_eq!(n.mean(), 2.5);
/// assert_eq!(n.pmf(0), 0.03125);
/// assert_eq!(n.pmf(3), 0.3125);
/// assert_eq!(n.pmf(6), 0.0);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Binomial<P, N>
    where P: Float,
          N: Integer
{
    p: P,
    n: N,
}

impl<P, N> Binomial<P, N>
    where P: Float,
          N: Integer
{
    /// Constructs a new binomial distribution
    /// with a given `p` probability of success of `n`
    /// trials.
    ///
    /// # Errors
    ///
    /// Returns an error if `p` is `NaN`, less than `0.0`,
    /// greater than `1.0`, or if `n` is less than `0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Binomial;
    ///
    /// let mut result = Binomial::new(0.5, 5);
    /// assert!(result.is_ok());
    ///
    /// result = Binomial::new(-0.5, -5);
    /// assert!(result.is_err());
    /// ```
    pub fn new(p: P, n: N) -> Result<Binomial<P, N>> {
        if p.is_nan() || p < P::zero() || p > P::one() || n < N::zero() {
            Err(StatsError::BadParams)
        } else {
            Ok(Binomial { p: p, n: n })
        }
    }

    /// Returns the probability of success `p` of
    /// the binomial distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Binomial;
    ///
    /// let n = Binomial::new(0.5, 5).unwrap();
    /// assert_eq!(n.p(), 0.5);
    /// ```
    pub fn p(&self) -> P {
        self.p
    }

    /// Returns the number of trials `n` of the
    /// binomial distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Binomial;
    ///
    /// let n = Binomial::new(0.5, 5).unwrap();
    /// assert_eq!(n.n(), 5);
    /// ```
    pub fn n(&self) -> N {
        self.n
    }
}

impl<P, N> Sample<P> for Binomial<P, N>
    where P: Float,
          N: Integer
{
    /// Generate a random sample from a binomial
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn sample<R: Rng>(&mut self, r: &mut R) -> P {
        super::Distribution::sample(self, r)
    }
}

/// Generate a random independent sample from a binomial
/// distribution using `r` as the source of randomness.
/// Refer [here](#method.sample-1) for implementation details
impl<P, N> IndependentSample<P> for Binomial<P, N>
    where P: Float,
          N: Integer
{
    fn ind_sample<R: Rng>(&self, r: &mut R) -> P {
        super::Distribution::sample(self, r)
    }
}

impl<P, N> Distribution<P> for Binomial<P, N>
    where P: Float,
          N: Integer
{
    /// Generate a random sample from the binomial distribution
    /// using `r` as the source of randomness  where the range of
    /// values is `[0.0, n]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate rand;
    /// # extern crate statrs;
    /// use rand::StdRng;
    /// use statrs::distribution::{Binomial, Distribution};
    ///
    /// # fn main() {
    /// let mut r = rand::StdRng::new().unwrap();
    /// let n = Binomial::new(0.5, 5).unwrap();
    /// print!("{}", n.sample::<StdRng>(&mut r));
    /// # }
    /// ```
    fn sample<R: Rng>(&self, r: &mut R) -> P {
        num::range(N::zero(), self.n).fold(P::zero(), |acc, _| {
            let n = r.gen::<P>();
            if n < self.p { acc + P::one() } else { acc }
        })
    }
}

impl<P, N> Univariate<N, P> for Binomial<P, N>
    where P: Float,
          N: Integer
{
    /// Calulcates the cumulative distribution function for the
    /// binomial distribution at `x`
    ///
    /// # Remarks
    ///
    /// Returns `0,0` if `x < 0.0` and `1.0` if `x >= n`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// I_(1 - p)(n - x, 1 + x)
    /// ```
    ///
    /// where `I_(x)(a, b)` is the regularized incomplete beta function
    fn cdf(&self, x: P) -> P {
        if x < P::zero() {
            P::zero()
        } else if x >= P::from(self.n).unwrap() {
            P::one()
        } else {
            let k = x.floor();
            beta::beta_reg(P::from(self.n).unwrap() - k,
                           k + P::one(),
                           P::one() - self.p)
        }
    }
}

impl<P, N> Min<N> for Binomial<P, N>
    where P: Float,
          N: Integer
{
    /// Returns the minimum value in the domain of the
    /// binomial distribution representable by a 64-bit
    /// integer
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 0
    /// ```
    fn min(&self) -> N {
        N::zero()
    }
}

impl<P, N> Max<N> for Binomial<P, N>
    where P: Float,
          N: Integer
{
    /// Returns the maximum value in the domain of the
    /// binomial distribution representable by a 64-bit
    /// integer
    ///
    /// # Formula
    ///
    /// ```ignore
    /// n
    /// ```
    fn max(&self) -> N {
        self.n
    }
}

impl<P, N> Mean<P> for Binomial<P, N>
    where P: Float,
          N: Integer
{
    /// Returns the mean of the binomial distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// p * n
    /// ```
    fn mean(&self) -> P {
        self.p * P::from(self.n).unwrap()
    }
}

impl<P, N> Variance<P> for Binomial<P, N>
    where P: Float,
          N: Integer
{
    /// Returns the variance of the binomial distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// n * p * (1 - p)
    /// ```
    fn variance(&self) -> P {
        self.p * (P::one() - self.p) * P::from(self.n).unwrap()
    }

    /// Returns the standard deviation of the binomial distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(n * p * (1 - p))
    /// ```
    fn std_dev(&self) -> P {
        self.variance().sqrt()
    }
}

impl<P, N> Entropy<P> for Binomial<P, N>
    where P: Float,
          N: Integer
{
    /// Returns the entropy of the binomial distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 / 2) * ln (2 * π * e * n * p * (1 - p))
    /// ```
    fn entropy(&self) -> P {
        if self.p.is_zero() || self.p == P::one() {
            P::zero()
        } else {
            num::range_inclusive(N::zero(), self.n).fold(P::zero(), |acc, x| {
                let p = self.pmf(x);
                acc - p * p.ln()
            })
        }
    }
}

impl<P, N> Skewness<P> for Binomial<P, N>
    where P: Float,
          N: Integer
{
    /// Returns the skewness of the binomial distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 - 2p) / sqrt(n * p * (1 - p)))
    /// ```
    fn skewness(&self) -> P {
        (P::one() - P::from(2.0).unwrap() * self.p) /
        (P::from(self.n).unwrap() * self.p * (P::one() - self.p)).sqrt()
    }
}

impl<P, N> Median<P> for Binomial<P, N>
    where P: Float,
          N: Integer
{
    /// Returns the median of the binomial distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// floor(n * p)
    /// ```
    fn median(&self) -> P {
        (self.p * P::from(self.n).unwrap()).floor()
    }
}

impl<P, N> Mode<N> for Binomial<P, N>
    where P: Float,
          N: Integer
{
    /// Returns the mode for the binomial distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// floor((n + 1) * p)
    /// ```
    fn mode(&self) -> N {
        if self.p.is_zero() {
            N::zero()
        } else if self.p == P::one() {
            self.n
        } else {
            N::from((P::from(self.n).unwrap() + P::one()).floor()).unwrap()
        }
    }
}

impl<P, N> Discrete<N, P> for Binomial<P, N>
    where P: Float,
          N: Integer
{
    /// Calculates the probability mass function for the binomial
    /// distribution at `x`
    ///
    /// # Remarks
    ///
    /// Returns `0` if `x > n || x < 0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (n choose k) * p^k * (1 - p)^(n - k)
    /// ```
    fn pmf(&self, x: N) -> P {
        if x > self.n || x < N::zero() {
            P::zero()
        } else {
            if self.p.is_zero() && x.is_zero() {
                P::one()
            } else if self.p.is_zero() {
                P::zero()
            } else if self.p == P::one() && x == self.n {
                P::one()
            } else if self.p == P::one() {
                P::zero()
            } else {
                (factorial::ln_binomial(self.n.to_u64().unwrap(), x.to_u64().unwrap()) +
                 P::from(x).unwrap() * self.p.ln() +
                 P::from(self.n - x).unwrap() * (P::one() - self.p).ln())
                    .exp()
            }
        }
    }

    /// Calculates the log probability mass function for the binomial
    /// distribution at `x`
    ///
    /// # Remarks
    ///
    /// Returns `f64::NEG_INFINITY` if `x > n || x < 0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln((n choose k) * p^k * (1 - p)^(n - k))
    /// ```
    fn ln_pmf(&self, x: N) -> P {
        if x > self.n || x < N::zero() {
            P::neg_infinity()
        } else {
            if self.p.is_zero() && x.is_zero() {
                P::zero()
            } else if self.p.is_zero() {
                P::neg_infinity()
            } else if self.p == P::one() && x == self.n {
                P::zero()
            } else if self.p == P::one() {
                P::neg_infinity()
            } else {
                factorial::ln_binomial(self.n.to_u64().unwrap(), x.to_u64().unwrap()) +
                P::from(x).unwrap() * self.p.ln() +
                P::from(self.n - x).unwrap() * (P::one() - self.p).ln()
            }
        }
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::cmp::PartialEq;
    use std::fmt::Debug;
    use std::f64;
    use statistics::*;
    use distribution::{Univariate, Discrete, Binomial};

    fn try_create(p: f64, n: i64) -> Binomial<f64, i64> {
        let n = Binomial::new(p, n);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(p: f64, n: i64) {
        let n = try_create(p, n);
        assert_eq!(p, n.p());
    }

    fn bad_create_case(p: f64, n: i64) {
        let n = Binomial::new(p, n);
        assert!(n.is_err());
    }

    fn get_value<T, F>(p: f64, n: i64, eval: F) -> T
        where T: PartialEq + Debug,
              F: Fn(Binomial<f64, i64>) -> T
    {
        let n = try_create(p, n);
        eval(n)
    }

    fn test_case<T, F>(p: f64, n: i64, expected: T, eval: F)
        where T: PartialEq + Debug,
              F: Fn(Binomial<f64, i64>) -> T
    {
        let x = get_value(p, n, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(p: f64, n: i64, expected: f64, acc: f64, eval: F)
        where F: Fn(Binomial<f64, i64>) -> f64
    {
        let x = get_value(p, n, eval);
        assert_almost_eq!(expected, x, acc);
    }

    #[test]
    fn test_create() {
        create_case(0.0, 4);
        create_case(0.3, 3);
        create_case(1.0, 2);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(f64::NAN, 1);
        bad_create_case(-1.0, 1);
        bad_create_case(2.0, 1);
        bad_create_case(0.3, -2);
    }

    #[test]
    fn test_mean() {
        test_case(0.0, 4, 0.0, |x| x.mean());
        test_almost(0.3, 3, 0.9, 1e-15, |x| x.mean());
        test_case(1.0, 2, 2.0, |x| x.mean());
    }

    #[test]
    fn test_variance() {
        test_case(0.0, 4, 0.0, |x| x.variance());
        test_case(0.3, 3, 0.63, |x| x.variance());
        test_case(1.0, 2, 0.0, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_case(0.0, 4, 0.0, |x| x.std_dev());
        test_case(0.3, 3, 0.7937253933193771771505, |x| x.std_dev());
        test_case(1.0, 2, 0.0, |x| x.std_dev());
    }

    #[test]
    fn test_entropy() {
        test_case(0.0, 4, 0.0, |x| x.entropy());
        test_almost(0.3, 3, 1.1404671643037712668976423399228972051669206536461, 1e-15, |x| x.entropy());
        test_case(1.0, 2, 0.0, |x| x.entropy());
    }

    #[test]
    fn test_skewness() {
        test_case(0.0, 4, f64::INFINITY, |x| x.skewness());
        test_case(0.3, 3, 0.503952630678969636286, |x| x.skewness());
        test_case(1.0, 2, f64::NEG_INFINITY, |x| x.skewness());
    }

    #[test]
    fn test_median() {
        test_case(0.0, 4, 0.0, |x| x.median());
        test_case(0.3, 3, 0.0, |x| x.median());
        test_case(1.0, 2, 2.0, |x| x.median());
    }

    #[test]
    fn test_mode() {
        test_case(0.0, 4, 0, |x| x.mode());
        test_case(0.3, 3, 1, |x| x.mode());
        test_case(1.0, 2, 2, |x| x.mode());
    }

    #[test]
    fn test_min_max() {
        test_case(0.3, 10, 0, |x| x.min());
        test_case(0.3, 10, 10, |x| x.max());
    }

    #[test]
    fn test_pmf() {
        test_case(0.0, 1, 1.0, |x| x.pmf(0));
        test_case(0.0, 1, 0.0, |x| x.pmf(1));
        test_case(0.0, 3, 1.0, |x| x.pmf(0));
        test_case(0.0, 3, 0.0, |x| x.pmf(1));
        test_case(0.0, 3, 0.0, |x| x.pmf(3));
        test_case(0.0, 10, 1.0, |x| x.pmf(0));
        test_case(0.0, 10, 0.0, |x| x.pmf(1));
        test_case(0.0, 10, 0.0, |x| x.pmf(10));
        test_case(0.3, 1, 0.69999999999999995559107901499373838305473327636719, |x| x.pmf(0));
        test_case(0.3, 1, 0.2999999999999999888977697537484345957636833190918, |x| x.pmf(1));
        test_case(0.3, 3, 0.34299999999999993471888615204079956461021032657166, |x| x.pmf(0));
        test_almost(0.3, 3, 0.44099999999999992772448109690231306411849135972008, 1e-15, |x| x.pmf(1));
        test_almost(0.3, 3, 0.026999999999999997002397833512077451789759292859569, 1e-16, |x| x.pmf(3));
        test_almost(0.3, 10, 0.02824752489999998207939855277004937778546385011091, 1e-17, |x| x.pmf(0));
        test_almost(0.3, 10, 0.12106082099999992639752977030555903089040470780077, 1e-15, |x| x.pmf(1));
        test_almost(0.3, 10, 0.0000059048999999999978147480206303047454017251032868501, 1e-20, |x| x.pmf(10));
        test_case(1.0, 1, 0.0, |x| x.pmf(0));
        test_case(1.0, 1, 1.0, |x| x.pmf(1));
        test_case(1.0, 3, 0.0, |x| x.pmf(0));
        test_case(1.0, 3, 0.0, |x| x.pmf(1));
        test_case(1.0, 3, 1.0, |x| x.pmf(3));
        test_case(1.0, 10, 0.0, |x| x.pmf(0));
        test_case(1.0, 10, 0.0, |x| x.pmf(1));
        test_case(1.0, 10, 1.0, |x| x.pmf(10));
    }

    #[test]
    fn test_ln_pmf() {
        test_case(0.0, 1, 0.0, |x| x.ln_pmf(0));
        test_case(0.0, 1, f64::NEG_INFINITY, |x| x.ln_pmf(1));
        test_case(0.0, 3, 0.0, |x| x.ln_pmf(0));
        test_case(0.0, 3, f64::NEG_INFINITY, |x| x.ln_pmf(1));
        test_case(0.0, 3, f64::NEG_INFINITY, |x| x.ln_pmf(3));
        test_case(0.0, 10, 0.0, |x| x.ln_pmf(0));
        test_case(0.0, 10, f64::NEG_INFINITY, |x| x.ln_pmf(1));
        test_case(0.0, 10, f64::NEG_INFINITY, |x| x.ln_pmf(10));
        test_case(0.3, 1, -0.3566749439387324423539544041072745145718090708995, |x| x.ln_pmf(0));
        test_case(0.3, 1, -1.2039728043259360296301803719337238685164245381839, |x| x.ln_pmf(1));
        test_case(0.3, 3, -1.0700248318161973270618632123218235437154272126985, |x| x.ln_pmf(0));
        test_almost(0.3, 3, -0.81871040353529122294284394322574719301255212216016, 1e-15, |x| x.ln_pmf(1));
        test_almost(0.3, 3, -3.6119184129778080888905411158011716055492736145517, 1e-15, |x| x.ln_pmf(3));
        test_case(0.3, 10, -3.566749439387324423539544041072745145718090708995, |x| x.ln_pmf(0));
        test_almost(0.3, 10, -2.1114622067804823267977785542148302920616046876506, 1e-14, |x| x.ln_pmf(1));
        test_case(0.3, 10, -12.039728043259360296301803719337238685164245381839, |x| x.ln_pmf(10));
        test_case(1.0, 1, f64::NEG_INFINITY, |x| x.ln_pmf(0));
        test_case(1.0, 1, 0.0, |x| x.ln_pmf(1));
        test_case(1.0, 3, f64::NEG_INFINITY, |x| x.ln_pmf(0));
        test_case(1.0, 3, f64::NEG_INFINITY, |x| x.ln_pmf(1));
        test_case(1.0, 3, 0.0, |x| x.ln_pmf(3));
        test_case(1.0, 10, f64::NEG_INFINITY, |x| x.ln_pmf(0));
        test_case(1.0, 10, f64::NEG_INFINITY, |x| x.ln_pmf(1));
        test_case(1.0, 10, 0.0, |x| x.ln_pmf(10));
    }

    #[test]
    fn test_cdf() {
        test_case(0.0, 1, 1.0, |x| x.cdf(0.0));
        test_case(0.0, 1, 1.0, |x| x.cdf(1.0));
        test_case(0.0, 3, 1.0, |x| x.cdf(0.0));
        test_case(0.0, 3, 1.0, |x| x.cdf(1.0));
        test_case(0.0, 3, 1.0, |x| x.cdf(3.0));
        test_case(0.0, 10, 1.0, |x| x.cdf(0.0));
        test_case(0.0, 10, 1.0, |x| x.cdf(1.0));
        test_case(0.0, 10, 1.0, |x| x.cdf(10.0));
        test_almost(0.3, 1, 0.7, 1e-15, |x| x.cdf(0.0));
        test_case(0.3, 1, 1.0, |x| x.cdf(1.0));
        test_almost(0.3, 3, 0.343, 1e-14, |x| x.cdf(0.0));
        test_almost(0.3, 3, 0.784, 1e-15, |x| x.cdf(1.0));
        test_case(0.3, 3, 1.0, |x| x.cdf(3.0));
        test_almost(0.3, 10, 0.0282475249, 1e-16, |x| x.cdf(0.0));
        test_almost(0.3, 10, 0.1493083459, 1e-14, |x| x.cdf(1.0));
        test_case(0.3, 10, 1.0, |x| x.cdf(10.0));
        test_case(1.0, 1, 0.0, |x| x.cdf(0.0));
        test_case(1.0, 1, 1.0, |x| x.cdf(1.0));
        test_case(1.0, 3, 0.0, |x| x.cdf(0.0));
        test_case(1.0, 3, 0.0, |x| x.cdf(1.0));
        test_case(1.0, 3, 1.0, |x| x.cdf(3.0));
        test_case(1.0, 10, 0.0, |x| x.cdf(0.0));
        test_case(1.0, 10, 0.0, |x| x.cdf(1.0));
        test_case(1.0, 10, 1.0, |x| x.cdf(10.0));
    }
}
