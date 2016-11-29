use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use statistics::*;
use distribution::{Univariate, Discrete, Distribution, Binomial};
use result::Result;
use Float;

/// Implements the [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution)
/// distribution which is a special case of the [Binomial](https://en.wikipedia.org/wiki/Binomial_distribution)
/// distribution where `n = 1` (referenced [Here](./struct.Binomial.html))
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Bernoulli, Discrete};
/// use statrs::statistics::Mean;
///
/// let n = Bernoulli::new(0.5f64).unwrap();
/// assert_eq!(n.mean(), 0.5);
/// assert_eq!(n.pmf(0), 0.5);
/// assert_eq!(n.pmf(1), 0.5);
/// assert_eq!(n.pmf(2), 0.0);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Bernoulli<P>
    where P: Float
{
    b: Binomial<P, u32>,
}

impl<P> Bernoulli<P>
    where P: Float
{
    /// Constructs a new bernoulli distribution with
    /// the given `p` probability of success.
    ///
    /// # Errors
    ///
    /// Returns an error if `p` is `NaN`, less than `0.0`
    /// or greater than `1.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Bernoulli;
    /// use statrs::Result;
    ///
    /// let mut result = Bernoulli::new(0.5f64);
    /// assert!(result.is_ok());
    ///
    /// result = Bernoulli::new(-0.5f64);
    /// assert!(result.is_err());
    /// ```
    pub fn new(p: P) -> Result<Bernoulli<P>> {
        Binomial::new(p, 1u32).map(|b| Bernoulli { b: b })
    }

    /// Returns the probability of success `p` of the
    /// bernoulli distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Bernoulli;
    ///
    /// let n = Bernoulli::new(0.5f64).unwrap();
    /// assert_eq!(n.p(), 0.5);
    /// ```
    pub fn p(&self) -> P {
        self.b.p()
    }

    /// Returns the number of trials `n` of the
    /// bernoulli distribution. Will always be `1.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Bernoulli;
    ///
    /// let n = Bernoulli::new(0.5f64).unwrap();
    /// assert_eq!(n.n(), 1);
    /// ```
    pub fn n(&self) -> u32 {
        1u32
    }
}

impl<P> Sample<P> for Bernoulli<P>
    where P: Float
{
    /// Generate a random sample from a bernoulli
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn sample<R: Rng>(&mut self, r: &mut R) -> P {
        super::Distribution::sample(self, r)
    }
}

impl<P> IndependentSample<P> for Bernoulli<P>
    where P: Float
{
    /// Generate a random independent sample from a bernoulli
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn ind_sample<R: Rng>(&self, r: &mut R) -> P {
        super::Distribution::sample(self, r)
    }
}

impl<P> Distribution<P> for Bernoulli<P>
    where P: Float
{
    /// Generate a random sample from the
    /// bernoulli distribution using `r` as the source
    /// of randomness where the generated
    /// values are `1` with probability `p` and `0`
    /// with probability `1-p`.
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate rand;
    /// # extern crate statrs;
    /// use rand::StdRng;
    /// use statrs::distribution::{Bernoulli, Distribution};
    ///
    /// # fn main() {
    /// let mut r = rand::StdRng::new().unwrap();
    /// let n = Bernoulli::new(0.5f64).unwrap();
    /// print!("{}", n.sample::<StdRng>(&mut r));
    /// # }
    /// ```
    fn sample<R: Rng>(&self, r: &mut R) -> P {
        self.b.sample(r)
    }
}

impl<P> Univariate<u32, P> for Bernoulli<P>
    where P: Float
{
    /// Calculates the cumulative distribution
    /// function for the bernoulli distribution at `x`.
    ///
    /// # Remarks
    ///
    /// Returns `0.0` if `x < 0.0` and `1.0` if `x >= 1.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if x < 0 { 0 }
    /// else if x >= 1 { 1 }
    /// else { 1 - p }
    /// ```
    fn cdf(&self, x: P) -> P {
        self.b.cdf(x)
    }
}

impl<P> Min<u32> for Bernoulli<P>
    where P: Float
{
    /// Returns the minimum value in the domain of the
    /// bernoulli distribution representable by a 32-
    /// bit unsigned integer
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 0
    /// ```
    fn min(&self) -> u32 {
        0u32
    }
}

impl<P> Max<u32> for Bernoulli<P>
    where P: Float
{
    /// Returns the maximum value in the domain of the
    /// bernoulli distribution representable by a 32-
    /// bit unsigned integer
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1
    /// ```
    fn max(&self) -> u32 {
        1u32
    }
}

impl<P> Mean<P> for Bernoulli<P>
    where P: Float
{
    /// Returns the mean of the bernoulli
    /// distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// p
    /// ```
    fn mean(&self) -> P {
        self.b.mean()
    }
}

impl<P> Variance<P> for Bernoulli<P>
    where P: Float
{
    /// Returns the variance of the bernoulli
    /// distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// p * (1 - p)
    /// ```
    fn variance(&self) -> P {
        self.b.variance()
    }

    /// Returns the standard deviation of the bernoulli
    /// distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(p * (1 - p))
    /// ```
    fn std_dev(&self) -> P {
        self.b.std_dev()
    }
}

impl<P> Entropy<P> for Bernoulli<P>
    where P: Float
{
    /// Returns the entropy of the bernoulli
    /// distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// q = (1 - p)
    /// -q * ln(q) - p * ln(p)
    /// ```
    fn entropy(&self) -> P {
        self.b.entropy()
    }
}

impl<P> Skewness<P> for Bernoulli<P>
    where P: Float
{
    /// Returns the skewness of the bernoulli
    /// distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// q = (1 - p)
    /// (1 - 2p) / sqrt(p * q)
    /// ```
    fn skewness(&self) -> P {
        self.b.skewness()
    }
}

impl<P> Median<P> for Bernoulli<P>
    where P: Float
{
    /// Returns the median of the bernoulli
    /// distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if p < 0.5 { 0 }
    /// else if p > 0.5 { 1 }
    /// else { 0.5 }
    /// ```
    fn median(&self) -> P {
        self.b.median()
    }
}

impl<P> Mode<u32> for Bernoulli<P>
    where P: Float
{
    /// Returns the mode of the bernoulli distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if p < 0.5 { 0 }
    /// else { 1 }
    /// ```
    fn mode(&self) -> u32 {
        self.b.mode()
    }
}

impl<P> Discrete<u32, P> for Bernoulli<P>
    where P: Float
{
    /// Calculates the probability mass function for the
    /// bernoulli distribution at `x`.
    ///
    /// # Remarks
    ///
    /// Returns `0.0` if `x < 0 || x > 1`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if x < 0 || x > 1 { 0.0 }
    /// else if x == 0 { 1 - p }
    /// else { p }
    /// ```
    fn pmf(&self, x: u32) -> P {
        self.b.pmf(x)
    }

    /// Calculates the log probability mass function for the
    /// bernoulli distribution at `x`.
    ///
    /// # Remarks
    ///
    /// Returns negative infinity if `x < 0 || x > 1`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if x < 0 || x > 1 { -INF }
    /// else if x == 0 { ln(1 - p) }
    /// else { ln(p) }
    /// ```
    fn ln_pmf(&self, x: u32) -> P {
        self.b.ln_pmf(x)
    }
}
