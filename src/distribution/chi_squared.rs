use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use statistics::*;
use distribution::{Univariate, Continuous, Distribution, Gamma};
use result::Result;
use Float;

/// Implements the [Chi-squared](https://en.wikipedia.org/wiki/Chi-squared_distribution)
/// distribution which is a special case of the [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) distribution
/// (referenced [Here](./struct.Gamma.html))
///
/// # Examples
///
/// ```
/// use statrs::distribution::{ChiSquared, Continuous};
/// use statrs::statistics::Mean;
/// use statrs::prec;
///
/// let n = ChiSquared::new(3.0).unwrap();
/// assert_eq!(n.mean(), 3.0);
/// assert!(prec::almost_eq(n.pdf(4.0), 0.107981933026376103901, 1e-15));
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ChiSquared<T>
    where T: Float
{
    freedom: T,
    g: Gamma<T>,
}

impl<T> ChiSquared<T>
    where T: Float
{
    /// Constructs a new chi-squared distribution with `freedom`
    /// degrees of freedom. This is equivalent to a Gamma distribution
    /// with a shape of `freedom / 2.0` and a rate of `0.5`.
    ///
    /// # Errors
    ///
    /// Returns an error if `freedom` is `NaN` or less than
    /// or equal to `0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::ChiSquared;
    ///
    /// let mut result = ChiSquared::new(3.0);
    /// assert!(result.is_ok());
    ///
    /// result = ChiSquared::new(0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(freedom: T) -> Result<ChiSquared<T>> {
        Gamma::new(freedom / T::from(2.0).unwrap(), T::from(0.5).unwrap()).map(|g| {
            ChiSquared {
                freedom: freedom,
                g: g,
            }
        })
    }

    /// Returns the degrees of freedom of the chi-squared
    /// distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::ChiSquared;
    ///
    /// let n = ChiSquared::new(3.0).unwrap();
    /// assert_eq!(n.freedom(), 3.0);
    /// ```
    pub fn freedom(&self) -> T {
        self.freedom
    }

    /// Returns the shape of the underlying Gamma distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::ChiSquared;
    ///
    /// let n = ChiSquared::new(3.0).unwrap();
    /// assert_eq!(n.shape(), 3.0 / 2.0);
    /// ```
    pub fn shape(&self) -> T {
        self.g.shape()
    }

    /// Returns the rate of the underlying Gamma distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::ChiSquared;
    ///
    /// let n = ChiSquared::new(3.0).unwrap();
    /// assert_eq!(n.rate(), 0.5);
    /// ```
    pub fn rate(&self) -> T {
        self.g.rate()
    }
}

impl<T> Sample<T> for ChiSquared<T>
    where T: Float
{
    /// Generate a random sample from a chi-squared
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn sample<R: Rng>(&mut self, r: &mut R) -> T {
        super::Distribution::sample(self, r)
    }
}

impl<T> IndependentSample<T> for ChiSquared<T>
    where T: Float
{
    /// Generate a random independent sample from a Chi
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn ind_sample<R: Rng>(&self, r: &mut R) -> T {
        super::Distribution::sample(self, r)
    }
}

impl<T> Distribution<T> for ChiSquared<T>
    where T: Float
{
    /// Generate a random sample from the chi-squared distribution
    /// using `r` as the source of randomness
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate rand;
    /// # extern crate statrs;
    /// use rand::StdRng;
    /// use statrs::distribution::{ChiSquared, Distribution};
    ///
    /// # fn main() {
    /// let mut r = rand::StdRng::new().unwrap();
    /// let n = ChiSquared::new(3.0).unwrap();
    /// print!("{}", n.sample::<StdRng>(&mut r));
    /// # }
    /// ```
    fn sample<R: Rng>(&self, r: &mut R) -> T {
        self.g.sample(r)
    }
}

impl<T> Univariate<T, T> for ChiSquared<T>
    where T: Float
{
    /// Calculates the cumulative distribution function for the
    /// chi-squared distribution at `x`
    ///
    /// # Panics
    ///
    /// If `x < 0.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 / Γ(k / 2)) * γ(k / 2, x / 2)
    /// ```
    ///
    /// where `k` is the degrees of freedom, `Γ` is the gamma function,
    /// and `γ` is the lower incomplete gamma function
    fn cdf(&self, x: T) -> T {
        self.g.cdf(x)
    }
}

impl<T> Min<T> for ChiSquared<T>
    where T: Float
{
    /// Returns the minimum value in the domain of the
    /// chi-squared distribution representable by a double precision
    /// float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 0
    /// ```
    fn min(&self) -> T {
        T::zero()
    }
}

impl<T> Max<T> for ChiSquared<T>
    where T: Float
{
    /// Returns the maximum value in the domain of the
    /// chi-squared distribution representable by a double precision
    /// float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// INF
    /// ```
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T> Mean<T> for ChiSquared<T>
    where T: Float
{
    /// Returns the mean of the chi-squared distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// k
    /// ```
    ///
    /// where `k` is the degrees of freedom
    fn mean(&self) -> T {
        self.g.mean()
    }
}

impl<T> Variance<T> for ChiSquared<T>
    where T: Float
{
    /// Returns the variance of the chi-squared distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 2k
    /// ```
    ///
    /// where `k` is the degrees of freedom
    fn variance(&self) -> T {
        self.g.variance()
    }

    /// Returns the standard deviation of the chi-squared distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(2k)
    /// ```
    ///
    /// where `k` is the degrees of freedom
    fn std_dev(&self) -> T {
        self.g.std_dev()
    }
}

impl<T> Entropy<T> for ChiSquared<T>
    where T: Float
{
    /// Returns the entropy of the chi-squared distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (k / 2) + ln(2 * Γ(k / 2)) + (1 - (k / 2)) * ψ(k / 2)
    /// ```
    ///
    /// where `k` is the degrees of freedom, `Γ` is the gamma function,
    /// and `ψ` is the digamma function
    fn entropy(&self) -> T {
        self.g.entropy()
    }
}

impl<T> Skewness<T> for ChiSquared<T>
    where T: Float
{
    /// Returns the skewness of the chi-squared distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(8 / k)
    /// ```
    ///
    /// where `k` is the degrees of freedom
    fn skewness(&self) -> T {
        self.g.skewness()
    }
}

impl<T> Median<T> for ChiSquared<T>
    where T: Float
{
    /// Returns the median  of the chi-squared distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// k * (1 - (2 / 9k))^3
    /// ```
    fn median(&self) -> T {
        if self.freedom < T::one() {
            // if k is small, calculate using expansion of formula
            self.freedom - T::from(2.0 / 3.0).unwrap() +
            T::from(12.0).unwrap() / (T::from(81.0).unwrap() * self.freedom) -
            T::from(8.0).unwrap() / (T::from(729.0).unwrap() * self.freedom * self.freedom)
        } else {
            // if k is large enough, median heads toward k - 2/3
            self.freedom - T::from(2.0 / 3.0).unwrap()
        }
    }
}

impl<T> Mode<T> for ChiSquared<T>
    where T: Float
{
    /// Returns the mode of the chi-squared distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// k - 2
    /// ```
    ///
    /// where `k` is the degrees of freedom
    fn mode(&self) -> T {
        self.g.mode()
    }
}

impl<T> Continuous<T, T> for ChiSquared<T>
    where T: Float
{
    /// Calculates the probability density function for the chi-squared
    /// distribution at `x`
    ///
    /// # Panics
    ///
    /// If `x < 0.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1 / (2^(k / 2) * Γ(k / 2)) * x^((k / 2) - 1) * e^(-x / 2)
    /// ```
    ///
    /// where `k` is the degrees of freedom and `Γ` is the gamma function
    fn pdf(&self, x: T) -> T {
        self.g.pdf(x)
    }

    /// Calculates the log probability density function for the chi-squared
    /// distribution at `x`
    ///
    /// # Panics
    ///
    /// If `x < 0.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(1 / (2^(k / 2) * Γ(k / 2)) * x^((k / 2) - 1) * e^(-x / 2))
    /// ```
    fn ln_pdf(&self, x: T) -> T {
        self.g.ln_pdf(x)
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use statistics::Median;
    use distribution::ChiSquared;

    fn try_create(freedom: f64) -> ChiSquared<f64> {
        let n = ChiSquared::new(freedom);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn test_case<F>(freedom: f64, expected: f64, eval: F)
        where F: Fn(ChiSquared<f64>) -> f64
    {
        let n = try_create(freedom);
        let x = eval(n);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(freedom: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(ChiSquared<f64>) -> f64
    {
        let n = try_create(freedom);
        let x = eval(n);
        assert_almost_eq!(expected, x, acc);
    }

    #[test]
    fn test_median() {
        test_almost(0.5, 0.0857338820301783264746, 1e-16, |x| x.median());
        test_case(1.0, 1.0 - 2.0 / 3.0, |x| x.median());
        test_case(2.0, 2.0 - 2.0 / 3.0, |x| x.median());
        test_case(2.5, 2.5 - 2.0 / 3.0, |x| x.median());
        test_case(3.0, 3.0 - 2.0 / 3.0, |x| x.median());
    }
}
