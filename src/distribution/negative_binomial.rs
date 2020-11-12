use crate::distribution::{self, poisson, Discrete, Univariate};
use crate::function::{beta, gamma};
use crate::statistics::*;
use crate::{Result, StatsError};
use rand::distributions::Distribution;
use rand::Rng;
use std::f64;

/// Implements the
/// [NegativeBinomial](http://en.wikipedia.org/wiki/Negative_binomial_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{NegativeBinomial, Discrete};
/// use statrs::statistics::Mean;
/// use statrs::prec::{almost_eq};
///
/// let r = NegativeBinomial::new(4.0, 0.5).unwrap();
/// assert_eq!(r.mean(), 4.0);
/// assert!(almost_eq(r.pmf(0), 0.0625, 1e-8));
/// assert!(almost_eq(r.pmf(3), 0.15625, 1e-8));
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct NegativeBinomial {
    r: f64,
    p: f64,
}

impl NegativeBinomial {
    /// Constructs a new negative binomial distribution
    /// with a given `p` probability of the number of successes `r`
    ///
    /// # Errors
    ///
    /// Returns an error if `p` is `NaN`, less than `0.0`,
    /// greater than `1.0`, or if `r` is `NaN` or less than `0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::NegativeBinomial;
    ///
    /// let mut result = NegativeBinomial::new(4.0, 0.5);
    /// assert!(result.is_ok());
    ///
    /// result = NegativeBinomial::new(-0.5, 5.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(r: f64, p: f64) -> Result<NegativeBinomial> {
        if p.is_nan() || p < 0.0 || p > 1.0 || r.is_nan() || r < 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(NegativeBinomial { p, r })
        }
    }

    /// Returns the probability of success `p` of
    /// the negative binomial distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::NegativeBinomial;
    ///
    /// let r = NegativeBinomial::new(5.0, 0.5).unwrap();
    /// assert_eq!(r.p(), 0.5);
    /// ```
    pub fn p(&self) -> f64 {
        self.p
    }

    /// Returns the number `r` of success of this negative
    /// binomial distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::NegativeBinomial;
    ///
    /// let r = NegativeBinomial::new(5.0, 0.5).unwrap();
    /// assert_eq!(r.r(), 5.0);
    /// ```
    pub fn r(&self) -> f64 {
        self.r
    }
}

impl Distribution<u64> for NegativeBinomial {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> u64 {
        let lambda = distribution::gamma::sample_unchecked(r, self.r, (1.0 - self.p) / self.p);
        poisson::sample_unchecked(r, lambda).floor() as u64
    }
}

impl Univariate<u64, f64> for NegativeBinomial {
    /// Calculates the cumulative distribution function for the
    /// negative binomial distribution at `x`
    ///
    /// Note that due to extending the distribution to the reals
    /// (allowing positive real values for `r`), while still technically
    /// a discrete distribution the CDF behaves more like that of a
    /// continuous distribution rather than a discrete distribution
    /// (i.e. a smooth graph rather than a step-ladder)
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1 - I_(1 - p)(x + 1, r)
    /// ```
    ///
    /// where `I_(x)(a, b)` is the regularized incomplete beta function
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else if x.is_infinite() {
            1.0
        } else {
            1.0 - beta::beta_reg(x + 1.0, self.r, 1.0 - self.p)
        }
    }
}

impl Min<u64> for NegativeBinomial {
    /// Returns the minimum value in the domain of the
    /// negative binomial distribution representable by a 64-bit
    /// integer
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 0
    /// ```
    fn min(&self) -> u64 {
        0
    }
}

impl Max<u64> for NegativeBinomial {
    /// Returns the maximum value in the domain of the
    /// negative binomial distribution representable by a 64-bit
    /// integer
    ///
    /// # Formula
    ///
    /// ```ignore
    /// u64::MAX
    /// ```
    fn max(&self) -> u64 {
        std::u64::MAX
    }
}

impl Mean<f64> for NegativeBinomial {
    /// Returns the mean of the negative binomial distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// r * (1-p) / p
    /// ```
    fn mean(&self) -> Option<f64> {
        Some(self.r * (1.0 - self.p) / self.p)
    }
}

impl Variance<f64> for NegativeBinomial {
    /// Returns the variance of the negative binomial distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// r * (1-p) / p^2
    /// ```
    fn variance(&self) -> Option<f64> {
        Some(self.r * (1.0 - self.p) / (self.p * self.p))
    }
}

impl Skewness<f64> for NegativeBinomial {
    /// Returns the skewness of the negative binomial distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (2-p) / sqrt(r * (1-p))
    /// ```
    fn skewness(&self) -> Option<f64> {
        Some((2.0 - self.p) / f64::sqrt(self.r * (1.0 - self.p)))
    }
}

impl Mode<Option<f64>> for NegativeBinomial {
    /// Returns the mode for the negative binomial distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if r > 1 then
    ///     floor((r - 1) * (1-p / p))
    /// else
    ///     0
    /// ```
    fn mode(&self) -> Option<f64> {
        let mode = if self.r > 1.0 {
            f64::floor((self.r - 1.0) * (1.0 - self.p) / self.p)
        } else {
            0.0
        };
        Some(mode)
    }
}

impl Discrete<u64, f64> for NegativeBinomial {
    /// Calculates the probability mass function for the negative binomial
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (x + r - 1 choose k) * (1 - p)^x * p^r
    /// ```
    fn pmf(&self, x: u64) -> f64 {
        self.ln_pmf(x).exp()
    }

    /// Calculates the log probability mass function for the negative binomial
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(x + r - 1 choose k) * (1 - p)^x * p^r))
    /// ```
    fn ln_pmf(&self, x: u64) -> f64 {
        let k = x as f64;
        gamma::ln_gamma(self.r + k) - gamma::ln_gamma(self.r) - gamma::ln_gamma(k + 1.0)
            + (self.r * self.p.ln())
            + (k * (1.0 - self.p).ln())
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use std::fmt::Debug;
    use std::f64;
    use crate::statistics::*;
    use crate::distribution::{Univariate, Discrete, NegativeBinomial};
    // use crate::distribution::internal::*;

    fn try_create(r: f64, p: f64) -> NegativeBinomial {
        let r = NegativeBinomial::new(r, p);
        assert!(r.is_ok());
        r.unwrap()
    }

    fn create_case(r: f64, p: f64) {
        let dist = try_create(r, p);
        assert_eq!(p, dist.p());
        assert_eq!(r, dist.r());
    }

    fn bad_create_case(r: f64, p: f64) {
        let r = NegativeBinomial::new(r, p);
        assert!(r.is_err());
    }

    fn get_value<T, F>(r: f64, p: f64, eval: F) -> T
        where T: PartialEq + Debug,
                F: Fn(NegativeBinomial) -> T
    {
        let r = try_create(r, p);
        eval(r)
    }

    fn test_case<T, F>(r: f64, p: f64, expected: T, eval: F)
        where T: PartialEq + Debug,
                F: Fn(NegativeBinomial) -> T
    {
        let x = get_value(r, p, eval);
        assert_eq!(expected, x);
    }


    fn test_case_or_nan<F>(r: f64, p: f64, expected: f64, eval: F)
        where F: Fn(NegativeBinomial) -> f64
    {
        let x = get_value(r, p, eval);
        if expected.is_nan() {
            assert!(x.is_nan())
        }
        else {
            assert_eq!(expected, x);
        }
    }
    fn test_almost<F>(r: f64, p: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(NegativeBinomial) -> f64
    {
        let x = get_value(r, p, eval);
        assert_almost_eq!(expected, x, acc);
    }

    #[test]
    fn test_create() {
        create_case(0.0, 0.0);
        create_case(0.3, 0.4);
        create_case(1.0, 0.3);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(f64::NAN, 1.0);
        bad_create_case(0.0, f64::NAN);
        bad_create_case(-1.0, 1.0);
        bad_create_case(2.0, 2.0);
    }

    #[test]
    fn test_mean() {
        test_case(4.0, 0.0, f64::INFINITY, |x| x.mean());
        test_almost(3.0, 0.3, 7.0, 1e-15 , |x| x.mean());
        test_case(2.0, 1.0, 0.0, |x| x.mean());
    }

    #[test]
    fn test_variance() {
        test_case(4.0, 0.0, f64::INFINITY, |x| x.variance());
        test_almost(3.0, 0.3, 23.333333333333, 1e-12, |x| x.variance());
        test_case(2.0, 1.0, 0.0, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_case(4.0, 0.0, f64::INFINITY, |x| x.std_dev());
        test_almost(3.0, 0.3, 4.830458915, 1e-9, |x| x.std_dev());
        test_case(2.0, 1.0, 0.0, |x| x.std_dev());
    }

    #[test]
    fn test_skewness() {
        test_case(0.0, 0.0, f64::INFINITY, |x| x.skewness());
        test_almost(0.1, 0.3, 6.425396041, 1e-09, |x| x.skewness());
        test_case(1.0, 1.0, f64::INFINITY, |x| x.skewness());
    }

    #[test]
    fn test_mode() {
        test_case(0.0, 0.0, 0.0, |x| x.mode());
        test_case(0.3, 0.0, 0.0, |x| x.mode());
        test_case(1.0, 1.0, 0.0, |x| x.mode());
        test_case(10.0, 0.01, 891.0, |x| x.mode());
    }

    #[test]
    fn test_min_max() {
        test_case(1.0, 0.5, 0, |x| x.min());
        test_case(1.0, 0.3, std::u64::MAX, |x| x.max());
    }

    #[test]
    fn test_pmf() {
        test_almost(4.0, 0.5, 0.0625, 1e-8, |x| x.pmf(0));
        test_almost(4.0, 0.5, 0.15625, 1e-8, |x| x.pmf(3));
        test_case(1.0, 0.0, 0.0, |x| x.pmf(0));
        test_case(1.0, 0.0, 0.0, |x| x.pmf(1));
        test_almost(3.0, 0.2, 0.008, 1e-15, |x| x.pmf(0));
        test_almost(3.0, 0.2, 0.0192, 1e-15, |x| x.pmf(1));
        test_almost(3.0, 0.2, 0.04096, 1e-15, |x| x.pmf(3));
        test_almost(10.0, 0.2, 1.024e-07, 1e-07, |x| x.pmf(0));
        test_almost(10.0, 0.2, 8.192e-07, 1e-07, |x| x.pmf(1));
        test_almost(10.0, 0.2, 0.001015706852, 1e-07, |x| x.pmf(10));
        test_almost(1.0, 0.3, 0.3, 1e-15,  |x| x.pmf(0));
        test_almost(1.0, 0.3, 0.21, 1e-15, |x| x.pmf(1));
        test_almost(3.0, 0.3, 0.027, 1e-15, |x| x.pmf(0));
        test_case(0.3, 1.0, 0.0, |x| x.pmf(1));
        test_case(0.3, 1.0, 0.0, |x| x.pmf(3));
        test_case_or_nan(0.3, 1.0, f64::NAN, |x| x.pmf(0));
        test_case(0.3, 1.0, 0.0, |x| x.pmf(1));
        test_case(0.3, 1.0, 0.0, |x| x.pmf(10));
        test_case_or_nan(1.0, 1.0, f64::NAN, |x| x.pmf(0));
        test_case(1.0, 1.0, 0.0, |x| x.pmf(1));
        test_case_or_nan(3.0, 1.0, f64::NAN, |x| x.pmf(0));
        test_case(3.0, 1.0, 0.0, |x| x.pmf(1));
        test_case(3.0, 1.0, 0.0, |x| x.pmf(3));
        test_case_or_nan(10.0, 1.0, f64::NAN, |x| x.pmf(0));
        test_case(10.0, 1.0, 0.0, |x| x.pmf(1));
        test_case(10.0, 1.0, 0.0, |x| x.pmf(10));
    }

    #[test]
    fn test_ln_pmf() {
        test_case(1.0, 0.0, f64::NEG_INFINITY, |x| x.ln_pmf(0));
        test_case(1.0, 0.0, f64::NEG_INFINITY, |x| x.ln_pmf(1));
        test_almost(3.0, 0.2, -4.828313737, 1e-08, |x| x.ln_pmf(0));
        test_almost(3.0, 0.2, -3.952845, 1e-08, |x| x.ln_pmf(1));
        test_almost(3.0, 0.2, -3.195159298, 1e-08, |x| x.ln_pmf(3));
        test_almost(10.0, 0.2, -16.09437912, 1e-08, |x| x.ln_pmf(0));
        test_almost(10.0, 0.2, -14.01493758, 1e-08, |x| x.ln_pmf(1));
        test_almost(10.0, 0.2, -6.892170503, 1e-08, |x| x.ln_pmf(10));
        test_almost(1.0, 0.3, -1.203972804, 1e-08,  |x| x.ln_pmf(0));
        test_almost(1.0, 0.3, -1.560647748, 1e-08, |x| x.ln_pmf(1));
        test_almost(3.0, 0.3, -3.611918413, 1e-08, |x| x.ln_pmf(0));
        test_case(0.3, 1.0, f64::NEG_INFINITY, |x| x.ln_pmf(1));
        test_case(0.3, 1.0, f64::NEG_INFINITY, |x| x.ln_pmf(3));
        test_case_or_nan(0.3, 1.0, f64::NAN, |x| x.ln_pmf(0));
        test_case(0.3, 1.0, f64::NEG_INFINITY, |x| x.ln_pmf(1));
        test_case(0.3, 1.0, f64::NEG_INFINITY, |x| x.ln_pmf(10));
        test_case_or_nan(1.0, 1.0, f64::NAN, |x| x.ln_pmf(0));
        test_case(1.0, 1.0, f64::NEG_INFINITY, |x| x.ln_pmf(1));
        test_case_or_nan(3.0, 1.0, f64::NAN, |x| x.ln_pmf(0));
        test_case(3.0, 1.0, f64::NEG_INFINITY, |x| x.ln_pmf(1));
        test_case(3.0, 1.0, f64::NEG_INFINITY, |x| x.ln_pmf(3));
        test_case_or_nan(10.0, 1.0, f64::NAN, |x| x.ln_pmf(0));
        test_case(10.0, 1.0, f64::NEG_INFINITY, |x| x.ln_pmf(1));
        test_case(10.0, 1.0, f64::NEG_INFINITY, |x| x.ln_pmf(10));
    }

    #[test]
    fn test_cdf() {
        test_case(1.0, 0.0, 0.0, |x| x.cdf(0.2));
        test_almost(3.0, 0.2, 0.01090199062, 1e-08, |x| x.cdf(0.2));
        test_almost(10.0, 0.2, 1.718008933e-07, 1e-08, |x| x.cdf(0.2));
        test_almost(1.0, 0.3, 0.3481950594, 1e-08, |x| x.cdf(0.2));
        test_almost(3.0, 0.3, 0.03611085389, 1e-08, |x| x.cdf(0.2));
        test_almost(1.0, 0.3, 0.3, 1e-08, |x| x.cdf(0.0));
        test_almost(1.0, 0.3, 0.3481950594, 1e-08, |x| x.cdf(0.2));
        test_almost(1.0, 0.3, 0.51, 1e-08, |x| x.cdf(1.0));
        test_almost(1.0, 0.3, 0.83193, 1e-08, |x| x.cdf(4.0));
        test_almost(1.0, 0.3, 0.9802267326, 1e-08, |x| x.cdf(10.0));
        test_case(1.0, 1.0, 1.0, |x| x.cdf(0.0));
        test_case(1.0, 1.0, 1.0, |x| x.cdf(1.0));
        test_almost(10.0, 0.75, 0.05631351471, 1e-08, |x| x.cdf(0.0));
        test_almost(10.0, 0.75, 0.1970973015, 1e-08, |x| x.cdf(1.0));
        test_almost(10.0, 0.75, 0.9960578583, 1e-08, |x| x.cdf(10.0));
    }

    #[test]
    fn test_cdf_lower_bound() {
        test_case(3.0, 0.5, 0.0, |x| x.cdf(-1.0));
    }

    #[test]
    fn test_cdf_upper_bound() {
        test_case(3.0, 0.5, 1.0, |x| x.cdf(100.0));
    }

    // TODO: figure out the best way to re-implement this test. We currently
    // do not have a good way to characterize a discrete distribution with a
    // CDF that is continuous
    //
    // #[test]
    // fn test_discrete() {
    //     tests::check_discrete_distribution(&try_create(5.0, 0.3), 35);
    //     tests::check_discrete_distribution(&try_create(10.0, 0.7), 21);
    // }
}
