use distribution::{Continuous, Distribution, Univariate, WeakRngDistribution};
use rand::distributions::{IndependentSample, Sample};
use rand::Rng;
use signum;
use {Result, StatsError};

/// Implements the [Laplace](https://en.wikipedia.org/wiki/Laplace_distribution)
/// distribution, sometimes known as the double exponential distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Laplace, Continuous};
/// use statrs::statistics::Mean;
///
/// // TODO
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Laplace {
    location: f64,
    scale: f64,
}

impl Laplace {
    /// Constructs a new laplace distribution with
    /// a location (μ) of `location` and a scale (b) of `scale`
    ///
    /// # Errors
    ///
    /// Returns an error if `location` or `scale` are `NaN`.
    /// Also returns an error if `scale <= 0.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Laplace;
    ///
    /// let mut result = Laplace::new(0.0, 1.0);
    /// assert!(result.is_ok());
    ///
    /// result = Laplace::new(0.0, 0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(location: f64, scale: f64) -> Result<Laplace> {
        let is_nan = location.is_nan() || scale.is_nan();
        match (location, scale, is_nan) {
            (_, _, true) => Err(StatsError::BadParams),
            (_, _, false) if scale <= 0.0 => Err(StatsError::BadParams),
            (_, _, false) => Ok(Laplace {
                location: location,
                scale: scale,
            }),
        }
    }

    /// Returns the location (μ) of the laplace distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Laplace;
    ///
    /// let n = Laplace::new(0.0, 1.0).unwrap();
    /// assert_eq!(n.location(), 0.0);
    /// ```
    pub fn location(&self) -> f64 {
        self.location
    }

    /// Returns the scale (b) of the laplace distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Laplace;
    ///
    /// let n = Laplace::new(0.0, 1.0).unwrap();
    /// assert_eq!(n.scale(), 1.0);
    /// ```
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Sample<f64> for Laplace {
    /// Generate a random sample from a laplace
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn sample<R: Rng>(&mut self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl IndependentSample<f64> for Laplace {
    /// Generate a random independent sample from a laplace
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn ind_sample<R: Rng>(&self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl Distribution<f64> for Laplace {
    /// Generate a random sample from a laplace distribution using
    /// `r` as the source of randomness.
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate rand;
    /// # extern crate statrs;
    /// use rand::StdRng;
    /// use statrs::distribution::{Laplace, Distribution};
    ///
    /// # fn main() {
    /// let mut r = rand::StdRng::new().unwrap();
    /// let n = Laplace::new(0.0, 1.0).unwrap();
    /// print!("{}", n.sample::<StdRng>(&mut r));
    /// # }
    /// ```
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        let u = r.next_f64() - 0.5;
        self.location - (self.scale * signum(u) * (1.0 - 2.0 * u.abs()).ln())
    }
}

impl WeakRngDistribution<f64> for Laplace {}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::f64;
    use distribution::Laplace;

    fn try_create(location: f64, scale: f64) -> Laplace {
        let n = Laplace::new(location, scale);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(location: f64, scale: f64) {
        let n = try_create(location, scale);
        assert_eq!(location, n.location());
        assert_eq!(scale, n.scale());
    }

    #[test]
    fn test_create() {
        create_case(f64::NEG_INFINITY, 0.1);
        create_case(-5.0, 1.0);
        create_case(0.0, 5.0);
        create_case(1.0, 7.0);
        create_case(5.0, 10.0);
    }
}
