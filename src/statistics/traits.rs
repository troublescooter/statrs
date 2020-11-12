use crate::Result;

/// The `Min` trait specifies than an object has a minimum value
pub trait Min<T> {
    /// Returns the minimum value in the domain of a given distribution
    /// if it exists, otherwise `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Min;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.0, n.min());
    /// ```
    fn min(&self) -> T;
}

/// The `Max` trait specifies that an object has a maximum value
pub trait Max<T> {
    /// Returns the maximum value in the domain of a given distribution
    /// if it exists, otherwise `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Max;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(Some(1.0), n.max());
    /// ```
    fn max(&self) -> Option<T>;
}

/// The `Mean` trait specifies that an object has a closed form
/// solution for its mean(s)
pub trait Mean<T> {
    /// Returns the mean, if it exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Mean;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(Some(0.5), n.mean());
    /// ```
    fn mean(&self) -> Option<T>;
}

pub trait StdDev<T>: Mean<T> {
    /// Returns the standard deviation, if it exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Variance;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(Some((1f64 / 12f64).sqrt()), n.std_dev());
    /// ```
    fn std_dev(&self) -> Option<T>;
}

impl<T, U> StdDev<T> for U
where
    U: Variance<T>,
    T: num_traits::float::Float,
{
    fn std_dev(&self) -> Option<T> {
        self.variance().map(|x| x.sqrt())
    }
}

impl<T, U> StdDev<Vec<T>> for U
where
    U: Variance<Vec<T>>,
    T: num_traits::float::Float,
{
    fn std_dev(&self) -> Option<Vec<T>> {
        let mut var = self.variance();
        var.map(|x| x.iter_mut().map(|v| *v = v.sqrt()).for_each(|_| {}));
        var
    }
}

/// The `Variance` trait specifies that an object has a closed form solution for
/// its variance(s). Requires `Mean` since a closed form solution to
/// variance by definition requires a closed form mean.
pub trait Variance<T>: Mean<T> {
    /// Returns the variance, if it exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Variance;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(Some(1.0 / 12.0), n.variance());
    /// ```
    fn variance(&self) -> Option<T>;
}

pub trait Covariance<T> {
    fn variance(&self) -> T;
}

/// The `Entropy` trait specifies an object that has a closed form solution
/// for its entropy
pub trait Entropy<T> {
    /// Returns the entropy, if it exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Entropy;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.0, n.entropy());
    /// ```
    fn entropy(&self) -> Option<T>;
}

/// The `Skewness` trait specifies an object that has a closed form solution
/// for its skewness(s)
pub trait Skewness<T> {
    /// Returns the skewness, if it exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Skewness;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.0, n.skewness());
    /// ```
    fn skewness(&self) -> Option<T>;
}

/// The `Median` trait specifies than an object has a closed form solution
/// for its median
pub trait Median<T> {
    /// Returns the median.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Median;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.5, n.median());
    /// ```
    fn median(&self) -> T;
}

/// The `Mode` trait specifies that an object has a closed form solution
/// for its mode(s)
pub trait Mode<T> {
    /// Returns the mode, if one exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Mode;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(Some(0.5), n.mode());
    /// ```
    fn mode(&self) -> Option<T>;
}
