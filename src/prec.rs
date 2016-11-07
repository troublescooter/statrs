//! Provides utility functions for working with floating point precision

use Float;

/// A trait for numeric types with a certain finite precision
/// and accuracy
pub trait Precision {
    fn precision() -> Self;
    fn accuracy() -> Self;
}

impl Precision for f64 {
    fn precision() -> f64 {
        2f64.powi(-53)
    }

    fn accuracy() -> f64 {
        10f64 * Self::precision()
    }
}

impl Precision for f32 {
    fn precision() -> f32 {
        2f32.powi(-24)
    }

    fn accuracy() -> f32 {
        10f32 * Self::precision()
    }
}

/// Returns true if `a` and `b `are within `acc` of each other.
/// If `a` or `b` are infinite, returns `true` only if both are
/// infinite and similarly signed. Always returns `false` if
/// either number is a `NAN`.
pub fn almost_eq<T>(a: T, b: T, acc: T) -> bool
    where T: Float
{
    // only true if a and b are infinite with same
    // sign
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }

    // NANs are never equal
    if a.is_nan() && b.is_nan() {
        return false;
    }

    (a - b).abs() < acc
}
