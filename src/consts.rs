//! Defines mathematical expressions commonly used when computing
//! distribution values as constants

// Trait for floating point numeric type constants
pub trait FloatConst {
    /// Constant value for `sqrt(2 * pi)`
    fn SQRT_2PI() -> Self;

    /// Constant value for `ln(pi)`
    fn LN_PI() -> Self;

    /// Constant value for `ln(sqrt(2 * pi))`
    fn LN_SQRT_2PI() -> Self;

    /// Constant value for `ln(sqrt(2 * pi * e))`
    fn LN_SQRT_2PIE() -> Self;

    /// Constant value for `ln(2 * sqrt(e / pi))`
    fn LN_2_SQRT_E_OVER_PI() -> Self;

    /// Constant value for `2 * sqrt(e / pi)`
    fn TWO_SQRT_E_OVER_PI() -> Self;

    /// Constant value for Euler-Masheroni constant `lim(n -> inf) { sum(k=1 -> n) { 1/k - ln(n) } }`
    fn EULER_MASCHERONI() -> Self;
}

macro_rules! impl_float_const_for {
    ($T:ty) => (
        impl FloatConst for $T {
            fn SQRT_2PI() -> Self {
                SQRT_2PI as $T
            }

            fn LN_PI() -> Self {
                LN_PI as $T
            }

            fn LN_SQRT_2PI() -> Self {
                LN_SQRT_2PI as $T
            }

            fn LN_SQRT_2PIE() -> Self {
                LN_SQRT_2PIE as $T
            }

            fn LN_2_SQRT_E_OVER_PI() -> Self {
                LN_2_SQRT_E_OVER_PI as $T
            }

            fn TWO_SQRT_E_OVER_PI() -> Self {
                TWO_SQRT_E_OVER_PI as $T
            }

            fn EULER_MASCHERONI() -> Self {
                EULER_MASCHERONI as $T
            }
        }
    );
}
impl_float_const_for!(f64);
impl_float_const_for!(f32);

const SQRT_2PI: f64 = 2.5066282746310005024157652848110452530069867406099;
const LN_PI: f64 = 1.1447298858494001741434273513530587116472948129153;
const LN_SQRT_2PI: f64 = 0.91893853320467274178032973640561763986139747363778;
const LN_SQRT_2PIE: f64 = 1.4189385332046727417803297364056176398613974736378;
const LN_2_SQRT_E_OVER_PI: f64 = 0.6207822376352452223455184457816472122518527279025978;
const TWO_SQRT_E_OVER_PI: f64 = 1.8603827342052657173362492472666631120594218414085755;
const EULER_MASCHERONI: f64 = 0.5772156649015328606065120900824024310421593359399235988057672348849;
