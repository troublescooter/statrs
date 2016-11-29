//! This crate aims to be a functional
//! port of the Math.NET Numerics Distribution package and in doing so providing the Rust numerical
//! computing community with a robust, well-tested statistical distribution package. This crate
//! also ports over some of the special statistical functions from Math.NET in so far as they are
//! used in the computation of distribution values. This crate depends on the `rand` crate to provide
//! RNG.
//!
//! # Example
//! The following example samples from a standard normal distribution
//!
//! ```
//! # extern crate rand;
//! # extern crate statrs;

//! use rand::StdRng;
//! use statrs::distribution::{Distribution, Normal};
//!
//! # fn main() {
//! let mut r = rand::StdRng::new().unwrap();
//! let n = Normal::new(0.0, 1.0).unwrap();
//! for _ in 0..10 {
//!     print!("{}", n.sample::<StdRng>(&mut r));
//! }
//! # }
//! ```

#![crate_type = "lib"]
#![crate_name = "statrs"]

extern crate rand;
extern crate num;

#[macro_export]
macro_rules! assert_almost_eq {
    ($a:expr, $b:expr, $prec:expr) => (
        if !$crate::prec::almost_eq($a, $b, $prec) {
            panic!(format!("assertion failed: `abs(left - right) < {:e}`, (left: `{}`, right: `{}`)", $prec, $a, $b));
        }
    );
}

pub mod distribution;
pub mod euclid;
pub mod function;
pub mod generate;
pub mod consts;
pub mod prec;
pub mod statistics;

mod result;
mod error;

#[cfg(test)]
mod testing;

pub use result::Result;
pub use error::StatsError;

/// Float is a wrapper trait for generic floating point types used
/// by statrs.
pub trait Float
    : num::Float + num::traits::FloatConst + consts::FloatConst + prec::Precision + function::factorial::Binomial + NumBase
    {
}

impl Float for f64 {}
impl Float for f32 {}

/// Signed is a wrapper trait for generic signed integer types
pub trait Signed: Integer + num::Signed {}

impl Signed for i8 {}
impl Signed for i16 {}
impl Signed for i32 {}
impl Signed for i64 {}

/// Unsigned is a wrapper trait for generic unsigned integer types
pub trait Unsigned: Integer + num::Unsigned {}

impl Unsigned for u8 {}
impl Unsigned for u16 {}
impl Unsigned for u32 {}
impl Unsigned for u64 {}

/// Integer is a wrapper trait for generic integer types used
/// by statrs
pub trait Integer
    : num::Integer + rand::distributions::range::SampleRange + NumBase {
}

impl Integer for u8 {}
impl Integer for u16 {}
impl Integer for u32 {}
impl Integer for u64 {}
impl Integer for i8 {}
impl Integer for i16 {}
impl Integer for i32 {}
impl Integer for i64 {}

/// Base trait for numeric types
pub trait NumBase
    : num::NumCast + num::ToPrimitive + rand::Rand + Copy + std::fmt::Display + std::fmt::Debug
    {
}

impl NumBase for u8 {}
impl NumBase for u16 {}
impl NumBase for u32 {}
impl NumBase for u64 {}
impl NumBase for i8 {}
impl NumBase for i16 {}
impl NumBase for i32 {}
impl NumBase for i64 {}
impl NumBase for f32 {}
impl NumBase for f64 {}
