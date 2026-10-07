//! Typed fragment expressions, CPU evaluation, and cached WGSL compilation.
//!
//! [`Paint`] composes expressions into one fragment program. Plain values
//! become uniforms; changing them reuses the program. Fill an element with
//! one through [`Fill::program`](crate::Fill::program): renderers link its
//! program into their standard shaders, where it is evaluated in the
//! element's own space, inside its clips and corners.

mod builtins;
mod compile;
mod expr;
mod library;
mod paint;
mod value;

pub(crate) use compile::MAX_PROGRAM_ID;
pub use compile::{CompiledPaint, Program, ShaderError, prelude_source};
pub use expr::{
    Bool, Expr, Scalar, Vec2, Vec3, Vec4, constant, iterate, iterate_until, vec2, vec3, vec4,
};
pub use library::{noise, prelude, shape};
pub use paint::{Paint, Pixel, color, paint, rgba};
pub use prelude::Fragment;
pub use value::{Arith, CpuValue, Float, Operand, Value, Vector, Widen};
pub use wgsl_rs::std::{Vec2f, Vec3f, Vec4f, vec2f, vec3f, vec4f};

#[cfg(test)]
mod tests;
