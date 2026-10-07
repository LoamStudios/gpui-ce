//! Shared typed rendering and shader contracts.

pub mod artifacts;
pub mod blur;
pub mod group;
#[cfg(any(windows, test))]
pub mod hlsl;
mod instances;
pub mod link;
pub mod linked;
pub mod msl;
pub mod path_types;
pub mod shaders;

pub use instances::InstanceRange;
