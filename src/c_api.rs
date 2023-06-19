use libc::size_t;
use std::{
    convert::TryFrom,
    ffi::{c_char, CStr},
};

mod decoder;
mod encoder;

unsafe fn c_to_string(s: *const c_char) -> String {
    String::from_utf8_lossy(CStr::from_ptr(s).to_bytes()).to_string()
}

#[allow(clippy::useless_conversion)]
fn size_t_to_usize(n: size_t) -> usize {
    usize::try_from(n).unwrap()
}
