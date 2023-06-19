use super::{c_to_string, size_t_to_usize};
use crate::{
    cli::ber::parse_puncturing_pattern, encoder::Encoder as LdpcEncoder, gf2::GF2,
    simulation::puncturing::Puncturer, sparse::SparseMatrix,
};
use libc::size_t;
use ndarray::Array1;
use num_traits::{One, Zero};
use std::{
    error::Error,
    ffi::{c_char, c_void},
};

#[derive(Debug)]
struct Encoder {
    encoder: LdpcEncoder,
    puncturer: Option<Puncturer>,
}

impl Encoder {
    fn new(alist: &str, puncturing: &str) -> Result<Encoder, Box<dyn Error>> {
        let h = SparseMatrix::from_alist(&std::fs::read_to_string(alist)?)?;
        let puncturing_pattern = if !puncturing.is_empty() {
            Some(parse_puncturing_pattern(puncturing)?)
        } else {
            None
        };
        let puncturer = puncturing_pattern.map(|v| Puncturer::new(&v));
        let encoder = LdpcEncoder::from_h(&h)?;
        Ok(Encoder { encoder, puncturer })
    }

    fn encode(&self, output: &mut [u8], input: &[u8]) {
        let encoded = self
            .encoder
            .encode(&Array1::from_iter(input.iter().map(|&b| {
                if b == 1 {
                    GF2::one()
                } else {
                    GF2::zero()
                }
            })));
        let encoded = if let Some(p) = &self.puncturer {
            p.puncture(&encoded).unwrap()
        } else {
            encoded
        };
        assert_eq!(output.len(), encoded.len());
        for (y, x) in output.iter_mut().zip(encoded.iter()) {
            *y = if x.is_one() { 1 } else { 0 };
        }
    }
}

#[no_mangle]
unsafe extern "C" fn ldpc_toolbox_encoder_ctor(
    alist: *const c_char,
    puncturing: *const c_char,
) -> *mut c_void {
    let alist = c_to_string(alist);
    let puncturing = c_to_string(puncturing);
    if let Ok(encoder) = Encoder::new(&alist, &puncturing) {
        Box::into_raw(Box::new(encoder)) as *mut c_void
    } else {
        std::ptr::null_mut()
    }
}

#[no_mangle]
unsafe extern "C" fn ldpc_toolbox_encoder_dtor(encoder: *mut c_void) {
    drop(Box::from_raw(encoder as *mut Encoder));
}

#[no_mangle]
unsafe extern "C" fn ldpc_toolbox_encoder_encode(
    encoder: *mut c_void,
    output: *mut u8,
    output_len: size_t,
    input: *const u8,
    input_len: size_t,
) {
    let output = std::slice::from_raw_parts_mut(output, size_t_to_usize(output_len));
    let input = std::slice::from_raw_parts(input, size_t_to_usize(input_len));
    let encoder = &mut *(encoder as *mut Encoder);
    encoder.encode(output, input);
}
