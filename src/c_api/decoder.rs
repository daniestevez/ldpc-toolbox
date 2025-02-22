use super::{c_to_string, size_t_to_usize};
use crate::{
    cli::ber::parse_puncturing_pattern,
    decoder::{
        LdpcDecoder,
        factory::{DecoderFactory, DecoderImplementation},
    },
    simulation::puncturing::Puncturer,
    sparse::SparseMatrix,
};
use libc::size_t;
use std::{
    convert::TryFrom,
    error::Error,
    ffi::{c_char, c_void},
};

#[derive(Debug)]
struct Decoder {
    decoder: Box<dyn LdpcDecoder>,
    puncturer: Option<Puncturer>,
}

impl Decoder {
    fn new(alist: &str, implementation: &str, puncturing: &str) -> Result<Decoder, Box<dyn Error>> {
        let h = SparseMatrix::from_alist(alist)?;
        let implementation: DecoderImplementation = implementation.parse()?;
        let puncturing_pattern = if !puncturing.is_empty() {
            Some(parse_puncturing_pattern(puncturing)?)
        } else {
            None
        };
        let puncturer = puncturing_pattern.map(|v| Puncturer::new(&v));
        let decoder = implementation.build_decoder(h);
        Ok(Decoder { decoder, puncturer })
    }

    fn from_alist_file(
        alist_file: &str,
        implementation: &str,
        puncturing: &str,
    ) -> Result<Decoder, Box<dyn Error>> {
        Decoder::new(
            &std::fs::read_to_string(alist_file)?,
            implementation,
            puncturing,
        )
    }

    fn decode_f64(&mut self, output: &mut [u8], llrs: &[f64], max_iterations: u32) -> i32 {
        let depunctured = self.puncturer.as_ref().map(|p| p.depuncture(llrs).unwrap());
        let llrs = if let Some(d) = &depunctured { d } else { llrs };
        let res = self
            .decoder
            .decode(llrs, usize::try_from(max_iterations).unwrap());
        let success = res.is_ok();
        let decoded = match res {
            Ok(o) => o,
            Err(o) => o,
        };
        output.copy_from_slice(&decoded.codeword[..output.len()]);
        if success {
            i32::try_from(decoded.iterations).unwrap()
        } else {
            -1
        }
    }

    fn decode_f32(&mut self, output: &mut [u8], llrs: &[f32], max_iterations: u32) -> i32 {
        let llrs_f64 = llrs.iter().copied().map(f64::from).collect::<Vec<f64>>();
        self.decode_f64(output, &llrs_f64, max_iterations)
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn ldpc_toolbox_decoder_ctor(
    alist_file_path: *const c_char,
    implementation: *const c_char,
    puncturing: *const c_char,
) -> *mut c_void {
    let alist_file_path = unsafe { c_to_string(alist_file_path) };
    let implementation = unsafe { c_to_string(implementation) };
    let puncturing = unsafe { c_to_string(puncturing) };
    Decoder::from_alist_file(&alist_file_path, &implementation, &puncturing)
        .map_or(std::ptr::null_mut(), |decoder| {
            Box::into_raw(Box::new(decoder)) as *mut c_void
        })
}

#[unsafe(no_mangle)]
unsafe extern "C" fn ldpc_toolbox_decoder_ctor_alist_string(
    alist: *const c_char,
    implementation: *const c_char,
    puncturing: *const c_char,
) -> *mut c_void {
    let alist = unsafe { c_to_string(alist) };
    let implementation = unsafe { c_to_string(implementation) };
    let puncturing = unsafe { c_to_string(puncturing) };
    Decoder::new(&alist, &implementation, &puncturing).map_or(std::ptr::null_mut(), |decoder| {
        Box::into_raw(Box::new(decoder)) as *mut c_void
    })
}

#[unsafe(no_mangle)]
unsafe extern "C" fn ldpc_toolbox_decoder_dtor(decoder: *mut c_void) {
    drop(unsafe { Box::from_raw(decoder as *mut Decoder) });
}

#[unsafe(no_mangle)]
unsafe extern "C" fn ldpc_toolbox_decoder_decode_f64(
    decoder: *mut c_void,
    output: *mut u8,
    output_len: size_t,
    llrs: *const f64,
    llrs_len: size_t,
    max_iterations: u32,
) -> i32 {
    let output = unsafe { std::slice::from_raw_parts_mut(output, size_t_to_usize(output_len)) };
    let llrs = unsafe { std::slice::from_raw_parts(llrs, size_t_to_usize(llrs_len)) };
    let decoder = unsafe { &mut *(decoder as *mut Decoder) };
    decoder.decode_f64(output, llrs, max_iterations)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn ldpc_toolbox_decoder_decode_f32(
    decoder: *mut c_void,
    output: *mut u8,
    output_len: size_t,
    llrs: *const f32,
    llrs_len: size_t,
    max_iterations: u32,
) -> i32 {
    let output = unsafe { std::slice::from_raw_parts_mut(output, size_t_to_usize(output_len)) };
    let llrs = unsafe { std::slice::from_raw_parts(llrs, size_t_to_usize(llrs_len)) };
    let decoder = unsafe { &mut *(decoder as *mut Decoder) };
    decoder.decode_f32(output, llrs, max_iterations)
}
