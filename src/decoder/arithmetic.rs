//! LDPC decoder arithmetic.
//!
//! This module contains the trait [`DecoderArithmetic`], which defines generic
//! arithmetic rules used by a belief propagation LDPC decoder, and implementors
//! of that trait. The LDCP decoder [`Decoder`](super::Decoder) is generic over
//! the `DecoderArithmetic` trait, so it can be used to obtain monomorphized
//! implementations for different arithemtic rules.
//!
//! # References
//!
//! Most of the arithmetic rules implemented here are taken from:
//!
//! [1] Jon Hamkins, [Performance of Low-Density Parity-Check Coded Modulation](https://ipnpr.jpl.nasa.gov/progress_report/42-184/184D.pdf),
//! IPN Progress Report 42-184, February 15, 2011.
//!
//! Another good resource is this book:
//!
//! [2] Sarah J. Johnson, Iterative Error Correction: Turbo, Low-Density
//! Parity-Check and Repeat-Accumulate Codes. Cambridge University Press. June
//! 2012.
//!
//! Other references:
//!
//! [3] C. Jones, et al. “Approximate-MIN* Constraint Node Updating for LDPC
//! Code Decoding.” In Proceedings of MILCOM 2003 (Boston, Massachusetts),
//! 1-157-1-162. Piscataway, NJ: IEEE, October 2003.
//!

use super::{Message, SentMessage};
use std::convert::identity;

/// LDPC decoder arithmetic.
///
/// This trait models generic arithmetic rules for a belief propagation LDPC
/// decoder. The trait defines the data types to use for LLRs and messages, and
/// how to compute the check node and variable node messages.
///
/// The LDPC decoder [`Decoder`](super::Decoder) is generic over objects
/// implementing this trait.
///
/// The methods in this trait depend on `&self` or `&mut self` so that the
/// decoder arithmetic object can have an internal state implement lookup
/// tables, caching, etc.
pub trait DecoderArithmetic: std::fmt::Debug + Send {
    /// LLR.
    ///
    /// Defines the type used to represent LLRs.
    type Llr: std::fmt::Debug + Copy + Default + Send;
    /// Check node message.
    ///
    /// Defines the type used to represent check node messages.
    type CheckMessage: std::fmt::Debug + Copy + Default + Send;
    /// Variable node message.
    ///
    /// Defines the type used to represent variable node messages.
    type VarMessage: std::fmt::Debug + Copy + Default + Send;

    /// Quantization function for input LLRs.
    ///
    /// Defines how the channel LLRs (the input to the decoder) are quantized
    /// and represented internally as a [`Self::Llr`].
    fn input_llr_quantize(&self, llr: f64) -> Self::Llr;

    /// Hard decision on LLRs.
    ///
    /// Returns the hard decision bit corresponding to an LLR.
    fn llr_hard_decision(&self, llr: Self::Llr) -> bool;

    /// Transform LLR to variable message.
    ///
    /// Defines how to transform an LLR into a variable message. This is used in
    /// the first iteration of the belief propagation algorithm, where the
    /// variable messages are simply the channel LLRs.
    fn llr_to_var_message(&self, llr: Self::Llr) -> Self::VarMessage;

    /// Send check messages from a check node.
    ///
    /// This function is called with the list of variable messages arriving to a
    /// check node, and closure that must be called to send each check message
    /// outgoing from that check node.
    ///
    /// This function should compute the values of the check node messages and
    /// call the `send` closure for each of the variable nodes connected to the
    /// check node being processed.
    fn send_check_messages<F>(&mut self, var_messages: &[Message<Self::VarMessage>], send: F)
    where
        F: FnMut(SentMessage<Self::CheckMessage>);

    /// Send variable messages from a variable node.
    ///
    /// This function is called with the channel LLR corresponding to a variable
    /// node, a list of check messages arriving to that variable node, and
    /// closure that must be called to send each variable message outgoing from
    /// that variable node.
    ///
    /// This function should compute the values of the variable node messages and
    /// call the `send` closure for each of the check nodes connected to the
    /// variable node being processed.
    ///
    /// Additionally, the function returns the new LLR for this variable node.
    fn send_var_messages<F>(
        &mut self,
        input_llr: Self::Llr,
        check_messages: &[Message<Self::CheckMessage>],
        send: F,
    ) -> Self::Llr
    where
        F: FnMut(SentMessage<Self::VarMessage>);
}

// The usual variable message update rule, without any clipping.
fn send_var_messages_no_clip<T, F>(input_llr: T, check_messages: &[Message<T>], mut send: F) -> T
where
    T: std::iter::Sum + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + Copy,
    F: FnMut(SentMessage<T>),
{
    // Compute new LLR
    let llr: T = input_llr + check_messages.iter().map(|m| m.value).sum::<T>();
    // Exclude the contribution of each check node to generate message for
    // that check node
    for msg in check_messages.iter() {
        send(SentMessage {
            dest: msg.source,
            value: llr - msg.value,
        });
    }
    llr
}

macro_rules! impl_phif {
    ($ty:ident, $f:ty, $min_x:expr) => {
        /// LDPC decoder arithmetic with `$f` and `phi(x)` involution.
        ///
        /// This is a [`DecoderArithmetic`] that uses `$f` to represent the LLRs and
        /// messages and computes the check node messages using the involution `phi(x) =
        /// -log(tanh(x/2))`.
        ///
        /// See (2.33) in page 68 in [2].
        #[derive(Debug, Clone, Default)]
        pub struct $ty {
            phis: Vec<$f>,
        }

        impl $ty {
            /// Creates a new [`$ty`] decoder arithmetic object.
            pub fn new() -> $ty {
                <$ty>::default()
            }
        }

        impl $ty {
            fn phi(x: $f) -> $f {
                // Ensure that x is not zero. Otherwise the output will be +inf, which gives
                // problems when computing (+inf) - (+inf).
                let x = x.max($min_x);
                -((0.5 * x).tanh().ln())
            }
        }

        impl DecoderArithmetic for $ty {
            type Llr = $f;
            type CheckMessage = $f;
            type VarMessage = $f;

            fn input_llr_quantize(&self, llr: f64) -> $f {
                llr as $f
            }

            fn llr_hard_decision(&self, llr: $f) -> bool {
                llr <= 0.0
            }

            fn llr_to_var_message(&self, llr: $f) -> $f {
                llr
            }

            fn send_check_messages<F>(&mut self, var_messages: &[Message<$f>], mut send: F)
            where
                F: FnMut(SentMessage<$f>),
            {
                // Compute combination of all variable messages
                let mut sign: u32 = 0;
                let mut sum = 0.0;
                if self.phis.len() < var_messages.len() {
                    self.phis.resize(var_messages.len(), 0.0);
                }
                for (msg, phi) in var_messages.iter().zip(self.phis.iter_mut()) {
                    let x = msg.value;
                    let phi_x = Self::phi(x.abs());
                    *phi = phi_x;
                    sum += phi_x;
                    if x < 0.0 {
                        sign ^= 1;
                    }
                }

                // Exclude the contribution of each variable to generate message for
                // that variable
                for (msg, phi) in var_messages.iter().zip(self.phis.iter()) {
                    let x = msg.value;
                    let y = Self::phi(sum - phi);
                    let s = if x < 0.0 { sign ^ 1 } else { sign };
                    let val = if s == 0 { y } else { -y };
                    send(SentMessage {
                        dest: msg.source,
                        value: val,
                    });
                }
            }

            fn send_var_messages<F>(
                &mut self,
                input_llr: $f,
                check_messages: &[Message<$f>],
                send: F,
            ) -> $f
            where
                F: FnMut(SentMessage<$f>),
            {
                send_var_messages_no_clip(input_llr, check_messages, send)
            }
        }
    };
}

impl_phif!(Phif64, f64, 1e-30);
impl_phif!(Phif32, f32, 1e-30);

macro_rules! impl_tanhf {
    ($ty:ident, $f:ty, $tanh_clamp:expr) => {
        /// LDPC decoder arithmetic with `$f` and `2 * atanh(\Prod tanh(x/2)` rule.
        ///
        /// This is a [`DecoderArithmetic`] that uses `$f` to represent the LLRs
        /// and messages and computes the check node messages using the usual
        /// tanh product rule.
        ///
        /// See (33) in [1].
        #[derive(Debug, Clone, Default)]
        pub struct $ty {
            tanhs: Vec<$f>,
        }

        impl $ty {
            /// Creates a new [`$ty`] decoder arithmetic object.
            pub fn new() -> $ty {
                <$ty>::default()
            }
        }

        impl DecoderArithmetic for $ty {
            type Llr = $f;
            type CheckMessage = $f;
            type VarMessage = $f;

            fn input_llr_quantize(&self, llr: f64) -> $f {
                llr as $f
            }

            fn llr_hard_decision(&self, llr: $f) -> bool {
                llr <= 0.0
            }

            fn llr_to_var_message(&self, llr: $f) -> $f {
                llr
            }

            fn send_check_messages<F>(&mut self, var_messages: &[Message<$f>], mut send: F)
            where
                F: FnMut(SentMessage<$f>),
            {
                // Compute tanh's of all variable messages
                if self.tanhs.len() < var_messages.len() {
                    self.tanhs.resize(var_messages.len(), 0.0);
                }
                for (msg, tanh) in var_messages.iter().zip(self.tanhs.iter_mut()) {
                    let x = msg.value;
                    let t = (0.5 * x).clamp(-$tanh_clamp, $tanh_clamp).tanh();
                    *tanh = t;
                }

                for exclude_msg in var_messages.iter() {
                    // product of all the tanh's except that of exclude_msg
                    let product = var_messages
                        .iter()
                        .zip(self.tanhs.iter())
                        .filter_map(|(msg, tanh)| {
                            if msg.source != exclude_msg.source {
                                Some(tanh)
                            } else {
                                None
                            }
                        })
                        .product::<$f>();
                    send(SentMessage {
                        dest: exclude_msg.source,
                        value: 2.0 * product.atanh(),
                    })
                }
            }

            fn send_var_messages<F>(
                &mut self,
                input_llr: $f,
                check_messages: &[Message<$f>],
                send: F,
            ) -> $f
            where
                F: FnMut(SentMessage<$f>),
            {
                send_var_messages_no_clip(input_llr, check_messages, send)
            }
        }
    };
}

// For f64, tanh(19) already gives 1.0 (and we want to avoid computing
// atanh(1.0) = inf).
impl_tanhf!(Tanhf64, f64, 18.0);
// For f32, tanh(10) already gives 1.0.
impl_tanhf!(Tanhf32, f32, 9.0);

macro_rules! impl_minstarapproxf {
    ($ty:ident, $f:ty) => {
        /// LDPC decoder arithmetic with `$f` and the following approximation to
        /// the min* function:
        ///
        /// min*(x,y) approx = sign(xy) * [min(|x|,|y|) - log(1 + exp(-||x|-|y||))].
        ///
        /// This is a [`DecoderArithmetic`] that uses `$f` to represent the LLRs
        /// and messages and computes the check node messages using an approximation
        /// to the min* rule.
        ///
        /// See (35) in [1].
        #[derive(Debug, Clone, Default)]
        pub struct $ty {}

        impl $ty {
            /// Creates a new [`$ty`] decoder arithmetic object.
            pub fn new() -> $ty {
                <$ty>::default()
            }
        }

        impl DecoderArithmetic for $ty {
            type Llr = $f;
            type CheckMessage = $f;
            type VarMessage = $f;

            fn input_llr_quantize(&self, llr: f64) -> $f {
                llr as $f
            }

            fn llr_hard_decision(&self, llr: $f) -> bool {
                llr <= 0.0
            }

            fn llr_to_var_message(&self, llr: $f) -> $f {
                llr
            }

            fn send_check_messages<F>(&mut self, var_messages: &[Message<$f>], mut send: F)
            where
                F: FnMut(SentMessage<$f>),
            {
                for exclude_msg in var_messages.iter() {
                    let mut sign: u32 = 0;
                    let mut minstar = None;
                    for msg in var_messages
                        .iter()
                        .filter(|msg| msg.source != exclude_msg.source)
                    {
                        let x = msg.value;
                        if x < 0.0 {
                            sign ^= 1;
                        }
                        let x = x.abs();
                        minstar = Some(match minstar {
                            None => x,
                            // We clamp the output to 0 from below because we
                            // are doing min* of positive numbers, but since
                            // we've thrown away a positive term in the
                            // approximation to min*, the approximation could
                            // come out negative.
                            Some(y) => (x.min(y) - (-(x - y).abs()).exp().ln_1p()).max(0.0),
                        });
                    }
                    let minstar =
                        minstar.expect("only one variable message connected to check node");
                    let minstar = if sign == 0 { minstar } else { -minstar };
                    send(SentMessage {
                        dest: exclude_msg.source,
                        value: minstar,
                    })
                }
            }

            fn send_var_messages<F>(
                &mut self,
                input_llr: $f,
                check_messages: &[Message<$f>],
                send: F,
            ) -> $f
            where
                F: FnMut(SentMessage<$f>),
            {
                send_var_messages_no_clip(input_llr, check_messages, send)
            }
        }
    };
}

impl_minstarapproxf!(Minstarapproxf64, f64);
impl_minstarapproxf!(Minstarapproxf32, f32);

macro_rules! impl_8bitquant {
    ($ty:ident) => {
        impl $ty {
            const QUANTIZER_C: f64 = 8.0;

            /// Creates a new [`$ty`] decoder arithmetic object.
            pub fn new() -> $ty {
                let table = (0..=127)
                    .map_while(|t| {
                        let x = (Self::QUANTIZER_C
                            * (-(t as f64 / Self::QUANTIZER_C)).exp().ln_1p())
                        .round() as i8;
                        if x > 0 {
                            Some(x)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                $ty { table }
            }

            fn lookup(&self, x: i8) -> i8 {
                assert!(x >= 0);
                self.table.get(x as usize).copied().unwrap_or(0)
            }

            fn clip(x: i16) -> i8 {
                if x >= 127 {
                    127
                } else if x <= -127 {
                    -127
                } else {
                    x as i8
                }
            }
        }
    };
}

macro_rules! impl_send_var_messages_i8 {
    ($degree_one_clip:expr, $jones_clip:expr) => {
        fn send_var_messages<F>(
            &mut self,
            input_llr: i8,
            check_messages: &[Message<i8>],
            mut send: F,
        ) -> i8
        where
            F: FnMut(SentMessage<i8>),
        {
            let degree_one = check_messages.len() == 1;
            // Compute new LLR. We use an i16 to avoid overflows.
            let llr = i16::from($degree_one_clip(input_llr, degree_one))
                + check_messages
                    .iter()
                    .map(|m| i16::from(m.value))
                    .sum::<i16>();
            // Optional Jones clipping
            let llr = $jones_clip(llr);
            // Exclude the contribution of each check node to generate message for
            // that check node
            for msg in check_messages.iter() {
                send(SentMessage {
                    dest: msg.source,
                    value: Self::clip(llr - i16::from(msg.value)),
                });
            }
            Self::clip(llr)
        }
    };
}

macro_rules! impl_minstarapproxi8 {
    ($ty:ident, $jones_clip:expr, $check_hardlimit:expr, $degree_one_clip:expr) => {
        /// LDPC decoder arithmetic with 8-bit quantization and an approximation to the
        /// min* function.
        ///
        /// This is a [`DecoderArithmetic`] that uses `i8` to represent the LLRs
        /// and messages and computes the check node messages using an approximation
        /// to the min* rule.
        ///
        /// See (36) in [1].
        #[derive(Debug, Clone)]
        pub struct $ty {
            table: Box<[i8]>,
        }

        impl_8bitquant!($ty);

        impl Default for $ty {
            fn default() -> $ty {
                <$ty>::new()
            }
        }

        impl DecoderArithmetic for $ty {
            type Llr = i8;
            type CheckMessage = i8;
            type VarMessage = i8;

            fn input_llr_quantize(&self, llr: f64) -> i8 {
                let x = Self::QUANTIZER_C * llr;
                if x >= 127.0 {
                    127
                } else if x <= -127.0 {
                    -127
                } else {
                    x.round() as i8
                }
            }

            fn llr_hard_decision(&self, llr: i8) -> bool {
                llr <= 0
            }

            fn llr_to_var_message(&self, llr: i8) -> i8 {
                llr
            }

            fn send_check_messages<F>(&mut self, var_messages: &[Message<i8>], mut send: F)
            where
                F: FnMut(SentMessage<i8>),
            {
                for exclude_msg in var_messages.iter() {
                    let mut sign: u32 = 0;
                    let mut minstar = None;
                    for msg in var_messages
                        .iter()
                        .filter(|msg| msg.source != exclude_msg.source)
                    {
                        let x = msg.value;
                        if x < 0 {
                            sign ^= 1;
                        }
                        let x = x.abs();
                        minstar = Some(match minstar {
                            None => x,
                            // We clamp the output to 0 from below because we
                            // are doing min* of positive numbers, but since
                            // we've thrown away a positive term in the
                            // approximation to min*, the approximation could
                            // come out negative.
                            Some(y) => (x.min(y) - self.lookup((x - y).abs())).max(0),
                        });
                    }
                    let minstar =
                        minstar.expect("only one variable message connected to check node");
                    let minstar = if sign == 0 { minstar } else { -minstar };
                    // Optional partial hard-limiting
                    let minstar = $check_hardlimit(minstar);
                    send(SentMessage {
                        dest: exclude_msg.source,
                        value: minstar,
                    })
                }
            }

            impl_send_var_messages_i8!($degree_one_clip, $jones_clip);
        }
    };
}

macro_rules! jones_clip {
    () => {
        |x| i16::from(Self::clip(x))
    };
}

macro_rules! partial_hard_limit {
    () => {
        |x| {
            if x <= -100 {
                -127
            } else if x >= 100 {
                127
            } else {
                x
            }
        }
    };
}

macro_rules! degree_one_clipping {
    () => {
        |x, degree_one| {
            if degree_one {
                if x <= -116 {
                    -116
                } else if x >= 116 {
                    116
                } else {
                    x
                }
            } else {
                x
            }
        }
    };
}

macro_rules! degree_one_no_clipping {
    () => {
        |x, _| x
    };
}

impl_minstarapproxi8!(
    Minstarapproxi8,
    identity,
    identity,
    degree_one_no_clipping!()
);
impl_minstarapproxi8!(
    Minstarapproxi8Jones,
    jones_clip!(),
    identity,
    degree_one_no_clipping!()
);
impl_minstarapproxi8!(
    Minstarapproxi8PartialHardLimit,
    identity,
    partial_hard_limit!(),
    degree_one_no_clipping!()
);
impl_minstarapproxi8!(
    Minstarapproxi8JonesPartialHardLimit,
    jones_clip!(),
    partial_hard_limit!(),
    degree_one_no_clipping!()
);
impl_minstarapproxi8!(
    Minstarapproxi8Deg1Clip,
    identity,
    identity,
    degree_one_clipping!()
);
impl_minstarapproxi8!(
    Minstarapproxi8JonesDeg1Clip,
    jones_clip!(),
    identity,
    degree_one_clipping!()
);
impl_minstarapproxi8!(
    Minstarapproxi8PartialHardLimitDeg1Clip,
    identity,
    partial_hard_limit!(),
    degree_one_clipping!()
);
impl_minstarapproxi8!(
    Minstarapproxi8JonesPartialHardLimitDeg1Clip,
    jones_clip!(),
    partial_hard_limit!(),
    degree_one_clipping!()
);

macro_rules! impl_aminstarf {
    ($ty:ident, $f:ty) => {
        /// LDPC decoder arithmetic with `$f` and the A-Min*-BP described in [3].
        ///
        /// This is a [`DecoderArithmetic`] that uses `$f` to represent the LLRs
        /// and messages and computes the check node messages using an approximation
        /// to the min* rule.
        #[derive(Debug, Clone, Default)]
        pub struct $ty {}

        impl $ty {
            /// Creates a new [`$ty`] decoder arithmetic object.
            pub fn new() -> $ty {
                <$ty>::default()
            }
        }

        impl DecoderArithmetic for $ty {
            type Llr = $f;
            type CheckMessage = $f;
            type VarMessage = $f;

            fn input_llr_quantize(&self, llr: f64) -> $f {
                llr as $f
            }

            fn llr_hard_decision(&self, llr: $f) -> bool {
                llr <= 0.0
            }

            fn llr_to_var_message(&self, llr: $f) -> $f {
                llr
            }

            fn send_check_messages<F>(&mut self, var_messages: &[Message<$f>], mut send: F)
            where
                F: FnMut(SentMessage<$f>),
            {
                let (argmin, msgmin) = var_messages
                    .iter()
                    .enumerate()
                    .min_by(|(_, msg1), (_, msg2)| {
                        msg1.value.abs().partial_cmp(&msg2.value.abs()).unwrap()
                    })
                    .expect("var_messages is empty");
                let mut sign: u32 = 0;
                let mut delta = None;
                for (j, msg) in var_messages.iter().enumerate() {
                    let x = msg.value;
                    if x < 0.0 {
                        sign ^= 1;
                    }
                    if j != argmin {
                        let x = x.abs();
                        delta = Some(match delta {
                            None => x,
                            Some(y) => {
                                (x.min(y) - (-(x - y).abs()).exp().ln_1p()
                                    + (-(x + y)).exp().ln_1p())
                            }
                        });
                    }
                }
                let delta = delta.expect("var_messages_empty");

                send(SentMessage {
                    dest: msgmin.source,
                    value: if (sign != 0) ^ (msgmin.value < 0.0) {
                        -delta
                    } else {
                        delta
                    },
                });

                let vmin = msgmin.value.abs();
                let delta = delta.min(vmin) - (-(delta - vmin).abs()).exp().ln_1p()
                    + (-(delta + vmin)).exp().ln_1p();
                for msg in var_messages.iter().enumerate().filter_map(|(j, msg)| {
                    if j != argmin {
                        Some(msg)
                    } else {
                        None
                    }
                }) {
                    send(SentMessage {
                        dest: msg.source,
                        value: if (sign != 0) ^ (msg.value < 0.0) {
                            -delta
                        } else {
                            delta
                        },
                    });
                }
            }

            fn send_var_messages<F>(
                &mut self,
                input_llr: $f,
                check_messages: &[Message<$f>],
                send: F,
            ) -> $f
            where
                F: FnMut(SentMessage<$f>),
            {
                send_var_messages_no_clip(input_llr, check_messages, send)
            }
        }
    };
}

impl_aminstarf!(Aminstarf64, f64);
impl_aminstarf!(Aminstarf32, f32);

macro_rules! impl_aminstari8 {
    ($ty:ident, $jones_clip:expr, $check_hardlimit:expr, $degree_one_clip:expr) => {
        /// LDPC decoder arithmetic with 8-bit quantization and the A-Min*-BP
        /// described in [3].
        ///
        /// This is a [`DecoderArithmetic`] that uses `i8` to represent the LLRs
        /// and messages and computes the check node messages using an approximation
        /// to the min* rule.
        #[derive(Debug, Clone, Default)]
        pub struct $ty {
            table: Box<[i8]>,
        }

        impl_8bitquant!($ty);

        impl DecoderArithmetic for $ty {
            type Llr = i8;
            type CheckMessage = i8;
            type VarMessage = i8;

            fn input_llr_quantize(&self, llr: f64) -> i8 {
                let x = Self::QUANTIZER_C * llr;
                if x >= 127.0 {
                    127
                } else if x <= -127.0 {
                    -127
                } else {
                    x.round() as i8
                }
            }

            fn llr_hard_decision(&self, llr: i8) -> bool {
                llr <= 0
            }

            fn llr_to_var_message(&self, llr: i8) -> i8 {
                llr
            }

            fn send_check_messages<F>(&mut self, var_messages: &[Message<i8>], mut send: F)
            where
                F: FnMut(SentMessage<i8>),
            {
                let (argmin, msgmin) = var_messages
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, msg)| msg.value.abs())
                    .expect("var_messages is empty");
                let mut sign: u32 = 0;
                let mut delta = None;
                for (j, msg) in var_messages.iter().enumerate() {
                    let x = msg.value;
                    if x < 0 {
                        sign ^= 1;
                    }
                    if j != argmin {
                        let x = x.abs();
                        delta = Some(match delta {
                            None => x,
                            // We clamp the output to 0 from below because we
                            // are doing min* of positive numbers, but since
                            // we've thrown away a positive term in the
                            // approximation to min*, the approximation could
                            // come out negative.
                            Some(y) => (x.min(y) - self.lookup((x - y).abs())
                                + self.lookup(x.saturating_add(y)))
                            .max(0),
                        });
                    }
                }
                let delta = delta.expect("var_messages_empty");
                let delta_hl = $check_hardlimit(delta);

                send(SentMessage {
                    dest: msgmin.source,
                    value: if (sign != 0) ^ (msgmin.value < 0) {
                        -delta_hl
                    } else {
                        delta_hl
                    },
                });

                let vmin = msgmin.value.abs();
                let delta = (delta.min(vmin) - self.lookup((delta - vmin).abs())
                    + self.lookup(delta.saturating_add(vmin)))
                .max(0);
                let delta_hl = $check_hardlimit(delta);
                for msg in var_messages.iter().enumerate().filter_map(|(j, msg)| {
                    if j != argmin {
                        Some(msg)
                    } else {
                        None
                    }
                }) {
                    send(SentMessage {
                        dest: msg.source,
                        value: if (sign != 0) ^ (msg.value < 0) {
                            -delta_hl
                        } else {
                            delta_hl
                        },
                    });
                }
            }

            impl_send_var_messages_i8!($degree_one_clip, $jones_clip);
        }
    };
}

impl_aminstari8!(Aminstari8, identity, identity, degree_one_no_clipping!());
impl_aminstari8!(
    Aminstari8Jones,
    jones_clip!(),
    identity,
    degree_one_no_clipping!()
);
impl_aminstari8!(
    Aminstari8PartialHardLimit,
    identity,
    partial_hard_limit!(),
    degree_one_no_clipping!()
);
impl_aminstari8!(
    Aminstari8JonesPartialHardLimit,
    jones_clip!(),
    partial_hard_limit!(),
    degree_one_no_clipping!()
);
impl_aminstari8!(
    Aminstari8Deg1Clip,
    identity,
    identity,
    degree_one_clipping!()
);
impl_aminstari8!(
    Aminstari8JonesDeg1Clip,
    jones_clip!(),
    identity,
    degree_one_clipping!()
);
impl_aminstari8!(
    Aminstari8PartialHardLimitDeg1Clip,
    identity,
    partial_hard_limit!(),
    degree_one_clipping!()
);
impl_aminstari8!(
    Aminstari8JonesPartialHardLimitDeg1Clip,
    jones_clip!(),
    partial_hard_limit!(),
    degree_one_clipping!()
);
