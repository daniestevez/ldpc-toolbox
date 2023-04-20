//! LDPC decoder arithmetic.
//!
//! This module contains the trait [`DecoderArithmetic`], which defines generic
//! arithmetic rules used by a belief propagation LDPC decoder, and implementors
//! of that trait. The LDCP decoder [`Decoder`](super::Decoder) is generic over
//! the `DecoderArithmetic` trait, so it can be used to obtain monomorphized
//! implementations for different arithemtic rules.

use super::{Message, SentMessage};

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
    fn input_llr_quantize(&self, llr: f32) -> Self::Llr;

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

            fn input_llr_quantize(&self, llr: f32) -> $f {
                <$f>::from(llr)
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

            fn input_llr_quantize(&self, llr: f32) -> $f {
                <$f>::from(llr)
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
