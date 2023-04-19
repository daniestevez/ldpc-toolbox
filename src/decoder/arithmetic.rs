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

/// LDPC decoder arithmetic with `f64` and `phi(x)` involution.
///
/// This is a [`DecoderArithmetic`] that uses `f64` to represent the LLRs and
/// messages and computes the check node messages using the involution `phi(x) =
/// -log(tanh(x/2))`.
#[derive(Debug, Clone, Default)]
pub struct Phif64 {
    phis: Vec<f64>,
}

impl Phif64 {
    /// Creates a new [`Phif64`] decoder arithmetic object.
    pub fn new() -> Phif64 {
        Phif64::default()
    }
}

impl Phif64 {
    fn phi(x: f64) -> f64 {
        // Ensure that x is not zero. Otherwise the output will be +inf, which gives
        // problems when computing (+inf) - (+inf).
        let x = x.max(1e-30);
        -((0.5 * x).tanh().ln())
    }
}

impl DecoderArithmetic for Phif64 {
    type Llr = f64;
    type CheckMessage = f64;
    type VarMessage = f64;

    fn input_llr_quantize(&self, llr: f32) -> f64 {
        f64::from(llr)
    }

    fn llr_hard_decision(&self, llr: f64) -> bool {
        llr <= 0.0
    }

    fn llr_to_var_message(&self, llr: f64) -> f64 {
        llr
    }

    fn send_check_messages<F>(&mut self, var_messages: &[Message<f64>], mut send: F)
    where
        F: FnMut(SentMessage<f64>),
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
        input_llr: f64,
        check_messages: &[Message<f64>],
        mut send: F,
    ) -> f64
    where
        F: FnMut(SentMessage<f64>),
    {
        // Compute new LLR
        let llr = input_llr + check_messages.iter().map(|m| m.value).sum::<f64>();
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
}
