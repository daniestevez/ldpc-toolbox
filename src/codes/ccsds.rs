//! CCSDS TM Synchronization and Channel Coding LDPC codes.
//!
//! This module contains the AR4JA LDPC codes and the C2 code described in the
//! TM Synchronization and Channel Coding Blue Book.
//!
//! ## References
//! \[1\] [CCSDS 131.0-B-5 TM Synchronization and Channel Coding Blue Book](https://public.ccsds.org/Pubs/131x0b5.pdf).

use crate::sparse::SparseMatrix;
use enum_iterator::Sequence;

/// AR4JA code definition.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct AR4JACode {
    rate: AR4JARate,
    k: AR4JAInfoSize,
}

/// AR4JA code rate.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Sequence)]
pub enum AR4JARate {
    /// Rate 1/2.
    R1_2,
    /// Rate 2/3.
    R2_3,
    /// Rate 4/5.
    R4_5,
}

/// AR4JA information block size `k`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Sequence)]
pub enum AR4JAInfoSize {
    /// k = 1024
    K1024,
    /// k = 4096,
    K4096,
    /// k = 16384,
    K16384,
}

impl AR4JACode {
    /// Creates an AR4JA code definition.
    pub fn new(rate: AR4JARate, information_block_size: AR4JAInfoSize) -> AR4JACode {
        AR4JACode {
            rate,
            k: information_block_size,
        }
    }

    /// Constructs the parity check matrix for the code.
    pub fn h(&self) -> SparseMatrix {
        let m = 1 << self.m().log2();
        let extra_column_blocks = match self.rate {
            AR4JARate::R1_2 => 0,
            AR4JARate::R2_3 => 2,
            AR4JARate::R4_5 => 6,
        };
        let extra_columns = m * extra_column_blocks;
        let mut h = SparseMatrix::new(3 * m, extra_columns + 5 * m);

        // fill common part (H_1/2)
        for i in 0..m {
            // block(0,2) = I_M
            h.insert(i, extra_columns + 2 * m + i);
            // block(0,4) = I_M + Pi_1
            h.insert(i, extra_columns + 4 * m + i);
            h.toggle(i, extra_columns + 4 * m + self.pi(1, i));
            // block(1,0) = I_M
            h.insert(m + i, extra_columns + i);
            // block(1,1) = I_M
            h.insert(m + i, extra_columns + m + i);
            // block(1,3) = I_M
            h.insert(m + i, extra_columns + 3 * m + i);
            // block(1,4) = Pi_2 + Pi_3 + Pi_4
            h.insert(m + i, extra_columns + 4 * m + self.pi(2, i));
            h.toggle(m + i, extra_columns + 4 * m + self.pi(3, i));
            h.toggle(m + i, extra_columns + 4 * m + self.pi(4, i));
            // block(2,0) = I_M
            h.insert(2 * m + i, extra_columns + i);
            // block(2,1) = Pi_5 + Pi_6
            h.insert(2 * m + i, extra_columns + m + self.pi(5, i));
            h.toggle(2 * m + i, extra_columns + m + self.pi(6, i));
            // block(2,3) = Pi_7 + Pi_8
            h.insert(2 * m + i, extra_columns + 3 * m + self.pi(7, i));
            h.toggle(2 * m + i, extra_columns + 3 * m + self.pi(8, i));
            // block(2,4) = I_M
            h.insert(2 * m + i, extra_columns + 4 * m + i);
        }

        if !matches!(self.rate, AR4JARate::R1_2) {
            // fill specific H_2/3 part
            let extra_columns = match self.rate {
                AR4JARate::R1_2 => unreachable!(),
                AR4JARate::R2_3 => 0,
                AR4JARate::R4_5 => 4 * m,
            };
            for i in 0..m {
                // block(1,0) = Pi_9 + Pi_10 + Pi_11
                h.insert(m + i, extra_columns + self.pi(9, i));
                h.toggle(m + i, extra_columns + self.pi(10, i));
                h.toggle(m + i, extra_columns + self.pi(11, i));
                // block(1,1) = I_M
                h.insert(m + i, extra_columns + m + i);
                // block(2,0) = I_M
                h.insert(2 * m + i, extra_columns + i);
                // block(2,1) = Pi_12 + Pi_13 + Pi_14
                h.insert(2 * m + i, extra_columns + m + self.pi(12, i));
                h.toggle(2 * m + i, extra_columns + m + self.pi(13, i));
                h.toggle(2 * m + i, extra_columns + m + self.pi(14, i));
            }
        }

        if matches!(self.rate, AR4JARate::R4_5) {
            // fill specific H_4/5 part
            for i in 0..m {
                // block(1,0) = Pi_21 + Pi_22 + Pi_23
                h.insert(m + i, self.pi(21, i));
                h.toggle(m + i, self.pi(22, i));
                h.toggle(m + i, self.pi(23, i));
                // block(1,1) = I_M
                h.insert(m + i, m + i);
                // block(1,2) = Pi_15 + Pi_16 + Pi_17
                h.insert(m + i, 2 * m + self.pi(15, i));
                h.toggle(m + i, 2 * m + self.pi(16, i));
                h.toggle(m + i, 2 * m + self.pi(17, i));
                // block(1,3) = I_M
                h.insert(m + i, 3 * m + i);
                // block(2,0) = I_M
                h.insert(2 * m + i, i);
                // block(2,1) = Pi_24 + Pi_25 + Pi_26
                h.insert(2 * m + i, m + self.pi(24, i));
                h.toggle(2 * m + i, m + self.pi(25, i));
                h.toggle(2 * m + i, m + self.pi(26, i));
                // block(2,2) = I_M
                h.insert(2 * m + i, 2 * m + i);
                // block(2,3) = Pi_18 + Pi_19 + Pi_20
                h.insert(2 * m + i, 3 * m + self.pi(18, i));
                h.toggle(2 * m + i, 3 * m + self.pi(19, i));
                h.toggle(2 * m + i, 3 * m + self.pi(20, i));
            }
        }

        h
    }

    // Table 7.2 in [1]
    fn m(&self) -> M {
        match (self.rate, self.k) {
            (AR4JARate::R1_2, AR4JAInfoSize::K1024) => M::M512,
            (AR4JARate::R2_3, AR4JAInfoSize::K1024) => M::M256,
            (AR4JARate::R4_5, AR4JAInfoSize::K1024) => M::M128,
            (AR4JARate::R1_2, AR4JAInfoSize::K4096) => M::M2048,
            (AR4JARate::R2_3, AR4JAInfoSize::K4096) => M::M1024,
            (AR4JARate::R4_5, AR4JAInfoSize::K4096) => M::M512,
            (AR4JARate::R1_2, AR4JAInfoSize::K16384) => M::M8192,
            (AR4JARate::R2_3, AR4JAInfoSize::K16384) => M::M4096,
            (AR4JARate::R4_5, AR4JAInfoSize::K16384) => M::M2048,
        }
    }

    // Table 7-3 and 7-4 in [1]
    fn theta(k: usize) -> usize {
        assert!((1..=26).contains(&k));
        THETA_K[k - 1].into()
    }

    // Table 7-3 and 7-4 in [1]
    fn phi(&self, k: usize, j: usize) -> usize {
        assert!((1..=26).contains(&k));
        assert!((0..4).contains(&j));
        let m_index = self.m().log2() - M::M128.log2();
        PHI_K[j][k - 1][m_index]
    }

    // Section 7.4.2.4 in [1]
    fn pi(&self, k: usize, i: usize) -> usize {
        let m_log2 = self.m().log2();
        let m = 1 << m_log2;
        let j = 4 * i / m;
        // & 0x3 gives mod 4
        let a = (Self::theta(k) + j) & 0x3;
        let m_div_4 = 1 << (m_log2 - 2);
        // & (m_div_4 - 1) gives mod M/4
        let b = (self.phi(k, j) + i) & (m_div_4 - 1);
        // << (m_log2 - 2) gives * M/4
        (a << (m_log2 - 2)) + b
    }
}

enum M {
    M128,
    M256,
    M512,
    M1024,
    M2048,
    M4096,
    M8192,
}

impl M {
    fn log2(&self) -> usize {
        match self {
            M::M128 => 7,
            M::M256 => 8,
            M::M512 => 9,
            M::M1024 => 10,
            M::M2048 => 11,
            M::M4096 => 12,
            M::M8192 => 13,
        }
    }
}

static THETA_K: [u8; 26] = [
    3, 0, 1, 2, 2, 3, 0, 1, 0, 1, 2, 0, 2, 3, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 3,
];

// Table 7-3 and 7-4 in [1]
static PHI_K: [[[usize; 7]; 26]; 4] = [
    // j = 0
    [
        [1, 59, 16, 160, 108, 226, 1148],
        [22, 18, 103, 241, 126, 618, 2032],
        [0, 52, 105, 185, 238, 404, 249],
        [26, 23, 0, 251, 481, 32, 1807],
        [0, 11, 50, 209, 96, 912, 485],
        [10, 7, 29, 103, 28, 950, 1044],
        [5, 22, 115, 90, 59, 534, 717],
        [18, 25, 30, 184, 225, 63, 873],
        [3, 27, 92, 248, 323, 971, 364],
        [22, 30, 78, 12, 28, 304, 1926],
        [3, 43, 70, 111, 386, 409, 1241],
        [8, 14, 66, 66, 305, 708, 1769],
        [25, 46, 39, 173, 34, 719, 532],
        [25, 62, 84, 42, 510, 176, 768],
        [2, 44, 79, 157, 147, 743, 1138],
        [27, 12, 70, 174, 199, 759, 965],
        [7, 38, 29, 104, 347, 674, 141],
        [7, 47, 32, 144, 391, 958, 1527],
        [15, 1, 45, 43, 165, 984, 505],
        [10, 52, 113, 181, 414, 11, 1312],
        [4, 61, 86, 250, 97, 413, 1840],
        [19, 10, 1, 202, 158, 925, 709],
        [7, 55, 42, 68, 86, 687, 1427],
        [9, 7, 118, 177, 168, 752, 989],
        [26, 12, 33, 170, 506, 867, 1925],
        [17, 2, 126, 89, 489, 323, 270],
    ],
    // j = 1
    [
        [0, 0, 0, 0, 0, 0, 0],
        [27, 32, 53, 182, 375, 767, 1822],
        [30, 21, 74, 249, 436, 227, 203],
        [28, 36, 45, 65, 350, 247, 882],
        [7, 30, 47, 70, 260, 284, 1989],
        [1, 29, 0, 141, 84, 370, 957],
        [8, 44, 59, 237, 318, 482, 1705],
        [20, 29, 102, 77, 382, 273, 1083],
        [26, 39, 25, 55, 169, 886, 1072],
        [24, 14, 3, 12, 213, 634, 354],
        [4, 22, 88, 227, 67, 762, 1942],
        [12, 15, 65, 42, 313, 184, 446],
        [23, 48, 62, 52, 242, 696, 1456],
        [15, 55, 68, 243, 188, 413, 1940],
        [15, 39, 91, 179, 1, 854, 1660],
        [22, 11, 70, 250, 306, 544, 1661],
        [31, 1, 115, 247, 397, 864, 587],
        [3, 50, 31, 164, 80, 82, 708],
        [29, 40, 121, 17, 33, 1009, 1466],
        [21, 62, 45, 31, 7, 437, 433],
        [2, 27, 56, 149, 447, 36, 1345],
        [5, 38, 54, 105, 336, 562, 867],
        [11, 40, 108, 183, 424, 816, 1551],
        [26, 15, 14, 153, 134, 452, 2041],
        [9, 11, 30, 177, 152, 290, 1383],
        [17, 18, 116, 19, 492, 778, 1790],
    ],
    // j = 2
    [
        [0, 0, 0, 0, 0, 0, 0],
        [12, 46, 8, 35, 219, 254, 318],
        [30, 45, 119, 167, 16, 790, 494],
        [18, 27, 89, 214, 263, 642, 1467],
        [10, 48, 31, 84, 415, 248, 757],
        [16, 37, 122, 206, 403, 899, 1085],
        [13, 41, 1, 122, 184, 328, 1630],
        [9, 13, 69, 67, 279, 518, 64],
        [7, 9, 92, 147, 198, 477, 689],
        [15, 49, 47, 54, 307, 404, 1300],
        [16, 36, 11, 23, 432, 698, 148],
        [18, 10, 31, 93, 240, 160, 777],
        [4, 11, 19, 20, 454, 497, 1431],
        [23, 18, 66, 197, 294, 100, 659],
        [5, 54, 49, 46, 479, 518, 352],
        [3, 40, 81, 162, 289, 92, 1177],
        [29, 27, 96, 101, 373, 464, 836],
        [11, 35, 38, 76, 104, 592, 1572],
        [4, 25, 83, 78, 141, 198, 348],
        [8, 46, 42, 253, 270, 856, 1040],
        [2, 24, 58, 124, 439, 235, 779],
        [11, 33, 24, 143, 333, 134, 476],
        [11, 18, 25, 63, 399, 542, 191],
        [3, 37, 92, 41, 14, 545, 1393],
        [15, 35, 38, 214, 277, 777, 1752],
        [13, 21, 120, 70, 412, 483, 1627],
    ],
    // j = 3
    [
        [0, 0, 0, 0, 0, 0, 0],
        [13, 44, 35, 162, 312, 285, 1189],
        [19, 51, 97, 7, 503, 554, 458],
        [14, 12, 112, 31, 388, 809, 460],
        [15, 15, 64, 164, 48, 185, 1039],
        [20, 12, 93, 11, 7, 49, 1000],
        [17, 4, 99, 237, 185, 101, 1265],
        [4, 7, 94, 125, 328, 82, 1223],
        [4, 2, 103, 133, 254, 898, 874],
        [11, 30, 91, 99, 202, 627, 1292],
        [17, 53, 3, 105, 285, 154, 1491],
        [20, 23, 6, 17, 11, 65, 631],
        [8, 29, 39, 97, 168, 81, 464],
        [22, 37, 113, 91, 127, 823, 461],
        [19, 42, 92, 211, 8, 50, 844],
        [15, 48, 119, 128, 437, 413, 392],
        [5, 4, 74, 82, 475, 462, 922],
        [21, 10, 73, 115, 85, 175, 256],
        [17, 18, 116, 248, 419, 715, 1986],
        [9, 56, 31, 62, 459, 537, 19],
        [20, 9, 127, 26, 468, 722, 266],
        [18, 11, 98, 140, 209, 37, 471],
        [31, 23, 23, 121, 311, 488, 1166],
        [13, 8, 38, 12, 211, 179, 1300],
        [2, 7, 18, 41, 510, 430, 1033],
        [18, 24, 62, 249, 320, 264, 1606],
    ],
];

/// C2 code definition.
///
/// This C2 code is the basic (8176, 7156) LDPC code. Expurgation, shortening
/// and extension used to construct the (8160, 7136) code should be handled
/// separately.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub struct C2Code {}

impl C2Code {
    /// Creates a C2 code definition.
    pub fn new() -> C2Code {
        C2Code::default()
    }

    /// Constructs the parity check matrix for the code.
    pub fn h(&self) -> SparseMatrix {
        const N: usize = 511;
        let mut h = SparseMatrix::new(Self::ROW_BLOCKS * N, Self::COL_BLOCKS * N);
        for (row, circs) in C2_CIRCULANTS.iter().enumerate() {
            for (col, circs) in circs.iter().enumerate() {
                for &circ in circs.iter() {
                    let circ = usize::from(circ);
                    for j in 0..N {
                        h.insert(row * N + j, col * N + (j + circ) % N);
                    }
                }
            }
        }
        h
    }

    const ROW_BLOCKS: usize = 2;
    const COL_BLOCKS: usize = 16;
    const BLOCK_WEIGHT: usize = 2;
}

// Table 7-1 in CCSDS 131.0-B-5
static C2_CIRCULANTS: [[[u16; C2Code::BLOCK_WEIGHT]; C2Code::COL_BLOCKS]; C2Code::ROW_BLOCKS] = [
    [
        [0, 176],
        [12, 239],
        [0, 352],
        [24, 431],
        [0, 392],
        [151, 409],
        [0, 351],
        [9, 359],
        [0, 307],
        [53, 329],
        [0, 207],
        [18, 281],
        [0, 399],
        [202, 457],
        [0, 247],
        [36, 261],
    ],
    [
        [99, 471],
        [130, 473],
        [198, 435],
        [260, 478],
        [215, 420],
        [282, 481],
        [48, 396],
        [193, 445],
        [273, 430],
        [302, 451],
        [96, 379],
        [191, 386],
        [244, 467],
        [364, 470],
        [51, 382],
        [192, 414],
    ],
];

#[cfg(test)]
mod test {
    use super::*;

    fn pi_k_model(code: &AR4JACode, k: usize, i: usize) -> usize {
        let m = 1 << code.m().log2();
        let theta_k = AR4JACode::theta(k);
        let phi_k = code.phi(k, 4 * i / m);
        m / 4 * ((theta_k + (4 * i / m)) % 4) + (phi_k + i) % (m / 4)
    }

    // Checks that AR4JACode::pi matches the simpler (but less efficient)
    // implementation given in pi_k_model.
    #[test]
    fn pi_k() {
        for rate in enum_iterator::all() {
            for info_k in enum_iterator::all() {
                let code = AR4JACode::new(rate, info_k);
                let m = 1 << code.m().log2();
                for k in 1..=26 {
                    for i in 0..m {
                        assert_eq!(code.pi(k, i), pi_k_model(&code, k, i));
                    }
                }
            }
        }
    }
}
