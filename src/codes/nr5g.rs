//! # 5G NR LDPC codes
//!
//! This module contains the LDPC codes used in 5G NR.
//!
//! The code definitions are handled as variants of the [`Code`] enum,
//! which defines methods to work with the codes.
//!
//! ## References
//! \[1\] [3GPP TS 38.212 V18.6.0 (2025-04)](https://www.etsi.org/deliver/etsi_ts/138200_138299/138212/18.06.00_60/ts_138212v180600p.pdf)

use crate::sparse::SparseMatrix;
use clap::ValueEnum;

/// 5G NR LDPC base graph.
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum BaseGraph {
    /// 5G NR LDPC base graph 1.
    #[value(name = "1")]
    BG1,
    /// 5G NR LDPC base graph 2.
    #[value(name = "2")]
    BG2,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct VijRow {
    column_index: usize,
    vij: [usize; 8],
}

impl VijRow {
    fn vij(&self, set_index: SetIndex) -> usize {
        self.vij[usize::from(set_index)]
    }
}

impl BaseGraph {
    /// Constructs the parity check matrix for this base graph with a given
    /// lifting size.
    pub fn h(&self, lifting_size: LiftingSize) -> SparseMatrix {
        let zc = usize::from(lifting_size);
        let mut h = SparseMatrix::new(self.num_rows() * zc, self.num_cols() * zc);
        for (j, rows) in self.graph().iter().enumerate() {
            for row in rows {
                let k = row.column_index;
                let vij = row.vij(lifting_size.set_index());
                for r in 0..zc {
                    h.insert(zc * j + r, zc * k + ((r + vij) % zc));
                }
            }
        }
        h
    }

    fn graph(&self) -> Box<[Vec<VijRow>]> {
        match self {
            BaseGraph::BG1 => base_graph_1().into(),
            BaseGraph::BG2 => base_graph_2().into(),
        }
    }

    fn num_rows(&self) -> usize {
        self.graph().len()
    }

    fn num_cols(&self) -> usize {
        match self {
            BaseGraph::BG1 => 68,
            BaseGraph::BG2 => 52,
        }
    }
}

/// 5G NR LDPC lifting size.
///
/// This enum lists the LDPC lifiting sizes defined in TS 38.212 Table 5.3.2-1.
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum LiftingSize {
    /// 5G NR lifting size 2.
    #[value(name = "2")]
    Z2,
    /// 5G NR lifting size 4.
    #[value(name = "4")]
    Z4,
    /// 5G NR lifting size 8.
    #[value(name = "8")]
    Z8,
    /// 5G NR lifting size 16.
    #[value(name = "16")]
    Z16,
    /// 5G NR lifting size 32.
    #[value(name = "32")]
    Z32,
    /// 5G NR lifting size 64.
    #[value(name = "64")]
    Z64,
    /// 5G NR lifting size 128.
    #[value(name = "128")]
    Z128,
    /// 5G NR lifting size 256.
    #[value(name = "256")]
    Z256,
    /// 5G NR lifting size 3.
    #[value(name = "3")]
    Z3,
    /// 5G NR lifting size 6.
    #[value(name = "6")]
    Z6,
    /// 5G NR lifting size 12.
    #[value(name = "12")]
    Z12,
    /// 5G NR lifting size 24.
    #[value(name = "24")]
    Z24,
    /// 5G NR lifting size 48.
    #[value(name = "48")]
    Z48,
    /// 5G NR lifting size 96.
    #[value(name = "96")]
    Z96,
    /// 5G NR lifting size 192.
    #[value(name = "192")]
    Z192,
    /// 5G NR lifting size 384.
    #[value(name = "384")]
    Z384,
    /// 5G NR lifting size 5.
    #[value(name = "5")]
    Z5,
    /// 5G NR lifting size 10.
    #[value(name = "10")]
    Z10,
    /// 5G NR lifting size 20.
    #[value(name = "20")]
    Z20,
    /// 5G NR lifting size 40.
    #[value(name = "40")]
    Z40,
    /// 5G NR lifting size 80.
    #[value(name = "80")]
    Z80,
    /// 5G NR lifting ize 160.
    #[value(name = "160")]
    Z160,
    /// 5G NR lifting size 320.
    #[value(name = "320")]
    Z320,
    /// 5G NR lifting size 7.
    #[value(name = "7")]
    Z7,
    /// 5G NR lifting size 14.
    #[value(name = "14")]
    Z14,
    /// 5G NR lifting size 28.
    #[value(name = "28")]
    Z28,
    /// 5G NR lifting size 56.
    #[value(name = "56")]
    Z56,
    /// 5G NR lifting size 112.
    #[value(name = "112")]
    Z112,
    /// 5G NR lifting size 224.
    #[value(name = "224")]
    Z224,
    /// 5G NR lifting size 9.
    #[value(name = "9")]
    Z9,
    /// 5G NR lifting size 18.
    #[value(name = "18")]
    Z18,
    /// 5G NR lifting size 36.
    #[value(name = "36")]
    Z36,
    /// 5G NR lifting size 72.
    #[value(name = "72")]
    Z72,
    /// 5G NR lifting size 144.
    #[value(name = "144")]
    Z144,
    /// 5G NR lifting size 288.
    #[value(name = "288")]
    Z288,
    /// 5G NR lifting size 11.
    #[value(name = "11")]
    Z11,
    /// 5G NR lifting size 22.
    #[value(name = "22")]
    Z22,
    /// 5G NR lifting size 44.
    #[value(name = "44")]
    Z44,
    /// 5G NR lifting size 88.
    #[value(name = "88")]
    Z88,
    /// 5G NR lifting size 176.
    #[value(name = "176")]
    Z176,
    /// 5G NR lifting size 352.
    #[value(name = "352")]
    Z352,
    /// 5G NR lifting size 13.
    #[value(name = "13")]
    Z13,
    /// 5G NR lifting size 26.
    #[value(name = "26")]
    Z26,
    /// 5G NR lifting size 52.
    #[value(name = "52")]
    Z52,
    /// 5G NR lifting size 104.
    #[value(name = "104")]
    Z104,
    /// 5G NR lifting size 204.
    #[value(name = "204")]
    Z208,
    /// 5G NR lifting size 15.
    #[value(name = "15")]
    Z15,
    /// 5G NR lifting size 30.
    #[value(name = "30")]
    Z30,
    /// 5G NR lifting size 60.
    #[value(name = "60")]
    Z60,
    /// 5G NR lifting size 120.
    #[value(name = "120")]
    Z120,
    /// 5G NR lifting size 240.
    #[value(name = "240")]
    Z240,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum SetIndex {
    Ils0,
    Ils1,
    Ils2,
    Ils3,
    Ils4,
    Ils5,
    Ils6,
    Ils7,
}

impl LiftingSize {
    fn set_index(&self) -> SetIndex {
        use LiftingSize::*;
        use SetIndex::*;
        match self {
            Z2 | Z4 | Z8 | Z16 | Z32 | Z64 | Z128 | Z256 => Ils0,
            Z3 | Z6 | Z12 | Z24 | Z48 | Z96 | Z192 | Z384 => Ils1,
            Z5 | Z10 | Z20 | Z40 | Z80 | Z160 | Z320 => Ils2,
            Z7 | Z14 | Z28 | Z56 | Z112 | Z224 => Ils3,
            Z9 | Z18 | Z36 | Z72 | Z144 | Z288 => Ils4,
            Z11 | Z22 | Z44 | Z88 | Z176 | Z352 => Ils5,
            Z13 | Z26 | Z52 | Z104 | Z208 => Ils6,
            Z15 | Z30 | Z60 | Z120 | Z240 => Ils7,
        }
    }
}

impl From<LiftingSize> for usize {
    fn from(value: LiftingSize) -> usize {
        use LiftingSize::*;
        match value {
            Z2 => 2,
            Z4 => 4,
            Z8 => 8,
            Z16 => 16,
            Z32 => 32,
            Z64 => 64,
            Z128 => 128,
            Z256 => 256,
            Z3 => 3,
            Z6 => 6,
            Z12 => 12,
            Z24 => 24,
            Z48 => 48,
            Z96 => 96,
            Z192 => 192,
            Z384 => 384,
            Z5 => 5,
            Z10 => 10,
            Z20 => 20,
            Z40 => 40,
            Z80 => 80,
            Z160 => 160,
            Z320 => 320,
            Z7 => 7,
            Z14 => 14,
            Z28 => 28,
            Z56 => 56,
            Z112 => 112,
            Z224 => 224,
            Z9 => 9,
            Z18 => 18,
            Z36 => 36,
            Z72 => 72,
            Z144 => 144,
            Z288 => 288,
            Z11 => 11,
            Z22 => 22,
            Z44 => 44,
            Z88 => 88,
            Z176 => 176,
            Z352 => 352,
            Z13 => 13,
            Z26 => 26,
            Z52 => 52,
            Z104 => 104,
            Z208 => 208,
            Z15 => 15,
            Z30 => 30,
            Z60 => 60,
            Z120 => 120,
            Z240 => 240,
        }
    }
}

impl From<SetIndex> for usize {
    fn from(value: SetIndex) -> usize {
        use SetIndex::*;
        match value {
            Ils0 => 0,
            Ils1 => 1,
            Ils2 => 2,
            Ils3 => 3,
            Ils4 => 4,
            Ils5 => 5,
            Ils6 => 6,
            Ils7 => 7,
        }
    }
}

macro_rules! row {
    ($col:literal $($row:literal)+) => {
        VijRow {
            column_index: $col,
            vij: [$($row),+],
        }
    }
}

// TS 38.212 Table 5.3.2-2
fn base_graph_1() -> [Vec<VijRow>; 46] {
    [
        // i = 0
        vec![
            row!(0 250 307 73 223 211 294 0 135),
            row!(1 69 19 15 16 198 118 0 227),
            row!(2 226 50 103 94 188 167 0 126),
            row!(3 159 369 49 91 186 330 0 134),
            row!(5 100 181 240 74 219 207 0 84),
            row!(6 10 216 39 10 4 165 0 83),
            row!(9 59 317 15 0 29 243 0 53),
            row!(10 229 288 162 205 144 250 0 225),
            row!(11 110 109 215 216 116 1 0 205),
            row!(12 191 17 164 21 216 339 0 128),
            row!(13 9 357 133 215 115 201 0 75),
            row!(15 195 215 298 14 233 53 0 135),
            row!(16 23 106 110 70 144 347 0 217),
            row!(18 190 242 113 141 95 304 0 220),
            row!(19 35 180 16 198 216 167 0 90),
            row!(20 239 330 189 104 73 47 0 105),
            row!(21 31 346 32 81 261 188 0 137),
            row!(22 1 1 1 1 1 1 0 1),
            row!(23 0 0 0 0 0 0 0 0),
        ],
        // i = 1
        vec![
            row!(0 2 76 303 141 179 77 22 96),
            row!(2 239 76 294 45 162 225 11 236),
            row!(3 117 73 27 151 223 96 124 136),
            row!(4 124 288 261 46 256 338 0 221),
            row!(5 71 144 161 119 160 268 10 128),
            row!(7 222 331 133 157 76 112 0 92),
            row!(8 104 331 4 133 202 302 0 172),
            row!(9 173 178 80 87 117 50 2 56),
            row!(11 220 295 129 206 109 167 16 11),
            row!(12 102 342 300 93 15 253 60 189),
            row!(14 109 217 76 79 72 334 0 95),
            row!(15 132 99 266 9 152 242 6 85),
            row!(16 142 354 72 118 158 257 30 153),
            row!(17 155 114 83 194 147 133 0 87),
            row!(19 255 331 260 31 156 9 168 163),
            row!(21 28 112 301 187 119 302 31 216),
            row!(22 0 0 0 0 0 0 105 0),
            row!(23 0 0 0 0 0 0 0 0),
            row!(24 0 0 0 0 0 0 0 0),
        ],
        // i = 2
        vec![
            row!(0 106 205 68 207 258 226 132 189),
            row!(1 111 250 7 203 167 35 37 4),
            row!(2 185 328 80 31 220 213 21 225),
            row!(4 63 332 280 176 133 302 180 151),
            row!(5 117 256 38 180 243 111 4 236),
            row!(6 93 161 227 186 202 265 149 117),
            row!(7 229 267 202 95 218 128 48 179),
            row!(8 177 160 200 153 63 237 38 92),
            row!(9 95 63 71 177 0 294 122 24),
            row!(10 39 129 106 70 3 127 195 68),
            row!(13 142 200 295 77 74 110 155 6),
            row!(14 225 88 283 214 229 286 28 101),
            row!(15 225 53 301 77 0 125 85 33),
            row!(17 245 131 184 198 216 131 47 96),
            row!(18 205 240 246 117 269 163 179 125),
            row!(19 251 205 230 223 200 210 42 67),
            row!(20 117 13 276 90 234 7 66 230),
            row!(24 0 0 0 0 0 0 0 0),
            row!(25 0 0 0 0 0 0 0 0),
        ],
        // i = 3
        vec![
            row!(0 121 276 220 201 187 97 4 128),
            row!(1 89 87 208 18 145 94 6 23),
            row!(3 84 0 30 165 166 49 33 162),
            row!(4 20 275 197 5 108 279 113 220),
            row!(6 150 199 61 45 82 139 49 43),
            row!(7 131 153 175 142 132 166 21 186),
            row!(8 243 56 79 16 197 91 6 96),
            row!(10 136 132 281 34 41 106 151 1),
            row!(11 86 305 303 155 162 246 83 216),
            row!(12 246 231 253 213 57 345 154 22),
            row!(13 219 341 164 147 36 269 87 24),
            row!(14 211 212 53 69 115 185 5 167),
            row!(16 240 304 44 96 242 249 92 200),
            row!(17 76 300 28 74 165 215 173 32),
            row!(18 244 271 77 99 0 143 120 235),
            row!(20 144 39 319 30 113 121 2 172),
            row!(21 12 357 68 158 108 121 142 219),
            row!(22 1 1 1 1 1 1 0 1),
            row!(25 0 0 0 0 0 0 0 0),
        ],
        // i = 4
        vec![
            row!(0 157 332 233 170 246 42 24 64),
            row!(1 102 181 205 10 235 256 204 211),
            row!(26 0 0 0 0 0 0 0 0),
        ],
        // i = 5
        vec![
            row!(0 205 195 83 164 261 219 185 2),
            row!(1 236 14 292 59 181 130 100 171),
            row!(3 194 115 50 86 72 251 24 47),
            row!(12 231 166 318 80 283 322 65 143),
            row!(16 28 241 201 182 254 295 207 210),
            row!(21 123 51 267 130 79 258 161 180),
            row!(22 115 157 279 153 144 283 72 180),
            row!(27 0 0 0 0 0 0 0 0),
        ],
        // i = 6
        vec![
            row!(0 183 278 289 158 80 294 6 199),
            row!(6 22 257 21 119 144 73 27 22),
            row!(10 28 1 293 113 169 330 163 23),
            row!(11 67 351 13 21 90 99 50 100),
            row!(13 244 92 232 63 59 172 48 92),
            row!(17 11 253 302 51 177 150 24 207),
            row!(18 157 18 138 136 151 284 38 52),
            row!(20 211 225 235 116 108 305 91 13),
            row!(28 0 0 0 0 0 0 0 0),
        ],
        // i = 7
        vec![
            row!(0 220 9 12 17 169 3 145 77),
            row!(1 44 62 88 76 189 103 88 146),
            row!(4 159 316 207 104 154 224 112 209),
            row!(7 31 333 50 100 184 297 153 32),
            row!(8 167 290 25 150 104 215 159 166),
            row!(14 104 114 76 158 164 39 76 18),
            row!(29 0 0 0 0 0 0 0 0),
        ],
        // i = 8
        vec![
            row!(0 112 307 295 33 54 348 172 181),
            row!(1 4 179 133 95 0 75 2 105),
            row!(3 7 165 130 4 252 22 131 141),
            row!(12 211 18 231 217 41 312 141 223),
            row!(16 102 39 296 204 98 224 96 177),
            row!(19 164 224 110 39 46 17 99 145),
            row!(21 109 368 269 58 15 59 101 199),
            row!(22 241 67 245 44 230 314 35 153),
            row!(24 90 170 154 201 54 244 116 38),
            row!(30 0 0 0 0 0 0 0 0),
        ],
        // i = 9
        vec![
            row!(0 103 366 189 9 162 156 6 169),
            row!(1 182 232 244 37 159 88 10 12),
            row!(10 109 321 36 213 93 293 145 206),
            row!(11 21 133 286 105 134 111 53 221),
            row!(13 142 57 151 89 45 92 201 17),
            row!(17 14 303 267 185 132 152 4 212),
            row!(18 61 63 135 109 76 23 164 92),
            row!(20 216 82 209 218 209 337 173 205),
            row!(31 0 0 0 0 0 0 0 0),
        ],
        // i = 10
        vec![
            row!(1 98 101 14 82 178 175 126 116),
            row!(2 149 339 80 165 1 253 77 151),
            row!(4 167 274 211 174 28 27 156 70),
            row!(7 160 111 75 19 267 231 16 230),
            row!(8 49 383 161 194 234 49 12 115),
            row!(14 58 354 311 103 201 267 70 84),
            row!(32 0 0 0 0 0 0 0 0),
        ],
        // i = 11
        vec![
            row!(0 77 48 16 52 55 25 184 45),
            row!(1 41 102 147 11 23 322 194 115),
            row!(12 83 8 290 2 274 200 123 134),
            row!(16 182 47 289 35 181 351 16 1),
            row!(21 78 188 177 32 273 166 104 152),
            row!(22 252 334 43 84 39 338 109 165),
            row!(23 22 115 280 201 26 192 124 107),
            row!(33 0 0 0 0 0 0 0 0),
        ],
        // i = 12
        vec![
            row!(0 160 77 229 142 225 123 6 186),
            row!(1 42 186 235 175 162 217 20 215),
            row!(10 21 174 169 136 244 142 203 124),
            row!(11 32 232 48 3 151 110 153 180),
            row!(13 234 50 105 28 238 176 104 98),
            row!(18 7 74 52 182 243 76 207 80),
            row!(34 0 0 0 0 0 0 0 0),
        ],
        // i = 13
        vec![
            row!(0 177 313 39 81 231 311 52 220),
            row!(3 248 177 302 56 0 251 147 185),
            row!(7 151 266 303 72 216 265 1 154),
            row!(20 185 115 160 217 47 94 16 178),
            row!(23 62 370 37 78 36 81 46 150),
            row!(35 0 0 0 0 0 0 0 0),
        ],
        // i = 14
        vec![
            row!(0 206 142 78 14 0 22 1 124),
            row!(12 55 248 299 175 186 322 202 144),
            row!(15 206 137 54 211 253 277 118 182),
            row!(16 127 89 61 191 16 156 130 95),
            row!(17 16 347 179 51 0 66 1 72),
            row!(21 229 12 258 43 79 78 2 76),
            row!(36 0 0 0 0 0 0 0 0),
        ],
        // i = 15
        vec![
            row!(0 40 241 229 90 170 176 173 39),
            row!(1 96 2 290 120 0 348 6 138),
            row!(10 65 210 60 131 183 15 81 220),
            row!(13 63 318 130 209 108 81 182 173),
            row!(18 75 55 184 209 68 176 53 142),
            row!(25 179 269 51 81 64 113 46 49),
            row!(37 0 0 0 0 0 0 0 0),
        ],
        // i = 16
        vec![
            row!(1 64 13 69 154 270 190 88 78),
            row!(3 49 338 140 164 13 293 198 152),
            row!(11 49 57 45 43 99 332 160 84),
            row!(20 51 289 115 189 54 331 122 5),
            row!(22 154 57 300 101 0 114 182 205),
            row!(38 0 0 0 0 0 0 0 0),
        ],
        // i = 17
        vec![
            row!(0 7 260 257 56 153 110 91 183),
            row!(14 164 303 147 110 137 228 184 112),
            row!(16 59 81 128 200 0 247 30 106),
            row!(17 1 358 51 63 0 116 3 219),
            row!(21 144 375 228 4 162 190 155 129),
            row!(39 0 0 0 0 0 0 0 0),
        ],
        // i = 18
        vec![
            row!(1 42 130 260 199 161 47 1 183),
            row!(12 233 163 294 110 151 286 41 215),
            row!(13 8 280 291 200 0 246 167 180),
            row!(18 155 132 141 143 241 181 68 143),
            row!(19 147 4 295 186 144 73 148 14),
            row!(40 0 0 0 0 0 0 0 0),
        ],
        // i = 19
        vec![
            row!(0 60 145 64 8 0 87 12 179),
            row!(1 73 213 181 6 0 110 6 108),
            row!(7 72 344 101 103 118 147 166 159),
            row!(8 127 242 270 198 144 258 184 138),
            row!(10 224 197 41 8 0 204 191 196),
            row!(41 0 0 0 0 0 0 0 0),
        ],
        // i = 20
        vec![
            row!(0 151 187 301 105 265 89 6 77),
            row!(3 186 206 162 210 81 65 12 187),
            row!(9 217 264 40 121 90 155 15 203),
            row!(11 47 341 130 214 144 244 5 167),
            row!(22 160 59 10 183 228 30 30 130),
            row!(42 0 0 0 0 0 0 0 0),
        ],
        // i = 21
        vec![
            row!(1 249 205 79 192 64 162 6 197),
            row!(5 121 102 175 131 46 264 86 122),
            row!(16 109 328 132 220 266 346 96 215),
            row!(20 131 213 283 50 9 143 42 65),
            row!(21 171 97 103 106 18 109 199 216),
            row!(43 0 0 0 0 0 0 0 0),
        ],
        // i = 22
        vec![
            row!(0 64 30 177 53 72 280 44 25),
            row!(12 142 11 20 0 189 157 58 47),
            row!(13 188 233 55 3 72 236 130 126),
            row!(17 158 22 316 148 257 113 131 178),
            row!(44 0 0 0 0 0 0 0 0),
        ],
        // i = 23
        vec![
            row!(1 156 24 249 88 180 18 45 185),
            row!(2 147 89 50 203 0 6 18 127),
            row!(10 170 61 133 168 0 181 132 117),
            row!(18 152 27 105 122 165 304 100 199),
            row!(45 0 0 0 0 0 0 0 0),
        ],
        // i = 24
        vec![
            row!(0 112 298 289 49 236 38 9 32),
            row!(3 86 158 280 157 199 170 125 178),
            row!(4 236 235 110 64 0 249 191 2),
            row!(11 116 339 187 193 266 288 28 156),
            row!(22 222 234 281 124 0 194 6 58),
            row!(46 0 0 0 0 0 0 0 0),
        ],
        // i = 25
        vec![
            row!(1 23 72 172 1 205 279 4 27),
            row!(6 136 17 295 166 0 255 74 141),
            row!(7 116 383 96 65 0 111 16 11),
            row!(14 182 312 46 81 183 54 28 181),
            row!(47 0 0 0 0 0 0 0 0),
        ],
        // i = 26
        vec![
            row!(0 195 71 270 107 0 325 21 163),
            row!(2 243 81 110 176 0 326 142 131),
            row!(4 215 76 318 212 0 226 192 169),
            row!(15 61 136 67 127 277 99 197 98),
            row!(48 0 0 0 0 0 0 0 0),
        ],
        // i = 27
        vec![
            row!(1 25 194 210 208 45 91 98 165),
            row!(6 104 194 29 141 36 326 140 232),
            row!(8 194 101 304 174 72 268 22 9),
            row!(49 0 0 0 0 0 0 0 0),
        ],
        // i = 28
        vec![
            row!(0 128 222 11 146 275 102 4 32),
            row!(4 165 19 293 153 0 1 1 43),
            row!(19 181 244 50 217 155 40 40 200),
            row!(21 63 274 234 114 62 167 93 205),
            row!(50 0 0 0 0 0 0 0 0),
        ],
        // i = 29
        vec![
            row!(1 86 252 27 150 0 273 92 232),
            row!(14 236 5 308 11 180 104 136 32),
            row!(18 84 147 117 53 0 243 106 118),
            row!(25 6 78 29 68 42 107 6 103),
            row!(51 0 0 0 0 0 0 0 0),
        ],
        // i = 30
        vec![
            row!(0 216 159 91 34 0 171 2 170),
            row!(10 73 229 23 130 90 16 88 199),
            row!(13 120 260 105 210 252 95 112 26),
            row!(24 9 90 135 123 173 212 20 105),
            row!(52 0 0 0 0 0 0 0 0),
        ],
        // i = 31
        vec![
            row!(1 95 100 222 175 144 101 4 73),
            row!(7 177 215 308 49 144 297 49 149),
            row!(22 172 258 66 177 166 279 125 175),
            row!(25 61 256 162 128 19 222 194 108),
            row!(53 0 0 0 0 0 0 0 0),
        ],
        // i = 32
        vec![
            row!(0 221 102 210 192 0 351 6 103),
            row!(12 112 201 22 209 211 265 126 110),
            row!(14 199 175 271 58 36 338 63 151),
            row!(24 121 287 217 30 162 83 20 211),
            row!(54 0 0 0 0 0 0 0 0),
        ],
        // i = 33
        vec![
            row!(1 2 323 170 114 0 56 10 199),
            row!(2 187 8 20 49 0 304 30 132),
            row!(11 41 361 140 161 76 141 6 172),
            row!(21 211 105 33 137 18 101 92 65),
            row!(55 0 0 0 0 0 0 0 0),
        ],
        // i = 34
        vec![
            row!(0 127 230 187 82 197 60 4 161),
            row!(7 167 148 296 186 0 320 153 237),
            row!(15 164 202 5 68 108 112 197 142),
            row!(17 159 312 44 150 0 54 155 180),
            row!(56 0 0 0 0 0 0 0 0),
        ],
        // i = 35
        vec![
            row!(1 161 320 207 192 199 100 4 231),
            row!(6 197 335 158 173 278 210 45 174),
            row!(12 207 2 55 26 0 195 168 145),
            row!(22 103 266 285 187 205 268 185 100),
            row!(57 0 0 0 0 0 0 0 0),
        ],
        // i = 36
        vec![
            row!(0 37 210 259 222 216 135 6 11),
            row!(14 105 313 179 157 16 15 200 207),
            row!(15 51 297 178 0 0 35 177 42),
            row!(18 120 21 160 6 0 188 43 100),
            row!(58 0 0 0 0 0 0 0 0),
        ],
        // i = 37
        vec![
            row!(1 198 269 298 81 72 319 82 59),
            row!(13 220 82 15 195 144 236 2 204),
            row!(23 122 115 115 138 0 85 135 161),
            row!(59 0 0 0 0 0 0 0 0),
        ],
        // i = 38
        vec![
            row!(0 167 185 151 123 190 164 91 121),
            row!(9 151 177 179 90 0 196 64 90),
            row!(10 157 289 64 73 0 209 198 26),
            row!(12 163 214 181 10 0 246 100 140),
            row!(60 0 0 0 0 0 0 0 0),
        ],
        // i = 39
        vec![
            row!(1 173 258 102 12 153 236 4 115),
            row!(3 139 93 77 77 0 264 28 188),
            row!(7 149 346 192 49 165 37 109 168),
            row!(19 0 297 208 114 117 272 188 52),
            row!(61 0 0 0 0 0 0 0 0),
        ],
        // i = 40
        vec![
            row!(0 157 175 32 67 216 304 10 4),
            row!(8 137 37 80 45 144 237 84 103),
            row!(17 149 312 197 96 2 135 12 30),
            row!(62 0 0 0 0 0 0 0 0),
        ],
        // i = 41
        vec![
            row!(1 167 52 154 23 0 123 2 53),
            row!(3 173 314 47 215 0 77 75 189),
            row!(9 139 139 124 60 0 25 142 215),
            row!(18 151 288 207 167 183 272 128 24),
            row!(63 0 0 0 0 0 0 0 0),
        ],
        // i = 42
        vec![
            row!(0 149 113 226 114 27 288 163 222),
            row!(4 157 14 65 91 0 83 10 170),
            row!(24 137 218 126 78 35 17 162 71),
            row!(64 0 0 0 0 0 0 0 0),
        ],
        // i = 43
        vec![
            row!(1 151 113 228 206 52 210 1 22),
            row!(16 163 132 69 22 243 3 163 127),
            row!(18 173 114 176 134 0 53 99 49),
            row!(25 139 168 102 161 270 167 98 125),
            row!(65 0 0 0 0 0 0 0 0),
        ],
        // i = 44
        vec![
            row!(0 139 80 234 84 18 79 4 191),
            row!(7 157 78 227 4 0 244 6 211),
            row!(9 163 163 259 9 0 293 142 187),
            row!(22 173 274 260 12 57 272 3 148),
            row!(66 0 0 0 0 0 0 0 0),
        ],
        // i = 45
        vec![
            row!(1 149 135 101 184 168 82 181 177),
            row!(6 151 149 228 121 0 67 45 114),
            row!(10 167 15 126 29 144 235 153 93),
            row!(67 0 0 0 0 0 0 0 0),
        ],
    ]
}

// TS 38.212 Table 5.3.2-3
fn base_graph_2() -> [Vec<VijRow>; 42] {
    [
        // i = 0
        vec![
            row!(0 9 174 0 72 3 156 143 145),
            row!(1 117 97 0 110 26 143 19 131),
            row!(2 204 166 0 23 53 14 176 71),
            row!(3 26 66 0 181 35 3 165 21),
            row!(6 189 71 0 95 115 40 196 23),
            row!(9 205 172 0 8 127 123 13 112),
            row!(10 0 0 0 1 0 0 0 1),
            row!(11 0 0 0 0 0 0 0 0),
        ],
        // i = 1
        vec![
            row!(0 167 27 137 53 19 17 18 142),
            row!(3 166 36 124 156 94 65 27 174),
            row!(4 253 48 0 115 104 63 3 183),
            row!(5 125 92 0 156 66 1 102 27),
            row!(6 226 31 88 115 84 55 185 96),
            row!(7 156 187 0 200 98 37 17 23),
            row!(8 224 185 0 29 69 171 14 9),
            row!(9 252 3 55 31 50 133 180 167),
            row!(11 0 0 0 0 0 0 0 0),
            row!(12 0 0 0 0 0 0 0 0),
        ],
        // i = 2
        vec![
            row!(0 81 25 20 152 95 98 126 74),
            row!(1 114 114 94 131 106 168 163 31),
            row!(3 44 117 99 46 92 107 47 3),
            row!(4 52 110 9 191 110 82 183 53),
            row!(8 240 114 108 91 111 142 132 155),
            row!(10 1 1 1 0 1 1 1 0),
            row!(12 0 0 0 0 0 0 0 0),
            row!(13 0 0 0 0 0 0 0 0),
        ],
        // i = 3
        vec![
            row!(1 8 136 38 185 120 53 36 239),
            row!(2 58 175 15 6 121 174 48 171),
            row!(4 158 113 102 36 22 174 18 95),
            row!(5 104 72 146 124 4 127 111 110),
            row!(6 209 123 12 124 73 17 203 159),
            row!(7 54 118 57 110 49 89 3 199),
            row!(8 18 28 53 156 128 17 191 43),
            row!(9 128 186 46 133 79 105 160 75),
            row!(10 0 0 0 1 0 0 0 1),
            row!(13 0 0 0 0 0 0 0 0),
        ],
        // i = 4
        vec![
            row!(0 179 72 0 200 42 86 43 29),
            row!(1 214 74 136 16 24 67 27 140),
            row!(11 71 29 157 101 51 83 117 180),
            row!(14 0 0 0 0 0 0 0 0),
        ],
        // i = 5
        vec![
            row!(0 231 10 0 185 40 79 136 121),
            row!(1 41 44 131 138 140 84 49 41),
            row!(5 194 121 142 170 84 35 36 169),
            row!(7 159 80 141 219 137 103 132 88),
            row!(11 103 48 64 193 71 60 62 207),
            row!(15 0 0 0 0 0 0 0 0),
        ],
        // i = 6
        vec![
            row!(0 155 129 0 123 109 47 7 137),
            row!(5 228 92 124 55 87 154 34 72),
            row!(7 45 100 99 31 107 10 198 172),
            row!(9 28 49 45 222 133 155 168 124),
            row!(11 158 184 148 209 139 29 12 56),
            row!(16 0 0 0 0 0 0 0 0),
        ],
        // i = 7
        vec![
            row!(1 129 80 0 103 97 48 163 86),
            row!(5 147 186 45 13 135 125 78 186),
            row!(7 140 16 148 105 35 24 143 87),
            row!(11 3 102 96 150 108 47 107 172),
            row!(13 116 143 78 181 65 55 58 154),
            row!(17 0 0 0 0 0 0 0 0),
        ],
        // i = 8
        vec![
            row!(0 142 118 0 147 70 53 101 176),
            row!(1 94 70 65 43 69 31 177 169),
            row!(12 230 152 87 152 88 161 22 225),
            row!(18 0 0 0 0 0 0 0 0),
        ],
        // i = 9
        vec![
            row!(1 203 28 0 2 97 104 186 167),
            row!(8 205 132 97 30 40 142 27 238),
            row!(10 61 185 51 184 24 99 205 48),
            row!(11 247 178 85 83 49 64 81 68),
            row!(19 0 0 0 0 0 0 0 0),
        ],
        // i = 10
        vec![
            row!(0 11 59 0 174 46 111 125 38),
            row!(1 185 104 17 150 41 25 60 217),
            row!(6 0 22 156 8 101 174 177 208),
            row!(7 117 52 20 56 96 23 51 232),
            row!(20 0 0 0 0 0 0 0 0),
        ],
        // i = 11
        vec![
            row!(0 11 32 0 99 28 91 39 178),
            row!(7 236 92 7 138 30 175 29 214),
            row!(9 210 174 4 110 116 24 35 168),
            row!(13 56 154 2 99 64 141 8 51),
            row!(21 0 0 0 0 0 0 0 0),
        ],
        // i = 12
        vec![
            row!(1 63 39 0 46 33 122 18 124),
            row!(3 111 93 113 217 122 11 155 122),
            row!(11 14 11 48 109 131 4 49 72),
            row!(22 0 0 0 0 0 0 0 0),
        ],
        // i = 13
        vec![
            row!(0 83 49 0 37 76 29 32 48),
            row!(1 2 125 112 113 37 91 53 57),
            row!(8 38 35 102 143 62 27 95 167),
            row!(13 222 166 26 140 47 127 186 219),
            row!(23 0 0 0 0 0 0 0 0),
        ],
        // i = 14
        vec![
            row!(1 115 19 0 36 143 11 91 82),
            row!(6 145 118 138 95 51 145 20 232),
            row!(11 3 21 57 40 130 8 52 204),
            row!(13 232 163 27 116 97 166 109 162),
            row!(24 0 0 0 0 0 0 0 0),
        ],
        // i = 15
        vec![
            row!(0 51 68 0 116 139 137 174 38),
            row!(10 175 63 73 200 96 103 108 217),
            row!(11 213 81 99 110 128 40 102 157),
            row!(25 0 0 0 0 0 0 0 0),
        ],
        // i = 16
        vec![
            row!(1 203 87 0 75 48 78 125 170),
            row!(9 142 177 79 158 9 158 31 23),
            row!(11 8 135 111 134 28 17 54 175),
            row!(12 242 64 143 97 8 165 176 202),
            row!(26 0 0 0 0 0 0 0 0),
        ],
        // i = 17
        vec![
            row!(1 254 158 0 48 120 134 57 196),
            row!(5 124 23 24 132 43 23 201 173),
            row!(11 114 9 109 206 65 62 142 195),
            row!(12 64 6 18 2 42 163 35 218),
            row!(27 0 0 0 0 0 0 0 0),
        ],
        // i = 18
        vec![
            row!(0 220 186 0 68 17 173 129 128),
            row!(6 194 6 18 16 106 31 203 211),
            row!(7 50 46 86 156 142 22 140 210),
            row!(28 0 0 0 0 0 0 0 0),
        ],
        // i = 19
        vec![
            row!(0 87 58 0 35 79 13 110 39),
            row!(1 20 42 158 138 28 135 124 84),
            row!(10 185 156 154 86 41 145 52 88),
            row!(29 0 0 0 0 0 0 0 0),
        ],
        // i = 20
        vec![
            row!(1 26 76 0 6 2 128 196 117),
            row!(4 105 61 148 20 103 52 35 227),
            row!(11 29 153 104 141 78 173 114 6),
            row!(30 0 0 0 0 0 0 0 0),
        ],
        // i = 21
        vec![
            row!(0 76 157 0 80 91 156 10 238),
            row!(8 42 175 17 43 75 166 122 13),
            row!(13 210 67 33 81 81 40 23 11),
            row!(31 0 0 0 0 0 0 0 0),
        ],
        // i = 22
        vec![
            row!(1 222 20 0 49 54 18 202 195),
            row!(2 63 52 4 1 132 163 126 44),
            row!(32 0 0 0 0 0 0 0 0),
        ],
        // i = 23
        vec![
            row!(0 23 106 0 156 68 110 52 5),
            row!(3 235 86 75 54 115 132 170 94),
            row!(5 238 95 158 134 56 150 13 111),
            row!(33 0 0 0 0 0 0 0 0),
        ],
        // i = 24
        vec![
            row!(1 46 182 0 153 30 113 113 81),
            row!(2 139 153 69 88 42 108 161 19),
            row!(9 8 64 87 63 101 61 88 130),
            row!(34 0 0 0 0 0 0 0 0),
        ],
        // i = 25
        vec![
            row!(0 228 45 0 211 128 72 197 66),
            row!(5 156 21 65 94 63 136 194 95),
            row!(35 0 0 0 0 0 0 0 0),
        ],
        // i = 26
        vec![
            row!(2 29 67 0 90 142 36 164 146),
            row!(7 143 137 100 6 28 38 172 66),
            row!(12 160 55 13 221 100 53 49 190),
            row!(13 122 85 7 6 133 145 161 86),
            row!(36 0 0 0 0 0 0 0 0),
        ],
        // i = 27
        vec![
            row!(0 8 103 0 27 13 42 168 64),
            row!(6 151 50 32 118 10 104 193 181),
            row!(37 0 0 0 0 0 0 0 0),
        ],
        // i = 28
        vec![
            row!(1 98 70 0 216 106 64 14 7),
            row!(2 101 111 126 212 77 24 186 144),
            row!(5 135 168 110 193 43 149 46 16),
            row!(38 0 0 0 0 0 0 0 0),
        ],
        // i = 29
        vec![
            row!(0 18 110 0 108 133 139 50 25),
            row!(4 28 17 154 61 25 161 27 57),
            row!(39 0 0 0 0 0 0 0 0),
        ],
        // i = 30
        vec![
            row!(2 71 120 0 106 87 84 70 37),
            row!(5 240 154 35 44 56 173 17 139),
            row!(7 9 52 51 185 104 93 50 221),
            row!(9 84 56 134 176 70 29 6 17),
            row!(40 0 0 0 0 0 0 0 0),
        ],
        // i = 31
        vec![
            row!(1 106 3 0 147 80 117 115 201),
            row!(13 1 170 20 182 139 148 189 46),
            row!(41 0 0 0 0 0 0 0 0),
        ],
        // i = 32
        vec![
            row!(0 242 84 0 108 32 116 110 179),
            row!(5 44 8 20 21 89 73 0 14),
            row!(12 166 17 122 110 71 142 163 116),
            row!(42 0 0 0 0 0 0 0 0),
        ],
        // i = 33
        vec![
            row!(2 132 165 0 71 135 105 163 46),
            row!(7 164 179 88 12 6 137 173 2),
            row!(10 235 124 13 109 2 29 179 106),
            row!(43 0 0 0 0 0 0 0 0),
        ],
        // i = 34
        vec![
            row!(0 147 173 0 29 37 11 197 184),
            row!(12 85 177 19 201 25 41 191 135),
            row!(13 36 12 78 69 114 162 193 141),
            row!(44 0 0 0 0 0 0 0 0),
        ],
        // i = 35
        vec![
            row!(1 57 77 0 91 60 126 157 85),
            row!(5 40 184 157 165 137 152 167 225),
            row!(11 63 18 6 55 93 172 181 175),
            row!(45 0 0 0 0 0 0 0 0),
        ],
        // i = 36
        vec![
            row!(0 140 25 0 1 121 73 197 178),
            row!(2 38 151 63 175 129 154 167 112),
            row!(7 154 170 82 83 26 129 179 106),
            row!(46 0 0 0 0 0 0 0 0),
        ],
        // i = 37
        vec![
            row!(10 219 37 0 40 97 167 181 154),
            row!(13 151 31 144 12 56 38 193 114),
            row!(47 0 0 0 0 0 0 0 0),
        ],
        // i = 38
        vec![
            row!(1 31 84 0 37 1 112 157 42),
            row!(5 66 151 93 97 70 7 173 41),
            row!(11 38 190 19 46 1 19 191 105),
            row!(48 0 0 0 0 0 0 0 0),
        ],
        // i = 39
        vec![
            row!(0 239 93 0 106 119 109 181 167),
            row!(7 172 132 24 181 32 6 157 45),
            row!(12 34 57 138 154 142 105 173 189),
            row!(49 0 0 0 0 0 0 0 0),
        ],
        // i = 40
        vec![
            row!(2 0 103 0 98 6 160 193 78),
            row!(10 75 107 36 35 73 156 163 67),
            row!(13 120 163 143 36 102 82 179 180),
            row!(50 0 0 0 0 0 0 0 0),
        ],
        // i = 41
        vec![
            row!(1 129 147 0 120 48 132 191 53),
            row!(5 229 7 2 101 47 6 197 215),
            row!(11 118 60 55 81 19 8 167 230),
            row!(51 0 0 0 0 0 0 0 0),
        ],
    ]
}
