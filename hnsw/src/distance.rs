pub trait Distance {
    fn distance(&self, other: &Self) -> u32;
}

macro_rules! impl_int_distance {
    ($($types:ty),+) => {
        $(impl Distance for $types {
            fn distance(&self, other: &Self) -> u32 {
                self.abs_diff(*other) as u32
            }
        })+
    }
}

impl_int_distance! {
    u8, u16, u32, u64, u128, usize,
    i8, i16, i32, i64, i128, isize
}