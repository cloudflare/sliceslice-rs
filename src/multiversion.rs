macro_rules! multiversion {
    ($vis:vis unsafe fn $name:ident $(<$($gen_name:ident : $gen_ty:ty),+>)? ( $($arg_name:ident : $arg_ty:ty ,)+ ) -> $ret:ty $block:block) => {
        paste::paste! {
            #[allow(dead_code)]
            #[inline(always)]
            $vis unsafe fn [<$name _ default_version>] $(<$($gen_name : $gen_ty),+>)? ( $($arg_name : $arg_ty),+ ) -> $ret {
                #[allow(dead_code)]
                const TARGET: crate::multiversion::Target = crate::multiversion::Target::Default;
                $block
            }

            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2")]
            #[inline]
            $vis unsafe fn [<$name _ avx2_version>] $(<$($gen_name : $gen_ty),+>)? ( $($arg_name : $arg_ty),+ ) -> $ret {
                #[allow(dead_code)]
                const TARGET: crate::multiversion::Target = crate::multiversion::Target::Avx2;
                $block
            }

            #[cfg(target_arch = "wasm32")]
            #[target_feature(enable = "simd128")]
            #[inline]
            $vis unsafe fn [<$name _ simd128_version>] $(<$($gen_name : $gen_ty),+>)? ( $($arg_name : $arg_ty),+ ) -> $ret {
                #[allow(dead_code)]
                const TARGET: crate::multiversion::Target = crate::multiversion::Target::Simd128;
                $block
            }

            #[cfg(target_arch = "aarch64")]
            #[target_feature(enable = "neon")]
            #[inline]
            $vis unsafe fn [<$name _ neon_version>] $(<$($gen_name : $gen_ty),+>)? ( $($arg_name : $arg_ty),+ ) -> $ret {
                #[allow(dead_code)]
                const TARGET: crate::multiversion::Target = crate::multiversion::Target::Neon;
                $block
            }
        }
    };
}

cfg_if::cfg_if! {
    if #[cfg(target_arch = "x86_64")] {
        pub(crate) enum Target {
            Default,
            Avx2,
        }

        macro_rules! dispatch {
            ($target:ident => $name:ident ( $($arg:expr),+ )) => {
                match $target {
                    crate::multiversion::Target::Default => paste::paste! { [<$name _ default_version>] ( $($arg),+ ) },
                    crate::multiversion::Target::Avx2 => paste::paste! { [<$name _ avx2_version>] ( $($arg),+ ) },
                }
            }
        }
    } else if #[cfg(target_arch = "wasm32")] {
        pub(crate) enum Target {
            Default,
            Simd128,
        }

        macro_rules! dispatch {
            ($target:ident => $name:ident ( $($arg:expr),+ )) => {
                match $target {
                    crate::multiversion::Target::Default => paste::paste! { [<$name _ default_version>] ( $($arg),+ ) },
                    crate::multiversion::Target::Simd128 => paste::paste! { [<$name _ simd128_version>] ( $($arg),+ ) },
                }
            }
        }
    } else if #[cfg(target_arch = "aarch64")] {
        pub(crate) enum Target {
            Default,
            Neon,
        }

        macro_rules! dispatch {
            ($target:ident => $name:ident ( $($arg:expr),+ )) => {
                match $target {
                    crate::multiversion::Target::Default => paste::paste! { [<$name _ default_version>] ( $($arg),+ ) },
                    crate::multiversion::Target::Neon => paste::paste! { [<$name _ neon_version>] ( $($arg),+ ) },
                }
            }
        }
    } else {
        pub(crate) enum Target {
            Default,
        }

        macro_rules! dispatch {
            ($target:ident => $name:ident ( $($arg:expr),+ )) => {
                paste::paste! { [<$name _ default_version>] ( $($arg),+ ) },
            }
        }
    }
}
