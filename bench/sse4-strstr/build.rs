use std::{env, path::PathBuf};

fn main() {
    if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

        println!("cargo:rerun-if-changed=src/wrapper.h");
        bindgen::Builder::default()
            .header("src/wrapper.h")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks))
            .generate()
            .unwrap()
            .write_to_file(out_path.join("sse4_strstr.rs"))
            .unwrap();

        println!("cargo:rerun-if-changed=src/wrapper.cpp");
        cc::Build::new()
            .cpp(true)
            .file("src/wrapper.cpp")
            .include("src/sse4-strstr")
            .flag("-std=c++11")
            .flag("-march=native")
            .compile("sse4-strstr");
    }
}
