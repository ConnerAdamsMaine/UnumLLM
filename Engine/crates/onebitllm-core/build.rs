use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/backend/hip/packed_dense_left_transposed.hip");
    println!("cargo:rerun-if-env-changed=ONEBITLLM_HIPCC");
    println!("cargo:rerun-if-env-changed=ONEBITLLM_HIP_ARCH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");

    if env::var_os("CARGO_FEATURE_ROCM").is_none() {
        return;
    }

    let Some(hipcc) = find_hipcc() else {
        println!(
            "cargo:warning=ROCm feature enabled but hipcc was not found; skipping HIP kernel build"
        );
        return;
    };

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR should be set"));
    let output = out_dir.join("libonebitllm_hip_kernels.so");
    let source = Path::new("src/backend/hip/packed_dense_left_transposed.hip");

    let mut command = Command::new(&hipcc);
    command
        .arg("-shared")
        .arg("-fPIC")
        .arg("-O3")
        .arg("-std=c++17")
        .arg(source)
        .arg("-o")
        .arg(&output);

    if let Some(arch) = env::var_os("ONEBITLLM_HIP_ARCH") {
        command.arg(format!("--offload-arch={}", arch.to_string_lossy()));
    }

    match command.status() {
        Ok(status) if status.success() => {
            println!(
                "cargo:rustc-env=ONEBITLLM_HIP_KERNEL_LIB={}",
                output.display()
            );
        }
        Ok(status) => {
            println!(
                "cargo:warning=hipcc returned status {status}; skipping HIP kernel library export"
            );
        }
        Err(err) => {
            println!(
                "cargo:warning=failed to invoke hipcc at {}: {err}",
                hipcc.display()
            );
        }
    }
}

fn find_hipcc() -> Option<PathBuf> {
    if let Some(path) = env::var_os("ONEBITLLM_HIPCC") {
        return Some(PathBuf::from(path));
    }

    if let Some(hip_path) = env::var_os("HIP_PATH") {
        let candidate = PathBuf::from(hip_path).join("bin").join("hipcc");
        if candidate.exists() {
            return Some(candidate);
        }
    }

    let path_var = env::var_os("PATH")?;
    for entry in env::split_paths(&path_var) {
        let candidate = entry.join("hipcc");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}
