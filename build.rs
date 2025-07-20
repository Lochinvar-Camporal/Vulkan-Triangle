use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=src/shaders");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let mut compiler = shaderc::Compiler::new().ok_or("shaderc not found!")?;

    for entry in std::fs::read_dir("src/shaders")? {
        let entry = entry?;
        let in_path = entry.path();

        if in_path.is_file() {
            if let Some(ext) = in_path.extension().and_then(|s| s.to_str()) {
                if ext == "vert" || ext == "frag" {
                    let source = std::fs::read_to_string(&in_path)?;
                    let kind = if ext == "vert" {
                        shaderc::ShaderKind::Vertex
                    } else {
                        shaderc::ShaderKind::Fragment
                    };

                    let compiled = compiler.compile_into_spirv(
                        &source,
                        kind,
                        in_path.to_str().unwrap(),
                        "main",
                        None,
                    )?;

                    let file_name = in_path.file_name().unwrap().to_str().unwrap();
                    let out_file_name = format!("{}.spv", file_name);
                    let out_path = out_dir.join(&out_file_name);

                    std::fs::write(&out_path, compiled.as_binary_u8())?;

                    let env_var_name = format!("{}_SHADER_PATH", ext.to_uppercase());
                    println!("cargo:rustc-env={}={}", env_var_name, out_path.to_str().unwrap());
                }
            }
        }
    }

    Ok(())
}
