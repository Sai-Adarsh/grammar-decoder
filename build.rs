fn main() {
    #[cfg(feature = "native-llama")]
    bind_llama();
}

#[cfg(feature = "native-llama")]
fn bind_llama() {
    use std::{env, path::PathBuf};

    for name in ["LLAMA_SOURCE_DIR", "LLAMA_LIB_DIR", "LIBCLANG_PATH"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let source = PathBuf::from(
        env::var("LLAMA_SOURCE_DIR").expect("set LLAMA_SOURCE_DIR to the source matching libllama"),
    );
    let lib = PathBuf::from(
        env::var("LLAMA_LIB_DIR").expect("set LLAMA_LIB_DIR to the directory containing libllama"),
    )
    .canonicalize()
    .expect("open LLAMA_LIB_DIR");
    let header = source.join("include/llama.h");
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-changed={}", source.join("ggml/include").display());
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-env=GRAMMAR_LLAMA_LIB_DIR={}", lib.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());

    bindgen::Builder::default()
        .header(header.to_str().expect("UTF-8 llama.h path"))
        .clang_arg(format!("-I{}", source.join("ggml/include").display()))
        .allowlist_function("llama_(backend_init|backend_free|log_set|model_default_params|model_load_from_file|model_free|model_get_vocab|model_has_encoder|model_n_ctx_train|model_n_params|vocab_n_tokens|vocab_eos|vocab_is_eog|vocab_is_control|token_to_piece|tokenize|context_default_params|init_from_model|free|n_ctx|batch_ext_.*|process|get_logits_ith|synchronize)")
        .allowlist_function("ggml_backend_(dev_by_type|load_all_from_path)")
        .allowlist_var("LLAMA_PROCESS_TYPE_DECODE")
        .prepend_enum_name(false)
        .formatter(bindgen::Formatter::None)
        .generate()
        .expect("generate Rust bindings from the matching llama.h")
        .write_to_file(PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("llama.rs"))
        .expect("write llama.cpp bindings");
}
