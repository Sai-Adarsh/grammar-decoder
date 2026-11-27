use grammar_decoder::LogitSource;
use std::error::Error;
use std::ffi::{c_void, CStr, CString};
use std::io::Write;
use std::path::Path;
use std::ptr::{self, NonNull};
use std::time::{Duration, Instant};

#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code
)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/llama.rs"));
}

type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub struct Backend;

impl Backend {
    pub fn new() -> Self {
        // The benchmark owns the native backend for the lifetime of all model objects.
        unsafe {
            ffi::llama_log_set(Some(log_errors), ptr::null_mut());
            if ffi::ggml_backend_dev_by_type(ffi::GGML_BACKEND_DEVICE_TYPE_CPU).is_null() {
                // Portable releases ship CPU plugins alongside libllama, away from our executable.
                let library_dir = CString::new(env!("GRAMMAR_LLAMA_LIB_DIR"))
                    .expect("the native library directory cannot contain NUL");
                ffi::ggml_backend_load_all_from_path(library_dir.as_ptr());
            }
            ffi::llama_backend_init();
        }
        Self
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        unsafe { ffi::llama_backend_free() }
    }
}

unsafe extern "C" fn log_errors(
    level: ffi::ggml_log_level,
    message: *const std::ffi::c_char,
    _: *mut c_void,
) {
    if level == ffi::GGML_LOG_LEVEL_ERROR && !message.is_null() {
        let _ = std::io::stderr().write_all(unsafe { CStr::from_ptr(message) }.to_bytes());
    }
}

pub struct Model<'backend> {
    handle: NonNull<ffi::llama_model>,
    _backend: &'backend Backend,
}

pub struct Tokens {
    pub pieces: Vec<Vec<u8>>,
    pub eos: usize,
}

impl<'backend> Model<'backend> {
    pub fn load(path: &Path, backend: &'backend Backend) -> Result<Self> {
        let filename = CString::new(path.as_os_str().as_encoded_bytes())?;
        let mut devices = [ptr::null_mut()];
        let handle = unsafe {
            let mut params = ffi::llama_model_default_params();
            params.n_gpu_layers = 0;
            params.devices = devices.as_mut_ptr();
            ffi::llama_model_load_from_file(filename.as_ptr(), params)
        };
        let handle = NonNull::new(handle).ok_or("llama.cpp could not load the GGUF model")?;
        let model = Self {
            handle,
            _backend: backend,
        };
        if unsafe { ffi::llama_model_has_encoder(model.handle.as_ptr()) } {
            return Err("the benchmark requires a decoder-only model".into());
        }
        Ok(model)
    }

    fn vocab(&self) -> *const ffi::llama_vocab {
        unsafe { ffi::llama_model_get_vocab(self.handle.as_ptr()) }
    }

    pub fn parameters(&self) -> u64 {
        unsafe { ffi::llama_model_n_params(self.handle.as_ptr()) }
    }

    pub fn context_limit(&self) -> Result<usize> {
        let limit = unsafe { ffi::llama_model_n_ctx_train(self.handle.as_ptr()) };
        if limit < 32 {
            return Err("model context must be at least 32 tokens".into());
        }
        Ok((limit as usize).min(2048))
    }

    pub fn tokens(&self) -> Result<Tokens> {
        let vocab = self.vocab();
        let count = unsafe { ffi::llama_vocab_n_tokens(vocab) };
        let eos = unsafe { ffi::llama_vocab_eos(vocab) };
        if count <= 0 || eos < 0 || eos >= count || !unsafe { ffi::llama_vocab_is_eog(vocab, eos) }
        {
            return Err("model vocabulary has no canonical EOS token".into());
        }
        let mut pieces = Vec::with_capacity(count as usize);
        let mut buffer = vec![0u8; 64];
        for id in 0..count {
            if unsafe {
                ffi::llama_vocab_is_control(vocab, id) || ffi::llama_vocab_is_eog(vocab, id)
            } {
                pieces.push(Vec::new());
                continue;
            }
            let read_piece = |buf: &mut [u8]| unsafe {
                ffi::llama_token_to_piece(
                    vocab,
                    id,
                    buf.as_mut_ptr().cast(),
                    buf.len() as i32,
                    0,
                    false,
                )
            };
            let mut length = read_piece(&mut buffer);
            if length < 0 {
                let needed = length
                    .checked_abs()
                    .ok_or("llama.cpp piece length overflow")?
                    as usize;
                buffer.resize(needed, 0);
                length = read_piece(&mut buffer);
            }
            let length = usize::try_from(length)?;
            pieces.push(
                buffer
                    .get(..length)
                    .ok_or("invalid token piece length")?
                    .to_vec(),
            );
        }
        Ok(Tokens {
            pieces,
            eos: eos as usize,
        })
    }

    pub fn tokenize(&self, text: &str, limit: usize) -> Result<Vec<ffi::llama_token>> {
        let mut tokens = vec![0; limit];
        let length = unsafe {
            ffi::llama_tokenize(
                self.vocab(),
                text.as_ptr().cast(),
                i32::try_from(text.len())?,
                tokens.as_mut_ptr(),
                i32::try_from(tokens.len())?,
                true,
                false,
            )
        };
        if length <= 0 || length as usize > tokens.len() {
            return Err("prompt must fit the model context and contain at least one token".into());
        }
        tokens.truncate(length as usize);
        Ok(tokens)
    }
}

impl Drop for Model<'_> {
    fn drop(&mut self) {
        unsafe { ffi::llama_model_free(self.handle.as_ptr()) }
    }
}

struct Context(NonNull<ffi::llama_context>);

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { ffi::llama_free(self.0.as_ptr()) }
    }
}

struct Batch(NonNull<ffi::llama_batch_ext>);

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe { ffi::llama_batch_ext_free(self.0.as_ptr()) }
    }
}

pub struct Session<'model, 'backend> {
    batch: Batch,
    context: Context,
    _model: &'model Model<'backend>,
    position: usize,
    width: usize,
    started: Instant,
    pub prefill_time: Duration,
    pub model_decode_time: Duration,
    pub logits_copy_time: Duration,
    pub first_selected: Option<Duration>,
    pub logit_rows: usize,
    pub nonfinite_logits: usize,
    pub generated: Vec<usize>,
    pub error: Option<String>,
}

impl<'model, 'backend> Session<'model, 'backend> {
    pub fn start(
        model: &'model Model<'backend>,
        prompt: &[ffi::llama_token],
        limit: usize,
        threads: i32,
        started: Instant,
    ) -> Result<Self> {
        let width = usize::try_from(unsafe { ffi::llama_vocab_n_tokens(model.vocab()) })?;
        let context = unsafe {
            let mut params = ffi::llama_context_default_params();
            params.n_ctx = u32::try_from(limit)?;
            params.n_batch = params.n_ctx;
            params.n_ubatch = params.n_ctx;
            params.n_threads = threads;
            params.n_threads_batch = threads;
            params.n_outputs_max = 1;
            params.offload_kqv = false;
            params.op_offload = false;
            ffi::llama_init_from_model(model.handle.as_ptr(), params)
        };
        let context = Context(NonNull::new(context).ok_or("llama.cpp could not create a context")?);
        let batch = unsafe { ffi::llama_batch_ext_init(context.0.as_ptr()) };
        let batch =
            Batch(NonNull::new(batch).ok_or("llama.cpp could not allocate an input batch")?);
        let mut session = Self {
            batch,
            context,
            _model: model,
            position: 0,
            width,
            started,
            prefill_time: Duration::ZERO,
            model_decode_time: Duration::ZERO,
            logits_copy_time: Duration::ZERO,
            first_selected: None,
            logit_rows: 0,
            nonfinite_logits: 0,
            generated: Vec::new(),
            error: None,
        };
        let start = Instant::now();
        session.forward(prompt)?;
        session.prefill_time = start.elapsed();
        Ok(session)
    }

    pub fn allocated_context(&self) -> u32 {
        unsafe { ffi::llama_n_ctx(self.context.0.as_ptr()) }
    }

    fn forward(&mut self, tokens: &[ffi::llama_token]) -> Result<()> {
        let batch = self.batch.0.as_ptr();
        let context = self.context.0.as_ptr();
        unsafe { ffi::llama_batch_ext_clear(batch) };
        for (offset, &token) in tokens.iter().enumerate() {
            let position = i32::try_from(self.position + offset)?;
            unsafe {
                let index = ffi::llama_batch_ext_add_token(batch, 0, token);
                if index < 0
                    || !ffi::llama_batch_ext_set_pos(batch, index, &position)
                    || !ffi::llama_batch_ext_set_output_logits(
                        batch,
                        index,
                        offset + 1 == tokens.len(),
                    )
                {
                    return Err("llama.cpp rejected an input batch entry".into());
                }
            }
        }
        let result = unsafe { ffi::llama_process(context, ffi::LLAMA_PROCESS_TYPE_DECODE, batch) };
        if result != 0 {
            return Err(format!("llama_process returned {result}").into());
        }
        unsafe { ffi::llama_synchronize(context) };
        self.position += tokens.len();
        Ok(())
    }
}

impl LogitSource for Session<'_, '_> {
    fn logits(&mut self, output: &mut [f64]) {
        let start = Instant::now();
        if self.error.is_some() {
            output.fill(f64::NEG_INFINITY);
            return;
        }
        assert_eq!(output.len(), self.width);
        let row = unsafe { ffi::llama_get_logits_ith(self.context.0.as_ptr(), -1) };
        if row.is_null() {
            self.error = Some("llama.cpp did not return the requested logit row".into());
            output.fill(f64::NEG_INFINITY);
            return;
        }
        // The native API exposes n_vocab floats at the last requested output index.
        // Copy the row before the next forward can invalidate its backing storage.
        let raw = unsafe { std::slice::from_raw_parts(row, self.width) };
        for (slot, &score) in output.iter_mut().zip(raw) {
            *slot = f64::from(score);
            self.nonfinite_logits += usize::from(!score.is_finite());
        }
        self.logit_rows += 1;
        self.logits_copy_time += start.elapsed();
    }

    fn accept(&mut self, token_id: usize) {
        if self.first_selected.is_none() {
            self.first_selected = Some(self.started.elapsed());
        }
        self.generated.push(token_id);
        let start = Instant::now();
        if let Err(error) = self.forward(&[token_id as ffi::llama_token]) {
            self.error = Some(error.to_string());
        }
        self.model_decode_time += start.elapsed();
    }
}
