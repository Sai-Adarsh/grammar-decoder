use crate::native_llama::{Backend, Model, Session, Tokens};
use grammar_decoder::{
    decode_with_vocabulary, json_schema, DecodeError, DecodeOptions, Grammar, LogitSource,
    Vocabulary,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::error::Error;
use std::ffi::OsStr;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{self, Write as _};
use std::path::Path;
use std::time::Instant;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Prompt {
    id: String,
    prompt: String,
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path, label: &str) -> Result<T> {
    let bytes = super::read_bounded(File::open(path)?, 64 * 1024, label)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn bounded_number(input: &OsStr, limit: usize, name: &str) -> Result<usize> {
    input
        .to_str()
        .and_then(|text| text.parse::<usize>().ok())
        .filter(|value| (1..=limit).contains(value))
        .ok_or_else(|| format!("{name} must be an integer from 1 to {limit}").into())
}

#[derive(Clone, Copy)]
enum Mode {
    Raw,
    Constrained,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Raw => "unconstrained",
            Self::Constrained => "constrained",
        }
    }
}

fn unconstrained(
    source: &mut impl LogitSource,
    tokens: &Tokens,
    max_tokens: usize,
) -> &'static str {
    let mut logits = vec![f64::NEG_INFINITY; tokens.pieces.len()];
    for step in 0..=max_tokens {
        source.logits(&mut logits);
        let best = logits
            .iter()
            .enumerate()
            .filter(|&(id, score)| {
                score.is_finite() && (id == tokens.eos || !tokens.pieces[id].is_empty())
            })
            .max_by(|a, b| a.1.total_cmp(b.1));
        match best {
            None => return "no_viable_token",
            Some((id, _)) if id == tokens.eos => return "eos",
            Some(_) if step == max_tokens => return "token_budget",
            Some((id, _)) => source.accept(id),
        }
    }
    unreachable!()
}

fn validate(bytes: &[u8], validator: &jsonschema::Validator) -> (bool, bool, Option<String>) {
    match serde_json::from_slice::<Value>(bytes) {
        Ok(value) => match validator.validate(&value) {
            Ok(()) => (true, true, None),
            Err(error) => (true, false, Some(error.to_string())),
        },
        Err(error) => (false, false, Some(error.to_string())),
    }
}

struct Benchmark<'a, 'backend> {
    model: &'a Model<'backend>,
    tokens: &'a Tokens,
    grammar: &'a Grammar,
    vocabulary: &'a Vocabulary<'a>,
    validator: &'a jsonschema::Validator,
    limit: usize,
    max_tokens: usize,
    threads: i32,
}

impl Benchmark<'_, '_> {
    fn measure(
        &self,
        prompt: &Prompt,
        mode: Mode,
        pair: usize,
        first: Mode,
    ) -> Result<(Value, bool)> {
        let start = Instant::now();
        let prompt_tokens = self
            .model
            .tokenize(&prompt.prompt, self.limit - self.max_tokens)?;
        let mut source =
            Session::start(self.model, &prompt_tokens, self.limit, self.threads, start)?;
        let decode_start = Instant::now();
        let (stop, attempts, rejected) = match mode {
            Mode::Raw => (
                unconstrained(&mut source, self.tokens, self.max_tokens),
                None,
                None,
            ),
            Mode::Constrained => {
                let options = DecodeOptions {
                    eos: self.tokens.eos,
                    max_tokens: self.max_tokens,
                    temperature: 0.0,
                    seed: 0,
                };
                match decode_with_vocabulary(self.grammar, self.vocabulary, &mut source, options) {
                    Ok(result) => (
                        "eos",
                        Some(result.attempted_mask_bytes),
                        Some(result.rejected_candidates),
                    ),
                    Err(DecodeError::MaxTokens) => ("token_budget", None, None),
                    Err(DecodeError::NoViableToken) => ("no_viable_token", None, None),
                    Err(DecodeError::InvalidConfig) => {
                        return Err("invalid native decoding configuration".into())
                    }
                }
            }
        };
        let generation_time = decode_start.elapsed();
        if let Some(error) = &source.error {
            return Err(error.clone().into());
        }
        let output: Vec<u8> = source
            .generated
            .iter()
            .flat_map(|&id| self.tokens.pieces[id].iter().copied())
            .collect();
        let (json_valid, schema_valid, validation_error) = validate(&output, self.validator);
        let request_time = start.elapsed();
        let text = std::str::from_utf8(&output).ok();
        let output_hex = text.is_none().then(|| {
            let mut hex = String::with_capacity(output.len() * 2);
            for byte in &output {
                write!(hex, "{byte:02x}").expect("write to String");
            }
            hex
        });
        let row = json!({
            "kind":"sample", "prompt_id":prompt.id, "pair":pair, "first":first.name(), "mode":mode.name(),
            "prompt_tokens":prompt_tokens.len(), "generated_tokens":source.generated.len(),
            "output_token_ids":source.generated, "output":text, "output_hex":output_hex,
            "status":stop, "json_valid":json_valid, "schema_valid":schema_valid, "validation_error":validation_error,
            "columns_per_logit_row":self.tokens.pieces.len(), "logit_rows":source.logit_rows,
            "nonfinite_logits":source.nonfinite_logits, "allocated_context":source.allocated_context(),
            "prefill_ns":source.prefill_time.as_nanos(), "native_decode_ns":source.model_decode_time.as_nanos(),
            "logits_copy_ns":source.logits_copy_time.as_nanos(), "generation_ns":generation_time.as_nanos(),
            "first_selected_ns":source.first_selected.map(|span| span.as_nanos()),
            "request_ns":request_time.as_nanos(), "mask_attempts":attempts, "rejected_candidates":rejected,
        });
        Ok((row, stop == "eos" && schema_valid))
    }
}

pub fn run(
    model_path: &Path,
    schema_path: &Path,
    prompts_path: &Path,
    tokens: &OsStr,
    pairs: &OsStr,
    threads: &OsStr,
) -> Result<()> {
    let max_tokens = bounded_number(tokens, 256, "MAX_TOKENS")?;
    let pairs = bounded_number(pairs, 50, "PAIRS")?;
    let threads = bounded_number(threads, 64, "THREADS")? as i32;
    let schema: Value = read_json(schema_path, "schema")?;
    let grammar = json_schema::compile(&schema)?;
    let validator = jsonschema::draft202012::new(&schema)?;
    let prompts: Vec<Prompt> = read_json(prompts_path, "prompts")?;
    let mut ids = HashSet::new();
    if prompts.is_empty()
        || prompts.len() > 32
        || prompts.iter().any(|prompt| {
            prompt.id.trim().is_empty()
                || prompt.prompt.trim().is_empty()
                || !ids.insert(&prompt.id)
        })
    {
        return Err("provide 1 to 32 prompts with unique nonempty IDs and nonempty text".into());
    }

    let backend = Backend::new();
    let start = Instant::now();
    let model = Model::load(model_path, &backend)?;
    let model_load_time = start.elapsed();
    let limit = model.context_limit()?;
    if max_tokens >= limit {
        return Err("MAX_TOKENS leaves no space for the prompt".into());
    }
    let start = Instant::now();
    let tokens = model.tokens()?;
    let token_pieces_time = start.elapsed();
    let start = Instant::now();
    let vocabulary = Vocabulary::for_grammar(&tokens.pieces, tokens.eos, &grammar)?;
    let projected_trie_time = start.elapsed();
    let benchmark = Benchmark {
        model: &model,
        tokens: &tokens,
        grammar: &grammar,
        vocabulary: &vocabulary,
        validator: &validator,
        limit,
        max_tokens,
        threads,
    };
    for mode in [Mode::Raw, Mode::Constrained] {
        benchmark.measure(&prompts[0], mode, 0, Mode::Raw)?;
    }
    let mut output = io::BufWriter::new(io::stdout().lock());
    writeln!(
        output,
        "{}",
        json!({
            "kind":"config", "model":model_path.file_name().unwrap_or(model_path.as_os_str()).to_string_lossy(),
            "model_parameters":model.parameters(), "vocabulary_size":tokens.pieces.len(), "eos_token":tokens.eos,
            "projected_trie_nodes":vocabulary.trie_nodes(), "threads":threads, "context_limit":limit, "max_tokens":max_tokens,
            "prompt_count":prompts.len(), "pairs_per_prompt":pairs, "temperature":0, "backend":"llama.cpp CPU",
            "model_load_ns":model_load_time.as_nanos(), "token_pieces_ns":token_pieces_time.as_nanos(),
            "projected_trie_ns":projected_trie_time.as_nanos(),
        })
    )?;
    let mut constrained_failures = 0;
    for (prompt_index, prompt) in prompts.iter().enumerate() {
        for pair in 1..=pairs {
            let order = if (prompt_index + pair) % 2 == 1 {
                [Mode::Raw, Mode::Constrained]
            } else {
                [Mode::Constrained, Mode::Raw]
            };
            for mode in order {
                let (row, complete) = benchmark.measure(prompt, mode, pair, order[0])?;
                writeln!(output, "{row}")?;
                constrained_failures += usize::from(matches!(mode, Mode::Constrained) && !complete);
            }
        }
    }
    output.flush()?;
    if constrained_failures != 0 {
        return Err(format!(
            "{constrained_failures} constrained requests failed to produce validated JSON and EOS"
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_distinguishes_syntax_from_schema_constraints() {
        let schema = json!({"type":"object","properties":{"count":{"type":"integer","enum":[1,2]}},
                         "required":["count"],"additionalProperties":false});
        let validator = jsonschema::draft202012::new(&schema).unwrap();
        assert_eq!(validate(br#"{"count":1}"#, &validator), (true, true, None));
        for bytes in [br#"{"count":1} extra"#.as_slice(), b"{\"count\":\"\xff\"}"] {
            let (syntax, schema, detail) = validate(bytes, &validator);
            assert!(!syntax && !schema && detail.is_some());
        }
        for bytes in [
            br#"{"count":3}"#.as_slice(),
            br#"{"count":1,"extra":0}"#,
            b"{}",
        ] {
            let (syntax, schema, detail) = validate(bytes, &validator);
            assert!(syntax && !schema && detail.is_some());
        }
    }
}
