use grammar_decoder::{decode, json_schema, DecodeError, DecodeOptions, Generation, LogitSource};
use std::error::Error;
use std::fs;
use std::io::{self, Read};
use std::path::Path;
use std::process::ExitCode;

struct FixtureLogits(usize);

impl LogitSource for FixtureLogits {
    fn logits(&mut self, output: &mut [f64]) {
        // The top scorer is illegal in every position, including after the final brace.
        output[0] = 15.0;
        output[(self.0 + 1).min(8)] = 8.0;
        for score in &mut output[1..] {
            if *score == f64::NEG_INFINITY {
                *score = -30.0;
            }
        }
    }

    fn accept(&mut self, _: usize) {
        self.0 += 1;
    }
}

fn fixture() -> Result<Generation, Box<dyn Error>> {
    let schema = serde_json::from_str(include_str!("../examples/tool-call.schema.json"))?;
    let vocabulary = vec![
        vec![0xff],
        b"{\"action\":\"".to_vec(),
        b"search\",\"query\":\"".to_vec(),
        b"caf\xc3".to_vec(),
        b"\xa9 and \\".to_vec(),
        b"\"quotes\\\"".to_vec(),
        b"\",\"limit\":".to_vec(),
        b"3}".to_vec(),
        vec![],
    ];
    Ok(decode(
        &json_schema::compile(&schema)?,
        &vocabulary,
        &mut FixtureLogits(0),
        DecodeOptions { eos: 8, max_tokens: 16, temperature: 0.7, seed: 42 },
    )?)
}

const BYTE_EOS: usize = 256;
const MAX_SCHEMA_INPUT: usize = 64 * 1024;
const MAX_CANDIDATE_INPUT: usize = 16 * 1024;

struct CandidateLogits<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl LogitSource for CandidateLogits<'_> {
    fn logits(&mut self, output: &mut [f64]) {
        output[0xff] = 15.0;
        let preferred = self.bytes.get(self.offset).map_or(BYTE_EOS, |byte| usize::from(*byte));
        output[preferred] = 8.0;
    }

    fn accept(&mut self, _: usize) {
        self.offset += 1;
    }
}

fn schema_demo(path: &Path) -> Result<Generation, Box<dyn Error>> {
    let schema_bytes = read_bounded(fs::File::open(path)?, MAX_SCHEMA_INPUT, "schema")?;
    let schema = serde_json::from_slice(&schema_bytes)?;
    let grammar = json_schema::compile(&schema)?;
    let input = read_bounded(io::stdin().lock(), MAX_CANDIDATE_INPUT, "candidate")?;
    let bytes = input.trim_ascii();
    let mut vocabulary: Vec<Vec<u8>> = (0..=u8::MAX).map(|byte| vec![byte]).collect();
    vocabulary.push(vec![]);
    let mut source = CandidateLogits { bytes, offset: 0 };
    decode(
        &grammar,
        &vocabulary,
        &mut source,
        DecodeOptions { eos: BYTE_EOS, max_tokens: bytes.len(), temperature: 0.0, seed: 0 },
    )
    .map_err(|err| match err {
        DecodeError::NoViableToken => format!("candidate rejected at byte {}", source.offset).into(),
        _ => err.into(),
    })
}

fn read_bounded(reader: impl Read, max_bytes: usize, label: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut bytes = Vec::new();
    reader.take((max_bytes + 1) as u64).read_to_end(&mut bytes)?;
    if bytes.len() > max_bytes {
        return Err(format!("{label} input exceeds {max_bytes} bytes").into());
    }
    Ok(bytes)
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let output = match args.as_slice() {
        [] => fixture()?,
        [flag, path] if flag == "--schema" => schema_demo(Path::new(path))?,
        _ => return Err("usage: grammar-decoder [--schema SCHEMA.json < compact.json]".into()),
    };
    println!("{}", std::str::from_utf8(&output.bytes)?);
    eprintln!(
        "{} emitted tokens, {} candidate rejections",
        output.token_ids.len(),
        output.rejected_candidates
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}
