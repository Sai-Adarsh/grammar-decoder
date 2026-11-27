use grammar_decoder::{json_schema, MaskMethod, Parser, Vocabulary};
use serde_json::Value;
use std::{
    collections::HashMap, error::Error, fs, path::Path, time::Instant,
};

fn byte_level_alphabet() -> HashMap<char, u8> {
    // GPT-2 stores each byte as a printable codepoint. Its decoder reverses
    // this alphabet, rather than interpreting vocabulary entries as UTF-8.
    let mut alphabet = HashMap::with_capacity(256);
    let mut extra = 256;
    for byte in 0..=255u8 {
        let codepoint = if matches!(byte, b'!'..=b'~' | 161..=172 | 174..=255) {
            u32::from(byte)
        } else {
            let code = extra;
            extra += 1;
            code
        };
        alphabet.insert(char::from_u32(codepoint).expect("GPT-2 alphabet"), byte);
    }
    alphabet
}

fn gpt2_tokens(path: &Path) -> Result<(Vec<Vec<u8>>, usize), Box<dyn Error>> {
    let source = super::read_bounded(fs::File::open(path)?, 16 * 1024 * 1024, "tokenizer")?;
    let tokenizer: Value = serde_json::from_slice(&source)?;
    if tokenizer["decoder"]["type"] != "ByteLevel" {
        return Err("expected a GPT-2 ByteLevel tokenizer.json".into());
    }
    let entries = tokenizer["model"]["vocab"].as_object().ok_or("missing model.vocab")?;
    if entries.len() != 50_257 {
        return Err("expected 50,257 base vocabulary entries".into());
    }
    let eos = entries.get("<|endoftext|>").and_then(Value::as_u64)
        .ok_or("missing GPT-2 EOS")? as usize;
    let alphabet = byte_level_alphabet();
    let mut tokens = vec![Vec::new(); entries.len()];
    for (encoded, index) in entries {
        let id = usize::try_from(index.as_u64().ok_or("noninteger token ID")?)?;
        let token = tokens.get_mut(id).ok_or("token ID is out of range")?;
        if !token.is_empty() || encoded.is_empty() {
            return Err("duplicate token ID or empty entry".into());
        }
        for codepoint in encoded.chars() {
            token.push(*alphabet.get(&codepoint).ok_or("unsupported ByteLevel codepoint")?);
        }
    }
    if !tokenizer["added_tokens"].as_array().is_some_and(|tokens| {
        tokens.iter().any(|token| {
            token["id"].as_u64() == Some(eos as u64)
                && token["content"] == "<|endoftext|>"
                && token["special"] == true
        })
    }) {
        return Err("missing GPT-2 special EOS declaration".into());
    }
    Ok((tokens, eos))
}

pub fn run(path: &Path) -> Result<(), Box<dyn Error>> {
    let (tokens, eos) = gpt2_tokens(path)?;
    let schema = serde_json::from_str(include_str!("../examples/tool-call.schema.json"))?;
    let grammar = json_schema::compile(&schema)?;
    let start = Instant::now();
    let vocabulary = Vocabulary::new(&tokens, eos)?;
    let raw_build = start.elapsed();
    let start = Instant::now();
    let projected = Vocabulary::for_grammar(&tokens, eos, &grammar)?;
    let projected_build = start.elapsed();
    eprintln!(
        "vocabulary={} trie_nodes={} build_us={} raw_build_ns={} projected_nodes={} projected_build_ns={}",
        tokens.len(), vocabulary.trie_nodes(), raw_build.as_micros(), raw_build.as_nanos(),
        projected.trie_nodes(), projected_build.as_nanos(),
    );

    let paths = [
        (&vocabulary, MaskMethod::Individual),
        (&vocabulary, MaskMethod::Trie),
        (&projected, MaskMethod::Trie),
    ];
    let mut masks: [Vec<bool>; 3] = std::array::from_fn(|_| vec![false; tokens.len()]);
    // Use all six orders twice so each path appears equally often in each
    // trial position.
    const ORDERS: [[usize; 3]; 6] = [
        [0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1], [2, 1, 0], [1, 0, 2],
    ];
    const TRIALS: usize = ORDERS.len() * 2;
    let cases: [(&str, &[u8]); 5] = [
        ("start", b""),
        ("action_enum", br#"{"action":""#),
        ("query_text", br#"{"action":"search","query":"open the door"#),
        ("unicode_escape", br#"{"action":"answer","query":"\uD83D"#),
        ("complete", br#"{"action":"search","query":"hi","limit":3}"#),
    ];

    println!(concat!(
        "case,vocabulary,allowed,individual_bytes,trie_bytes,individual_median_us,trie_median_us,",
        "projected_bytes,trie_median_ns,projected_median_ns"
    ));
    for (label, prefix) in cases {
        let mut parsers = paths.map(|_| Parser::new(&grammar));
        if parsers.iter_mut().any(|parser| !parser.commit(prefix)) {
            return Err(format!("invalid benchmark prefix {label}").into());
        }
        let mut timings = [[0; TRIALS]; 3];
        let mut attempts = [0; 3];
        for (index, &(vocabulary, method)) in paths.iter().enumerate() {
            vocabulary.mask(&mut parsers[index], &mut masks[index], method);
        }
        for (trial, order) in ORDERS.iter().cycle().take(TRIALS).enumerate() {
            for &index in order {
                let (vocabulary, method) = paths[index];
                let start = Instant::now();
                attempts[index] = vocabulary.mask(&mut parsers[index], &mut masks[index], method);
                timings[index][trial] = start.elapsed().as_nanos();
            }
            if masks[0] != masks[1] || masks[0] != masks[2] {
                return Err(format!("mask differs for prefix {label}").into());
            }
        }
        let [individual_ns, raw_ns, projected_ns] = timings.map(|mut samples| {
            samples.sort_unstable();
            let upper = TRIALS / 2;
            let lower = samples[upper - 1];
            lower + (samples[upper] - lower) / 2
        });
        let [individual_bytes, trie_bytes, projected_bytes] = attempts;
        println!(
            "{label},{},{},{individual_bytes},{trie_bytes},{},{},{projected_bytes},{raw_ns},{projected_ns}",
            tokens.len(), masks[0].iter().filter(|allow| **allow).count(), individual_ns / 1000,
            raw_ns / 1000,
        );
    }
    Ok(())
}
