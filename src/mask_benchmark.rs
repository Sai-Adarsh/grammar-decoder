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
    let start = Instant::now();
    let vocabulary = Vocabulary::new(&tokens, eos)?;
    eprintln!("vocabulary={} trie_nodes={} build_us={}", tokens.len(), vocabulary.trie_nodes(),
              start.elapsed().as_micros());
    let schema = serde_json::from_str(include_str!("../examples/tool-call.schema.json"))?;
    let grammar = json_schema::compile(&schema)?;
    let mut individual = vec![false; tokens.len()];
    let mut shared = vec![false; tokens.len()];
    let cases: [(&str, &[u8]); 5] = [
        ("start", b""),
        ("action_enum", br#"{"action":""#),
        ("query_text", br#"{"action":"search","query":"open the door"#),
        ("unicode_escape", br#"{"action":"answer","query":"\uD83D"#),
        ("complete", br#"{"action":"search","query":"hi","limit":3}"#),
    ];

    println!("case,vocabulary,allowed,individual_bytes,trie_bytes,individual_median_us,trie_median_us");
    for (label, prefix) in cases {
        let mut parser = Parser::new(&grammar);
        if !parser.commit(prefix) {
            return Err(format!("invalid benchmark prefix {label}").into());
        }
        let mut individual_us = [0; 9];
        let mut trie_us = [0; 9];
        let (mut individual_bytes, mut trie_bytes) = (0, 0);
        for trial in 0..9 {
            let order = if trial % 2 == 0 {
                [MaskMethod::Individual, MaskMethod::Trie]
            } else {
                [MaskMethod::Trie, MaskMethod::Individual]
            };
            for method in order {
                let start = Instant::now();
                match method {
                    MaskMethod::Individual => {
                        individual_bytes = vocabulary.mask(&mut parser, &mut individual, method);
                        individual_us[trial] = start.elapsed().as_micros();
                    }
                    MaskMethod::Trie => {
                        trie_bytes = vocabulary.mask(&mut parser, &mut shared, method);
                        trie_us[trial] = start.elapsed().as_micros();
                    }
                }
            }
            if individual != shared {
                return Err(format!("mask differs for prefix {label}").into());
            }
        }
        individual_us.sort_unstable();
        trie_us.sort_unstable();
        println!("{label},{},{},{individual_bytes},{trie_bytes},{},{}", tokens.len(),
                 shared.iter().filter(|allow| **allow).count(), individual_us[4], trie_us[4]);
    }
    Ok(())
}
