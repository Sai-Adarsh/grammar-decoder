use serde_json::{json, Map, Value};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Work(PathBuf);
impl Work {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let name = format!(
            "grammar-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let path = std::env::temp_dir().join(name);
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn write(&self, name: &str, bytes: impl AsRef<[u8]>) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, bytes).unwrap();
        path
    }
}

impl Drop for Work {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn cli(args: &[&str], input: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_grammar-decoder"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    if let Err(error) = child.stdin.take().unwrap().write_all(input) {
        assert_eq!(error.kind(), std::io::ErrorKind::BrokenPipe);
    }
    child.wait_with_output().unwrap()
}

fn failure(out: Output, needle: &str) {
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(stderr.contains(needle), "{stderr}");
}

#[test]
fn fixture_and_schema_entry_points_decode_valid_json() {
    let fixture = cli(&[], b"");
    assert!(fixture.status.success());
    let actual: Value = serde_json::from_slice(&fixture.stdout).unwrap();
    assert_eq!(
        actual,
        json!({"action":"search","query":"café and \"quotes\"","limit":3})
    );

    let work = Work::new();
    let path = work.write(
        "schema with spaces.json",
        include_bytes!("../examples/batch.schema.json"),
    );
    let candidate =
        br#"{"jobs":[{"name":"index","timeout":1.5},{"name":"clean","dryRun":true}],"version":1}"#;
    let out = cli(&["--schema", path.to_str().unwrap()], candidate);
    assert!(out.status.success(), "{:?}", out.stderr);
    assert_eq!(out.stdout.strip_suffix(b"\n"), Some(candidate.as_slice()));
}

#[test]
fn schema_cli_rejects_partial_or_oversized_documents() {
    let work = Work::new();
    let path = work.write("number.json", br#"{"type":"integer"}"#);
    failure(cli(&["--other"], b""), "usage:");
    failure(
        cli(&["--schema", path.to_str().unwrap()], b"x"),
        "candidate rejected at byte 0",
    );
    failure(
        cli(&["--schema", path.to_str().unwrap()], b"-"),
        "candidate rejected at byte 1",
    );
    failure(
        cli(
            &["--schema", path.to_str().unwrap()],
            &[b' '; 16 * 1024 + 1],
        ),
        "candidate input exceeds",
    );
    let large_schema = work.write("large.json", vec![b' '; 64 * 1024 + 1]);
    failure(
        cli(&["--schema", large_schema.to_str().unwrap()], b""),
        "schema input exceeds",
    );
    let malformed = work.write("bad.json", b"not-json");
    failure(
        cli(&["--schema", malformed.to_str().unwrap()], b""),
        "expected ident",
    );
    let unsupported = work.write("unsupported.json", br#"{"type":"integer","minimum":0}"#);
    failure(
        cli(&["--schema", unsupported.to_str().unwrap()], b"1"),
        "/minimum",
    );
}

#[cfg(not(feature = "native-llama"))]
#[test]
fn native_benchmark_reports_the_required_feature() {
    failure(
        cli(
            &[
                "--llama-bench",
                "model",
                "schema",
                "prompts",
                "48",
                "1",
                "4",
            ],
            b"",
        ),
        "rebuild with --features native-llama",
    );
}

#[cfg(feature = "native-llama")]
#[test]
fn native_benchmark_validates_inputs_before_loading_models() {
    failure(
        cli(
            &["--llama-bench", "model", "schema", "prompts", "0", "1", "4"],
            b"",
        ),
        "MAX_TOKENS must be an integer from 1 to 256",
    );
    let work = Work::new();
    let schema = work.write(
        "schema.json",
        include_bytes!("../examples/scene.schema.json"),
    );
    let duplicate = work.write(
        "prompts.json",
        br#"[{"id":"a","prompt":"one"},{"id":"a","prompt":"two"}]"#,
    );
    failure(
        cli(
            &[
                "--llama-bench",
                "missing-model",
                schema.to_str().unwrap(),
                duplicate.to_str().unwrap(),
                "48",
                "1",
                "4",
            ],
            b"",
        ),
        "unique nonempty IDs",
    );
}

#[cfg(feature = "native-llama")]
#[test]
#[ignore = "requires the cached TinyStories GGUF in GRAMMAR_TEST_MODEL"]
fn native_model_emits_independently_validated_json_and_full_logit_rows() {
    let model = std::env::var("GRAMMAR_TEST_MODEL").expect("set GRAMMAR_TEST_MODEL");
    let work = Work::new();
    let schema = work.write(
        "schema.json",
        include_bytes!("../examples/scene.schema.json"),
    );
    let prompts = work.write(
        "prompts.json",
        include_bytes!("../examples/scene-prompts.json"),
    );
    let out = cli(
        &[
            "--llama-bench",
            &model,
            schema.to_str().unwrap(),
            prompts.to_str().unwrap(),
            "48",
            "1",
            "1",
        ],
        b"",
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let rows: Vec<Value> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|row| serde_json::from_str(row).unwrap())
        .collect();
    assert_eq!(rows.len(), 7);
    assert_eq!(rows[0]["kind"], "config");
    assert_eq!(rows[0]["vocabulary_size"], 32_000);
    assert!(rows[1..].iter().all(|row| {
        row["columns_per_logit_row"] == 32_000
            && row["nonfinite_logits"] == 0
            && row["logit_rows"].as_u64().unwrap() == row["generated_tokens"].as_u64().unwrap() + 1
    }));
    let constrained: Vec<_> = rows[1..]
        .iter()
        .filter(|row| row["mode"] == "constrained")
        .collect();
    assert_eq!(constrained.len(), 3);
    for row in constrained {
        assert_eq!(row["json_valid"], true);
        assert_eq!(row["schema_valid"], true);
        assert_eq!(row["status"], "eos");
        assert!(row["mask_attempts"].as_u64().unwrap() > 0);
        let generated: Value = serde_json::from_str(row["output"].as_str().unwrap()).unwrap();
        assert_eq!(generated.as_object().unwrap().len(), 3);
    }
}

fn tokenizer() -> Value {
    let mut vocab = Map::new();
    for id in 0..50_253 {
        vocab.insert(format!("ÿpiece{id}"), json!(id));
    }
    vocab.insert("Ġ".into(), json!(50_253));
    vocab.insert("Ċ".into(), json!(50_254));
    vocab.insert("d".into(), json!(50_255));
    vocab.insert("<|endoftext|>".into(), json!(50_256));
    json!({
        "decoder":{"type":"ByteLevel"},
        "model":{"vocab":vocab},
        "added_tokens":[{"id":50_256,"content":"<|endoftext|>","special":true}]
    })
}

#[test]
fn real_sized_bytelevel_fixture_compares_all_three_masks() {
    let work = Work::new();
    let path = work.write("tokens.json", serde_json::to_vec(&tokenizer()).unwrap());
    let out = cli(&["--mask-bench", path.to_str().unwrap()], b"");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let csv = String::from_utf8(out.stdout).unwrap();
    assert_eq!(csv.lines().count(), 6);
    assert!(csv.contains("case,vocabulary,allowed,individual_bytes,trie_bytes,"));
    assert!(csv.contains(",projected_bytes,trie_median_ns,projected_median_ns"));
    assert!(csv.contains("query_text,50257,"));
    assert!(csv.contains("complete,50257,1,"));
    let reductions: Vec<_> = csv
        .lines()
        .skip(1)
        .map(|row| {
            let columns: Vec<_> = row.split(',').collect();
            let raw = columns[4].parse::<usize>().unwrap();
            let projected = columns[7].parse::<usize>().unwrap();
            raw.checked_sub(projected)
                .expect("projection cannot add transitions")
        })
        .collect();
    assert!(reductions.into_iter().any(|reduction| reduction > 0));
}

#[test]
fn mask_bench_refuses_incompatible_tokenizer_metadata() {
    let work = Work::new();
    let mut input = tokenizer();
    let check = |input: &Value, message: &str| {
        let path = work.write("tokens.json", serde_json::to_vec(input).unwrap());
        failure(cli(&["--mask-bench", path.to_str().unwrap()], b""), message);
    };
    input["added_tokens"] = json!([]);
    check(&input, "missing GPT-2 special EOS declaration");
    input["model"]["vocab"]["ÿpiece3"] = json!(0);
    check(&input, "duplicate token ID");
    input["model"]["vocab"]["ÿpiece3"] = json!(99999);
    check(&input, "token ID is out of range");
    input["model"]["vocab"]["ÿpiece3"] = json!("3");
    check(&input, "noninteger token ID");
    let vocab = input["model"]["vocab"].as_object_mut().unwrap();
    vocab.remove("ÿpiece3");
    vocab.insert("😃".into(), json!(3));
    check(&input, "unsupported ByteLevel codepoint");
    let vocab = input["model"]["vocab"].as_object_mut().unwrap();
    vocab.remove("<|endoftext|>");
    vocab.insert("replacement".into(), json!(50_256));
    check(&input, "missing GPT-2 EOS");
    input["model"]["vocab"] = json!({});
    check(&input, "expected 50,257");
    input["model"] = json!({});
    check(&input, "missing model.vocab");
    input["decoder"]["type"] = json!("WordPiece");
    check(&input, "expected a GPT-2 ByteLevel tokenizer");
}
