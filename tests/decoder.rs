use grammar_decoder::{
    decode, decode_with_vocabulary, json_schema::compile, ByteSet, DecodeError, DecodeOptions,
    Grammar, LogitSource, MaskMethod, Parser, Symbol, Vocabulary,
};
use serde_json::{json, Value};

fn accepts(grammar: &Grammar, bytes: &[u8]) -> bool {
    let mut parser = Parser::new(grammar);
    parser.commit(bytes) && parser.accepting()
}

#[test]
fn grammar_rejects_incomplete_or_invalid_production_sets() {
    type Case = (usize, Vec<(usize, Vec<Symbol>)>, &'static str);
    let cases: Vec<Case> = vec![
        (0, vec![], "no rules"),
        (0, vec![(usize::MAX, vec![])], "invalid rule id"),
        (0, vec![(usize::MAX - 1, vec![])], "invalid rule id"),
        (2, vec![(0, vec![])], "start rule has no productions"),
        (0, vec![(0, vec![Symbol::Rule(3)])], "undefined rule"),
        (
            0,
            vec![(0, vec![Symbol::Byte(ByteSet::ranges(&[]))])],
            "empty byte class",
        ),
        (
            0,
            vec![(0, vec![Symbol::Rule(0)])],
            "rule ids must be contiguous",
        ),
    ];
    for (start, rules, message) in cases {
        match Grammar::new(start, rules) {
            Err(error) => assert!(error.starts_with(message), "{error}"),
            Ok(_) => panic!("unexpected valid grammar"),
        }
    }
}

#[test]
fn trie_preserves_duplicate_tokens_and_rolls_back_failed_prefixes() {
    let grammar = Grammar::new(
        0,
        vec![
            (0, vec![Symbol::Rule(1), b'b'.into()]),
            (1, vec![]),
            (1, vec![b'a'.into()]),
        ],
    )
    .unwrap();
    let tokens = [
        b"ab".to_vec(),
        b"ax".to_vec(),
        b"a".to_vec(),
        b"ab".to_vec(),
        b"b".to_vec(),
        b"bb".to_vec(),
        vec![],
        b"ignored".to_vec(),
    ];
    let vocabulary = Vocabulary::new(&tokens, 7).unwrap();
    assert_eq!(vocabulary.trie_nodes(), 6);
    let mut parser = Parser::new(&grammar);
    assert!(!parser.commit(b"ax"));
    assert!(!parser.accepting());

    for expected in [
        [true, false, true, true, true, false, false, false],
        [false, false, false, false, true, false, false, false],
        [false, false, false, false, false, false, false, true],
    ] {
        for method in [MaskMethod::Individual, MaskMethod::Trie] {
            let mut mask = [true; 8];
            vocabulary.mask(&mut parser, &mut mask, method);
            assert_eq!(mask, expected);
        }
        if expected[2] {
            assert!(parser.commit(b"a"));
        } else if expected[4] {
            assert!(parser.commit(b"b"));
        }
    }
    assert!(parser.accepting());
}

struct Scores {
    step: usize,
    dense: bool,
    accepted: Vec<usize>,
}

impl LogitSource for Scores {
    fn logits(&mut self, output: &mut [f64]) {
        if self.dense {
            output.fill(-30.0);
        }
        output[2] = 9.0;
        output[if self.step == 0 { 0 } else { 3 }] = 5.0;
    }

    fn accept(&mut self, token: usize) {
        self.step += 1;
        self.accepted.push(token);
    }
}

fn options(eos: usize, max_tokens: usize, temperature: f64) -> DecodeOptions {
    DecodeOptions {
        eos,
        max_tokens,
        temperature,
        seed: 5,
    }
}

#[test]
fn decoder_runs_dense_and_sparse_paths_without_emitting_illegal_tokens() {
    let grammar = compile(&json!({"const":"ok"})).unwrap();
    let mut tokens = vec![br#""ok""#.to_vec(), Vec::new(), b"bad".to_vec(), Vec::new()];
    tokens.extend((0..128).map(|index| format!("bad{index}").into_bytes()));
    let vocabulary = Vocabulary::new(&tokens, 3).unwrap();
    for (dense, temperature) in [(true, 0.2), (false, 0.0)] {
        let mut source = Scores {
            dense,
            step: 0,
            accepted: vec![],
        };
        let output = decode_with_vocabulary(
            &grammar,
            &vocabulary,
            &mut source,
            options(3, 1, temperature),
        )
        .unwrap();
        assert_eq!(output.bytes, br#""ok""#);
        assert_eq!(output.token_ids, [0]);
        assert_eq!(source.accepted, [0]);
        assert!(output.rejected_candidates > 0);
        assert!(output.attempted_mask_bytes > 0);
    }

    let mut source = Scores {
        dense: false,
        step: 0,
        accepted: vec![],
    };
    for opt in [
        options(4, 1, 0.0),
        options(3, 1, f64::NAN),
        options(3, 1, -0.1),
    ] {
        let error = decode_with_vocabulary(&grammar, &vocabulary, &mut source, opt)
            .err()
            .unwrap();
        assert!(matches!(error, DecodeError::InvalidConfig));
        assert_eq!(error.to_string(), "invalid sampling configuration");
    }
    assert!(matches!(
        decode(
            &grammar,
            &tokens,
            &mut source,
            options(tokens.len(), 1, 0.0)
        ),
        Err(DecodeError::InvalidConfig)
    ));
    let max_tokens = decode(&grammar, &tokens, &mut source, options(3, 0, 0.0))
        .err()
        .unwrap();
    assert!(matches!(max_tokens, DecodeError::MaxTokens));
    assert_eq!(
        max_tokens.to_string(),
        "generation ended before the grammar accepted EOS"
    );
    source.step = 1;
    let error = decode(&grammar, &tokens, &mut source, options(3, 1, 0.0))
        .err()
        .unwrap();
    assert!(matches!(error, DecodeError::NoViableToken));
    assert_eq!(
        error.to_string(),
        "model assigned no finite score to a legal token"
    );
}

#[test]
fn sampler_returns_a_legal_token_when_upper_endpoint_rounding_exhausts_the_weights() {
    struct NearEndpoint(bool);

    impl LogitSource for NearEndpoint {
        fn logits(&mut self, output: &mut [f64]) {
            if self.0 {
                output[4] = 0.0;
            } else {
                output[..4].copy_from_slice(&[0.0, -3.5, -3.5, -3.5]);
            }
        }

        fn accept(&mut self, _: usize) {
            self.0 = true;
        }
    }

    let grammar = Grammar::new(
        0,
        vec![(0, vec![Symbol::Byte(ByteSet::ranges(&[(b'a', b'd')]))])],
    )
    .unwrap();
    let tokens = [vec![b'a'], vec![b'b'], vec![b'c'], vec![b'd'], vec![]];
    let output = decode(
        &grammar,
        &tokens,
        &mut NearEndpoint(false),
        DecodeOptions {
            // The first uniform draw is the largest representable draw below one.
            seed: 0x3162_8af6_7b21_31ab,
            ..options(4, 1, 1.0)
        },
    )
    .unwrap();
    assert_eq!(output.token_ids, [0]);
    assert_eq!(output.bytes, b"a");
}

#[test]
fn schema_covers_closed_objects_arrays_numbers_unicode_and_literal_types() {
    let grammar = compile(&json!({
        "$schema":"https://json-schema.org/draft/2020-12/schema",
        "$comment":"fixture",
        "type":"object", "additionalProperties":false,
        "properties": {
            "tags":{"type":"array", "items":{"type":"string"}},
            "count":{"type":"integer"},
            "score":{"type":"number"},
            "flag":{"type":"boolean"},
            "none":{"type":"null"}
        },
        "required":["count"]
    }))
    .unwrap();
    for valid in [
        br#"{"count":0}"#.as_slice(),
        br#"{"tags":[],"count":-7,"score":-0.33E+2,"flag":false,"none":null}"#,
        br#"{"tags":["\ud83d\ude03","\b\f\n\r\t\/\\\""],"count":10,"score":7.25,"flag":true}"#,
        r#"{"tags":["café","日","😃"],"count":1,"score":2e-2}"#.as_bytes(),
    ] {
        assert!(
            accepts(&grammar, valid),
            "{:?}",
            String::from_utf8_lossy(valid)
        );
    }
    for invalid in [
        br#"{}"#.as_slice(),
        br#"{"count":00}"#,
        br#"{"count":1,"extra":1}"#,
        br#"{"count":1,"tags":[]}"#,
        br#"{"tags":["\ud83d"],"count":1}"#,
        br#"{"tags":["\ude03"],"count":1}"#,
        br#"{"tags":["\q"],"count":1}"#,
        b"{\"tags\":[\"\xc0\x80\"],\"count\":1}",
        b"{\"tags\":[\"\n\"],\"count\":1}",
    ] {
        assert!(
            !accepts(&grammar, invalid),
            "{:?}",
            String::from_utf8_lossy(invalid)
        );
    }
    for (schema, valid) in [
        (json!({"type":"object","additionalProperties":false}), "{}"),
        (
            json!({"type":"array","const":[1,true,null]}),
            "[1,true,null]",
        ),
        (json!({"type":"object","const":{"a":1}}), r#"{"a":1}"#),
        (json!({"type":"string","const":"text"}), r#""text""#),
        (
            json!({"type":"integer","enum":[-1, u64::MAX]}),
            "18446744073709551615",
        ),
        (json!({"type":"number","const":1}), "1"),
        (json!({"type":"boolean","const":false}), "false"),
        (json!({"type":"null","const":null}), "null"),
    ] {
        assert!(accepts(&compile(&schema).unwrap(), valid.as_bytes()));
    }
}

#[test]
fn unsupported_schemas_report_the_constraint_and_preflight_limits() {
    let mut deep = json!(null);
    for _ in 0..65 {
        deep = json!([deep]);
    }
    let wide = Value::Array(vec![Value::Null; 512]);
    let cases = [
        (json!(true), "expected a schema object"),
        (json!({"$schema":"old","type":"string"}), "/$schema"),
        (json!({"type":["string","null"]}), "/type"),
        (json!({"type":"integer","minimum":0}), "/minimum"),
        (
            json!({"type":"array","items":{"type":"string"},"minItems":1}),
            "/minItems",
        ),
        (
            json!({"type":"array","items":{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"string"}}),
            "/items/$schema",
        ),
        (
            json!({"type":"object","additionalProperties":false,"maxProperties":1}),
            "/maxProperties",
        ),
        (json!({"type":"string","title":42}), "/title"),
        (json!({}), "expected type"),
        (json!({"properties":{}}), "/properties"),
        (json!({"const":1,"enum":[2]}), "choose either"),
        (json!({"const":"hi","minLength":1}), "/minLength"),
        (json!({"enum":[]}), "/enum"),
        (json!({"enum":1}), "/enum"),
        (json!({"const":[1.1]}), "integer notation"),
        (json!({"const":{"a":1.1}}), "integer notation"),
        (json!({"type":"integer","const":false}), "declared type"),
        (json!({"type":"array"}), "/items"),
        (json!({"type":"object"}), "/additionalProperties"),
        (
            json!({"type":"object","additionalProperties":false,"properties":[]}),
            "/properties",
        ),
        (
            json!({"type":"object","additionalProperties":false,"required":true}),
            "/required",
        ),
        (
            json!({"type":"object","additionalProperties":false,"required":[0]}),
            "expected a property name",
        ),
        (
            json!({"type":"object","additionalProperties":false,"required":["missing"]}),
            "is not declared",
        ),
        (
            json!({"type":"object","additionalProperties":false,"properties":{"a":{"type":"string"}},"required":["a","a"]}),
            "appears twice",
        ),
        (
            json!({"type":"object","additionalProperties":false,"properties":{"a~/b":{"type":"string","format":"uri"}}}),
            "/properties/a~0~1b/format",
        ),
        (json!({"const":deep}), "JSON value levels"),
        (json!({"const":wide}), "JSON values"),
        (
            json!({"const":"x".repeat(8 * 1024)}),
            "bytes of keys and strings",
        ),
        (
            json!({"const":{"a".repeat(8 * 1024):null}}),
            "bytes of keys and strings",
        ),
    ];
    for (schema, message) in cases {
        let error = compile(&schema).err().expect("schema rejected");
        assert!(error.to_string().contains(message), "{schema}: {error}");
    }
}
