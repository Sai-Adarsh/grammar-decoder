use grammar_decoder::{decode, ByteSet, DecodeOptions, Grammar, LogitSource, Symbol};
use Symbol::Rule;

fn lit(bytes: &[u8]) -> Vec<Symbol> {
    bytes.iter().copied().map(Symbol::from).collect()
}

fn class(ranges: &[(u8, u8)]) -> Symbol {
    Symbol::Byte(ByteSet::ranges(ranges))
}

fn tool_call_grammar() -> Grammar {
    const CALL: usize = 0;
    const ACTION: usize = 1;
    const LIMIT: usize = 2;
    const STRING: usize = 3;
    const BODY: usize = 4;
    const CH: usize = 5;
    const HEX: usize = 6;
    const NON_SURROGATE: usize = 7;
    const HIGH_SURROGATE: usize = 8;
    const LOW_SURROGATE: usize = 9;
    let mut root = lit(b"{\"action\":");
    root.push(Rule(ACTION));
    root.extend(lit(b",\"query\":"));
    root.push(Rule(STRING));
    root.extend(lit(b",\"limit\":"));
    root.push(Rule(LIMIT));
    root.extend(lit(b"}"));

    let cont = || class(&[(0x80, 0xbf)]);
    let hex = || Rule(HEX);
    let start_escape = || lit(br"\u");
    let mut unicode = start_escape();
    unicode.push(Rule(NON_SURROGATE));
    let mut surrogate_pair = start_escape();
    surrogate_pair.push(Rule(HIGH_SURROGATE));
    surrogate_pair.extend(start_escape());
    surrogate_pair.push(Rule(LOW_SURROGATE));
    let mut rules = vec![
        (CALL, root),
        (ACTION, lit(b"\"search\"")),
        (ACTION, lit(b"\"answer\"")),
        (LIMIT, vec![class(&[(b'1', b'9')])]),
        (LIMIT, lit(b"10")),
        (STRING, vec![b'"'.into(), Rule(BODY), b'"'.into()]),
        (BODY, vec![]),
        (BODY, vec![Rule(CH), Rule(BODY)]),
        (CH, vec![class(&[(0x20, 0x21), (0x23, 0x5b), (0x5d, 0x7f)])]),
        (CH, vec![class(&[(0xc2, 0xdf)]), cont()]),
        (CH, vec![0xe0.into(), class(&[(0xa0, 0xbf)]), cont()]),
        (CH, vec![class(&[(0xe1, 0xec), (0xee, 0xef)]), cont(), cont()]),
        (CH, vec![0xed.into(), class(&[(0x80, 0x9f)]), cont()]),
        (CH, vec![0xf0.into(), class(&[(0x90, 0xbf)]), cont(), cont()]),
        (CH, vec![class(&[(0xf1, 0xf3)]), cont(), cont(), cont()]),
        (CH, vec![0xf4.into(), class(&[(0x80, 0x8f)]), cont(), cont()]),
        (CH, unicode),
        (CH, surrogate_pair),
        (HEX, vec![class(&[(b'0', b'9'), (b'A', b'F'), (b'a', b'f')])]),
        (NON_SURROGATE, vec![class(&[(b'0', b'9'), (b'A', b'C'), (b'E', b'F'), (b'a', b'c'), (b'e', b'f')]), hex(), hex(), hex()]),
        (NON_SURROGATE, vec![class(&[(b'D', b'D'), (b'd', b'd')]), class(&[(b'0', b'7')]), hex(), hex()]),
        (HIGH_SURROGATE, vec![class(&[(b'D', b'D'), (b'd', b'd')]), class(&[(b'8', b'9'), (b'A', b'B'), (b'a', b'b')]), hex(), hex()]),
        (LOW_SURROGATE, vec![class(&[(b'D', b'D'), (b'd', b'd')]), class(&[(b'C', b'F'), (b'c', b'f')]), hex(), hex()]),
    ];
    for escaped in [b'"', b'\\', b'/', b'b', b'f', b'n', b'r', b't'] {
        rules.push((CH, vec![b'\\'.into(), escaped.into()]));
    }
    Grammar::new(CALL, rules).expect("valid tool-call grammar")
}

struct FixtureLogits(usize);

impl LogitSource for FixtureLogits {
    fn logits(&mut self, output: &mut [f64]) {
        // The top scorer is illegal in every position, including after the final brace.
        output[0] = 15.0;
        output[(self.0 + 1).min(8)] = 8.0;
        for id in 1..output.len() {
            if output[id] == f64::NEG_INFINITY {
                output[id] = -30.0;
            }
        }
    }

    fn accept(&mut self, _: usize) {
        self.0 += 1;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let output = decode(
        &tool_call_grammar(),
        &vocabulary,
        &mut FixtureLogits(0),
        DecodeOptions { eos: 8, max_tokens: 16, temperature: 0.7, seed: 42 },
    )?;
    println!("{}", std::str::from_utf8(&output.bytes)?);
    eprintln!(
        "{} emitted tokens, {} candidate rejections",
        output.token_ids.len(),
        output.rejected_candidates
    );
    Ok(())
}
