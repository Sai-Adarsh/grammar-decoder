use crate::{ByteSet, Grammar, Symbol};
use Symbol::Rule;

#[derive(Default)]
pub(super) struct Rules(Vec<Vec<Vec<Symbol>>>);

impl Rules {
    pub(super) fn reserve(&mut self) -> usize {
        let id = self.0.len();
        self.0.push(Vec::new());
        id
    }

    pub(super) fn add(&mut self, id: usize, symbols: Vec<Symbol>) {
        self.0[id].push(symbols);
    }

    pub(super) fn choices(&mut self, alternatives: impl IntoIterator<Item = Vec<Symbol>>) -> usize {
        let id = self.reserve();
        self.0[id].extend(alternatives);
        id
    }

    pub(super) fn finish(self, root: usize) -> Result<Grammar, &'static str> {
        let rules = self.0.into_iter().enumerate()
            .flat_map(|(id, alternatives)| alternatives.into_iter().map(move |rhs| (id, rhs)))
            .collect();
        Grammar::new(root, rules)
    }

    pub(super) fn object(&mut self, fields: Vec<(Vec<Symbol>, bool)>) -> usize {
        let end = self.choices([vec![]]);
        let (mut no_fields, mut some_fields) = (end, end);
        // Track separately whether a prior field has emitted, so skipped optional
        // fields cannot introduce leading, doubled, or trailing commas.
        for (mut field, required) in fields.into_iter().rev() {
            field.push(Rule(some_fields));
            let mut with_comma = vec![b','.into()];
            with_comma.extend(field.clone());
            let next_no_fields = self.choices([field]);
            let next_some_fields = self.choices([with_comma]);
            if !required {
                self.add(next_no_fields, vec![Rule(no_fields)]);
                self.add(next_some_fields, vec![Rule(some_fields)]);
            }
            (no_fields, some_fields) = (next_no_fields, next_some_fields);
        }
        self.choices([vec![b'{'.into(), Rule(no_fields), b'}'.into()]])
    }

    pub(super) fn array(&mut self, item: usize) -> usize {
        let rest = self.reserve();
        self.add(rest, vec![]);
        self.add(rest, vec![Rule(rest), b','.into(), Rule(item)]);
        self.choices([
            literal(b"[]"),
            vec![b'['.into(), Rule(item), Rule(rest), b']'.into()],
        ])
    }

    pub(super) fn number(&mut self, decimal: bool) -> usize {
        let digits = self.reserve();
        self.add(digits, vec![]);
        self.add(digits, vec![Rule(digits), digit()]);
        let unsigned = self.choices([literal(b"0"), vec![class(&[(b'1', b'9')]), Rule(digits)]]);
        let minus = self.choices([vec![], literal(b"-")]);
        let mut number = vec![Rule(minus), Rule(unsigned)];
        if decimal {
            let fraction = self.choices([vec![], vec![b'.'.into(), digit(), Rule(digits)]]);
            let sign = self.choices([vec![], literal(b"+"), literal(b"-")]);
            let exponent = self.choices([
                vec![],
                vec![class(&[(b'E', b'E'), (b'e', b'e')]), Rule(sign), digit(), Rule(digits)],
            ]);
            number.extend([Rule(fraction), Rule(exponent)]);
        }
        self.choices([number])
    }

    pub(super) fn string(&mut self) -> usize {
        let hex = self.choices([vec![class(&[(b'0', b'9'), (b'A', b'F'), (b'a', b'f')])]]);
        let d = || class(&[(b'D', b'D'), (b'd', b'd')]);
        let basic_escape = self.choices([
            vec![
                class(&[(b'0', b'9'), (b'A', b'C'), (b'E', b'F'), (b'a', b'c'), (b'e', b'f')]),
                Rule(hex),
                Rule(hex),
                Rule(hex),
            ],
            vec![d(), class(&[(b'0', b'7')]), Rule(hex), Rule(hex)],
        ]);
        let high = self.choices([
            vec![d(), class(&[(b'8', b'9'), (b'A', b'B'), (b'a', b'b')]), Rule(hex), Rule(hex)],
        ]);
        let low = self.choices([
            vec![d(), class(&[(b'C', b'F'), (b'c', b'f')]), Rule(hex), Rule(hex)],
        ]);
        let mut unicode = literal(br"\u");
        unicode.push(Rule(basic_escape));
        let mut surrogate_pair = literal(br"\u");
        surrogate_pair.push(Rule(high));
        surrogate_pair.extend(literal(br"\u"));
        surrogate_pair.push(Rule(low));

        let cont = || class(&[(0x80, 0xbf)]);
        let character = self.choices([
            vec![class(&[(0x20, 0x21), (0x23, 0x5b), (0x5d, 0x7f)])],
            vec![class(&[(0xc2, 0xdf)]), cont()],
            vec![0xe0.into(), class(&[(0xa0, 0xbf)]), cont()],
            vec![class(&[(0xe1, 0xec), (0xee, 0xef)]), cont(), cont()],
            vec![0xed.into(), class(&[(0x80, 0x9f)]), cont()],
            vec![0xf0.into(), class(&[(0x90, 0xbf)]), cont(), cont()],
            vec![class(&[(0xf1, 0xf3)]), cont(), cont(), cont()],
            vec![0xf4.into(), class(&[(0x80, 0x8f)]), cont(), cont()],
            unicode,
            surrogate_pair,
            vec![b'\\'.into(), class(&[(b'"', b'"'), (b'\\', b'\\'), (b'/', b'/'),
                (b'b', b'b'), (b'f', b'f'), (b'n', b'n'), (b'r', b'r'), (b't', b't')])],
        ]);
        let body = self.reserve();
        self.add(body, vec![]);
        self.add(body, vec![Rule(body), Rule(character)]);
        self.choices([vec![b'"'.into(), Rule(body), b'"'.into()]])
    }
}

pub(super) fn literal(bytes: &[u8]) -> Vec<Symbol> {
    bytes.iter().copied().map(Symbol::from).collect()
}

fn digit() -> Symbol {
    class(&[(b'0', b'9')])
}

fn class(ranges: &[(u8, u8)]) -> Symbol {
    Symbol::Byte(ByteSet::ranges(ranges))
}
