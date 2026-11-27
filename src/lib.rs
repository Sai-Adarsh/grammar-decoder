pub mod json_schema;

use std::collections::HashSet;
use std::fmt;

#[derive(Clone)]
pub struct ByteSet([u64; 4]);

impl ByteSet {
    pub fn ranges(ranges: &[(u8, u8)]) -> Self {
        let mut bits = [0; 4];
        for &(first, last) in ranges {
            for byte in first..=last {
                bits[usize::from(byte) / 64] |= 1 << (byte % 64);
            }
        }
        Self(bits)
    }

    fn contains(&self, byte: u8) -> bool {
        self.0[usize::from(byte) / 64] & (1 << (byte % 64)) != 0
    }

    fn is_empty(&self) -> bool {
        self.0.iter().all(|bits| *bits == 0)
    }
}

#[derive(Clone)]
pub enum Symbol {
    Rule(usize),
    Byte(ByteSet),
}

impl From<u8> for Symbol {
    fn from(byte: u8) -> Self {
        Self::Byte(ByteSet::ranges(&[(byte, byte)]))
    }
}

pub struct Grammar {
    productions: Vec<(usize, Vec<Symbol>)>,
    by_lhs: Vec<Vec<usize>>,
    nullable: Vec<bool>,
}

impl Grammar {
    pub fn new(start: usize, rules: Vec<(usize, Vec<Symbol>)>) -> Result<Self, &'static str> {
        let max_lhs = rules.iter().map(|(lhs, _)| *lhs).max().ok_or("no rules")?;
        let augmented = max_lhs.checked_add(1).ok_or("invalid rule id")?;
        let num_rules = augmented.checked_add(1).ok_or("invalid rule id")?;
        if start > max_lhs {
            return Err("start rule has no productions");
        }

        let mut productions = vec![(augmented, vec![Symbol::Rule(start)])];
        productions.extend(rules);
        let mut by_lhs = vec![vec![]; num_rules];
        for (id, (lhs, rhs)) in productions.iter().enumerate() {
            by_lhs[*lhs].push(id);
            if rhs.iter().any(|symbol| matches!(symbol, Symbol::Rule(rule) if *rule > max_lhs)) {
                return Err("undefined rule");
            }
            if rhs.iter().any(|symbol| matches!(symbol, Symbol::Byte(set) if set.is_empty())) {
                return Err("empty byte class");
            }
        }

        // Productive rules keep every surviving parser prefix extendable to a sentence.
        let fixed_point = |epsilon_only: bool| {
            let mut resolved = vec![false; num_rules];
            loop {
                let mut changed = false;
                for (lhs, rhs) in &productions {
                    if !resolved[*lhs]
                        && rhs.iter().all(|symbol| match symbol {
                            Symbol::Byte(_) => !epsilon_only,
                            Symbol::Rule(rule) => resolved[*rule],
                        })
                    {
                        resolved[*lhs] = true;
                        changed = true;
                    }
                }
                if !changed {
                    return resolved;
                }
            }
        };
        if fixed_point(false).iter().any(|valid| !valid) {
            return Err("rule ids must be contiguous and all rules must derive terminal bytes");
        }
        let nullable = fixed_point(true);
        Ok(Self { productions, by_lhs, nullable })
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct Item {
    production: usize,
    dot: usize,
    origin: usize,
}

impl Item {
    fn advance(self) -> Self {
        Self { dot: self.dot + 1, ..self }
    }
}

#[derive(Default)]
struct Column {
    items: Vec<Item>,
    seen: HashSet<Item>,
}

impl Column {
    fn insert(&mut self, item: Item) {
        if self.seen.insert(item) {
            self.items.push(item);
        }
    }
}

pub struct Parser<'grammar> {
    grammar: &'grammar Grammar,
    chart: Vec<Column>,
}

impl<'grammar> Parser<'grammar> {
    pub fn new(grammar: &'grammar Grammar) -> Self {
        let mut parser = Self { grammar, chart: vec![Column::default()] };
        parser.chart[0].insert(Item { production: 0, dot: 0, origin: 0 });
        parser.close();
        parser
    }

    fn close(&mut self) {
        let current = self.chart.len() - 1;
        let mut cursor = 0;
        while cursor < self.chart[current].items.len() {
            let item = self.chart[current].items[cursor];
            cursor += 1;
            let (lhs, rhs) = &self.grammar.productions[item.production];
            match rhs.get(item.dot) {
                Some(Symbol::Byte(_)) => {}
                Some(Symbol::Rule(rule)) => {
                    for &production in &self.grammar.by_lhs[*rule] {
                        self.chart[current].insert(Item { production, dot: 0, origin: current });
                    }
                    if self.grammar.nullable[*rule] {
                        self.chart[current].insert(item.advance());
                    }
                }
                None => {
                    // Same-column completions are already advanced by the nullable check.
                    if item.origin == current {
                        continue;
                    }
                    for index in 0..self.chart[item.origin].items.len() {
                        let previous = self.chart[item.origin].items[index];
                        let rhs = &self.grammar.productions[previous.production].1;
                        if matches!(rhs.get(previous.dot), Some(Symbol::Rule(rule)) if rule == lhs) {
                            self.chart[current].insert(previous.advance());
                        }
                    }
                }
            }
        }
    }

    fn scan(&mut self, byte: u8) -> bool {
        let mut next = Column::default();
        for &item in &self.chart.last().expect("initial column").items {
            let rhs = &self.grammar.productions[item.production].1;
            if matches!(rhs.get(item.dot), Some(Symbol::Byte(set)) if set.contains(byte)) {
                next.insert(item.advance());
            }
        }
        if next.items.is_empty() {
            return false;
        }
        self.chart.push(next);
        self.close();
        true
    }

    pub fn accepting(&self) -> bool {
        self.chart
            .last()
            .expect("initial column")
            .seen
            .contains(&Item { production: 0, dot: 1, origin: 0 })
    }

    /// Feeds all bytes of a token atomically. A rejected token leaves the state untouched.
    pub fn commit(&mut self, bytes: &[u8]) -> bool {
        let checkpoint = self.chart.len();
        if bytes.iter().all(|byte| self.scan(*byte)) {
            true
        } else {
            self.chart.truncate(checkpoint);
            false
        }
    }

    fn viable(&mut self, bytes: &[u8]) -> bool {
        let checkpoint = self.chart.len();
        let valid = self.commit(bytes);
        self.chart.truncate(checkpoint);
        valid
    }
}

pub trait LogitSource {
    /// Write one logit per vocabulary entry for the next token in this generation.
    fn logits(&mut self, output: &mut [f64]);
    /// Advance the model state by the selected vocabulary index.
    fn accept(&mut self, token_id: usize);
}

#[derive(Debug)]
pub enum DecodeError {
    InvalidConfig,
    NoViableToken,
    MaxTokens,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidConfig => "invalid sampling configuration",
            Self::NoViableToken => "model assigned no finite score to a legal token",
            Self::MaxTokens => "generation ended before the grammar accepted EOS",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for DecodeError {}

pub struct DecodeOptions {
    pub eos: usize,
    pub max_tokens: usize,
    pub temperature: f64,
    pub seed: u64,
}

pub struct Generation {
    pub bytes: Vec<u8>,
    pub token_ids: Vec<usize>,
    pub rejected_candidates: usize,
}

pub fn decode(
    grammar: &Grammar,
    vocabulary: &[Vec<u8>],
    source: &mut impl LogitSource,
    options: DecodeOptions,
) -> Result<Generation, DecodeError> {
    if options.eos >= vocabulary.len() || !options.temperature.is_finite() || options.temperature < 0.0 {
        return Err(DecodeError::InvalidConfig);
    }

    let mut parser = Parser::new(grammar);
    let mut logits = vec![f64::NEG_INFINITY; vocabulary.len()];
    let mut candidates = Vec::new();
    let mut generation = Generation { bytes: vec![], token_ids: vec![], rejected_candidates: 0 };
    let mut rng = options.seed;
    for step in 0..=options.max_tokens {
        candidates.clear();
        logits.fill(f64::NEG_INFINITY);
        source.logits(&mut logits);
        for (id, (&score, token)) in logits.iter().zip(vocabulary).enumerate() {
            if !score.is_finite() {
                continue;
            }
            if if id == options.eos {
                parser.accepting()
            } else {
                !token.is_empty() && parser.viable(token)
            } {
                candidates.push((id, score));
            } else {
                generation.rejected_candidates += 1;
            }
        }
        let token_id = sample(&candidates, options.temperature, &mut rng).ok_or(DecodeError::NoViableToken)?;
        if token_id == options.eos {
            return Ok(generation);
        }
        if step == options.max_tokens {
            return Err(DecodeError::MaxTokens);
        }
        let bytes = &vocabulary[token_id];
        assert!(parser.commit(bytes));
        generation.bytes.extend(bytes);
        generation.token_ids.push(token_id);
        source.accept(token_id);
    }
    unreachable!()
}

fn sample(candidates: &[(usize, f64)], temperature: f64, rng: &mut u64) -> Option<usize> {
    let &(best, maximum) = candidates.iter().max_by(|a, b| a.1.total_cmp(&b.1))?;
    if temperature == 0.0 {
        return Some(best);
    }
    let weight = |score: f64| ((score - maximum) / temperature).exp();
    let total: f64 = candidates.iter().map(|&(_, score)| weight(score)).sum();
    *rng = rng.wrapping_add(0x9e3779b97f4a7c15);
    let mut bits = *rng;
    bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
    bits ^= bits >> 31;
    let mut draw = (bits >> 11) as f64 * (1.0 / ((1u64 << 53) as f64)) * total;
    for &(id, score) in candidates {
        draw -= weight(score);
        if draw < 0.0 {
            return Some(id);
        }
    }
    Some(best)
}
