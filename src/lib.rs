pub mod json_schema;

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::OnceLock;

#[derive(Clone, Eq, Hash, PartialEq)]
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
    byte_representatives: OnceLock<[u8; 256]>,
}

impl Grammar {
    pub fn new(start: usize, rules: Vec<(usize, Vec<Symbol>)>) -> Result<Self, &'static str> {
        let max_lhs = rules.iter().map(|(lhs, _)| *lhs).max().ok_or("no rules")?;
        let augmented = max_lhs.checked_add(1).ok_or("invalid rule id")?;
        let num_rules = augmented.checked_add(1).ok_or("invalid rule id")?;
        if start > max_lhs {
            return Err("start rule has no productions");
        }
        let productive_rules_error =
            "rule ids must be contiguous and all rules must derive terminal bytes";
        // Contiguous IDs require at least one production per ID, starting at zero.
        if max_lhs >= rules.len() {
            return Err(productive_rules_error);
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
            return Err(productive_rules_error);
        }
        let nullable = fixed_point(true);
        Ok(Self { productions, by_lhs, nullable, byte_representatives: OnceLock::new() })
    }

    fn byte_representatives(&self) -> &[u8; 256] {
        self.byte_representatives.get_or_init(|| {
            byte_representatives(
                self.productions.iter().flat_map(|(_, rhs)| rhs).filter_map(|symbol| match symbol {
                    Symbol::Byte(set) => Some(set),
                    Symbol::Rule(_) => None,
                }),
            )
        })
    }
}

fn byte_representatives<'a>(sets: impl Iterator<Item = &'a ByteSet>) -> [u8; 256] {
    let mut representatives = [0; 256];
    let mut seen = HashSet::new();
    for set in sets.filter(|&set| seen.insert(set)) {
        // Refine each existing class by membership in this terminal. The
        // smallest byte in each class is its actual scanner input.
        let mut first = [None; 512];
        let mut classes = 0;
        for (byte, representative) in (0..=u8::MAX).zip(&mut representatives) {
            let part = usize::from(*representative) * 2 + usize::from(set.contains(byte));
            *representative = *first[part].get_or_insert_with(|| {
                classes += 1;
                byte
            });
        }
        if classes == 256 {
            break;
        }
    }
    representatives
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
    terminals: Vec<Item>,
    waiting_heads: HashMap<usize, usize>,
    waiting: Vec<(Item, Option<usize>)>,
}

impl Column {
    fn insert(&mut self, item: Item) {
        if self.seen.insert(item) {
            self.items.push(item);
        }
    }

    fn wait_for(&mut self, rule: usize, item: Item) {
        let previous = self.waiting_heads.insert(rule, self.waiting.len());
        self.waiting.push((item, previous));
    }

    fn clear(&mut self) {
        self.items.clear();
        self.seen.clear();
        self.terminals.clear();
        self.waiting_heads.clear();
        self.waiting.clear();
    }
}

const REUSABLE_COLUMNS: usize = 64;

pub struct Parser<'grammar> {
    grammar: &'grammar Grammar,
    chart: Vec<Column>,
    scratch: Vec<Column>,
}

impl<'grammar> Parser<'grammar> {
    pub fn new(grammar: &'grammar Grammar) -> Self {
        let mut parser = Self { grammar, chart: vec![Column::default()], scratch: Vec::new() };
        parser.chart[0].insert(Item { production: 0, dot: 0, origin: 0 });
        parser.close();
        parser
    }

    fn close(&mut self) {
        let current = self.chart.len() - 1;
        let (prior, active) = self.chart.split_at_mut(current);
        let column = &mut active[0];
        let mut cursor = 0;
        while cursor < column.items.len() {
            let item = column.items[cursor];
            cursor += 1;
            let (lhs, rhs) = &self.grammar.productions[item.production];
            match rhs.get(item.dot) {
                Some(Symbol::Byte(_)) => column.terminals.push(item),
                Some(Symbol::Rule(rule)) => {
                    column.wait_for(*rule, item);
                    for &production in &self.grammar.by_lhs[*rule] {
                        column.insert(Item { production, dot: 0, origin: current });
                    }
                    if self.grammar.nullable[*rule] {
                        column.insert(item.advance());
                    }
                }
                None => {
                    // Same-column completions are already advanced by the nullable check.
                    if item.origin == current {
                        continue;
                    }
                    let origin = &prior[item.origin];
                    let mut waiter = origin.waiting_heads.get(lhs).copied();
                    while let Some(index) = waiter {
                        let (previous, next) = origin.waiting[index];
                        column.insert(previous.advance());
                        waiter = next;
                    }
                }
            }
        }
    }

    fn scan(&mut self, byte: u8) -> bool {
        let grammar = self.grammar;
        let mut matches = self.chart.last().expect("initial column").terminals.iter()
            .filter_map(|&item| {
                let rhs = &grammar.productions[item.production].1;
                matches!(rhs.get(item.dot), Some(Symbol::Byte(set)) if set.contains(byte))
                    .then(|| item.advance())
            });
        let Some(first) = matches.next() else {
            return false;
        };
        let mut next = self.scratch.pop().unwrap_or_default();
        next.insert(first);
        for item in matches {
            next.insert(item);
        }
        self.chart.push(next);
        self.close();
        true
    }

    fn recycle(&mut self, mut column: Column) {
        if self.scratch.len() < REUSABLE_COLUMNS {
            column.clear();
            self.scratch.push(column);
        }
    }

    fn restore(&mut self, checkpoint: usize) {
        while self.chart.len() > checkpoint {
            let column = self.chart.pop().expect("column after checkpoint");
            self.recycle(column);
        }
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
            self.restore(checkpoint);
            false
        }
    }

    fn viable(&mut self, bytes: &[u8], attempted_bytes: &mut usize) -> bool {
        let checkpoint = self.chart.len();
        let valid = bytes.iter().all(|byte| {
            *attempted_bytes += 1;
            self.scan(*byte)
        });
        self.restore(checkpoint);
        valid
    }
}

#[derive(Default)]
struct TrieNode {
    edges: Vec<(u8, usize)>,
    token_ids: Vec<usize>,
}

/// An immutable vocabulary trie reusable across requests.
pub struct Vocabulary<'tokens> {
    tokens: &'tokens [Vec<u8>],
    eos: usize,
    nodes: Vec<TrieNode>,
    byte_representatives: Option<[u8; 256]>,
}

impl<'tokens> Vocabulary<'tokens> {
    pub fn new(tokens: &'tokens [Vec<u8>], eos: usize) -> Result<Self, DecodeError> {
        Self::build(tokens, eos, None)
    }

    /// Build a trie that shares transitions for bytes indistinguishable by the grammar.
    pub fn for_grammar(
        tokens: &'tokens [Vec<u8>],
        eos: usize,
        grammar: &Grammar,
    ) -> Result<Self, DecodeError> {
        Self::build(tokens, eos, Some(*grammar.byte_representatives()))
    }

    fn build(
        tokens: &'tokens [Vec<u8>],
        eos: usize,
        byte_representatives: Option<[u8; 256]>,
    ) -> Result<Self, DecodeError> {
        if eos >= tokens.len() {
            return Err(DecodeError::InvalidConfig);
        }
        let identity = std::array::from_fn(|byte| byte as u8);
        let alphabet = byte_representatives.as_ref().unwrap_or(&identity);
        let mut nodes = vec![TrieNode::default()];
        for (id, token) in tokens.iter().enumerate() {
            if id == eos || token.is_empty() {
                continue;
            }
            let mut parent = 0;
            for &byte in token {
                let byte = alphabet[usize::from(byte)];
                parent = match nodes[parent].edges.iter().find(|edge| edge.0 == byte) {
                    Some(&(_, child)) => child,
                    None => {
                        let child = nodes.len();
                        nodes.push(TrieNode::default());
                        nodes[parent].edges.push((byte, child));
                        child
                    }
                };
            }
            nodes[parent].token_ids.push(id);
        }
        Ok(Self { tokens, eos, nodes, byte_representatives })
    }

    pub fn trie_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Fill a dense mask and return the number of attempted parser byte transitions.
    /// The individual path checks original bytes and also serves as a fallback when
    /// a prepared trie is used with a grammar whose byte partition differs.
    pub fn mask(&self, parser: &mut Parser<'_>, allowed: &mut [bool], method: MaskMethod) -> usize {
        assert_eq!(allowed.len(), self.tokens.len());
        allowed.fill(false);
        let method = match method {
            MaskMethod::Trie if self.byte_representatives
                .as_ref()
                .is_some_and(|alphabet| alphabet != parser.grammar.byte_representatives()) =>
                MaskMethod::Individual,
            other => other,
        };
        let mut attempted = 0;
        match method {
            MaskMethod::Individual => {
                for (id, token) in self.tokens.iter().enumerate() {
                    if id != self.eos && !token.is_empty() {
                        allowed[id] = parser.viable(token, &mut attempted);
                    }
                }
            }
            MaskMethod::Trie => {
                // One parser column per successful edge. Popping an exhausted child
                // restores its parent so each sibling starts from the same state.
                let checkpoint = parser.chart.len();
                let mut stack = vec![(0, 0)];
                while let Some((node, next_edge)) = stack.last_mut() {
                    if let Some(&(byte, child)) = self.nodes[*node].edges.get(*next_edge) {
                        *next_edge += 1;
                        attempted += 1;
                        if parser.scan(byte) {
                            for &id in &self.nodes[child].token_ids {
                                allowed[id] = true;
                            }
                            stack.push((child, 0));
                        }
                    } else {
                        stack.pop();
                        parser.restore(checkpoint + stack.len().saturating_sub(1));
                    }
                }
            }
        }
        allowed[self.eos] = parser.accepting();
        attempted
    }
}

#[derive(Clone, Copy)]
pub enum MaskMethod {
    Individual,
    Trie,
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
    /// Maximum number of non-EOS tokens. At the limit only EOS can complete a
    /// generation, and it must still have a finite score and an accepting parse.
    pub max_tokens: usize,
    pub temperature: f64,
    pub seed: u64,
}

pub struct Generation {
    pub bytes: Vec<u8>,
    pub token_ids: Vec<usize>,
    pub rejected_candidates: usize,
    pub attempted_mask_bytes: usize,
}

pub fn decode(
    grammar: &Grammar,
    vocabulary: &[Vec<u8>],
    source: &mut impl LogitSource,
    options: DecodeOptions,
) -> Result<Generation, DecodeError> {
    decode_with_vocabulary(
        grammar,
        &Vocabulary::for_grammar(vocabulary, options.eos, grammar)?,
        source,
        options,
    )
}

pub fn decode_with_vocabulary(
    grammar: &Grammar,
    vocabulary: &Vocabulary<'_>,
    source: &mut impl LogitSource,
    options: DecodeOptions,
) -> Result<Generation, DecodeError> {
    if options.eos != vocabulary.eos || !options.temperature.is_finite() || options.temperature < 0.0 {
        return Err(DecodeError::InvalidConfig);
    }

    let mut parser = Parser::new(grammar);
    let mut logits = vec![f64::NEG_INFINITY; vocabulary.tokens.len()];
    let mut allowed = vec![false; vocabulary.tokens.len()];
    let mut candidates = Vec::new();
    let mut generation = Generation {
        bytes: vec![], token_ids: vec![], rejected_candidates: 0, attempted_mask_bytes: 0,
    };
    let mut rng = options.seed;
    for step in 0..=options.max_tokens {
        candidates.clear();
        logits.fill(f64::NEG_INFINITY);
        source.logits(&mut logits);
        // A continuation cannot displace EOS once the output budget is spent.
        if step == options.max_tokens && parser.accepting() && logits[options.eos].is_finite() {
            return Ok(generation);
        }
        let finite = logits.iter().filter(|score| score.is_finite()).count();
        // Pre-filtered logits can be sparse enough that walking the whole trie
        // costs more than parsing the few candidates independently.
        let dense = finite > (vocabulary.tokens.len() / 8).max(16);
        if dense {
            generation.attempted_mask_bytes += vocabulary.mask(&mut parser, &mut allowed, MaskMethod::Trie);
        }
        for (id, &score) in logits.iter().enumerate() {
            if !score.is_finite() {
                continue;
            }
            let token = &vocabulary.tokens[id];
            if if dense {
                allowed[id]
            } else if id == options.eos {
                parser.accepting()
            } else {
                !token.is_empty() && parser.viable(token, &mut generation.attempted_mask_bytes)
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
        let bytes = &vocabulary.tokens[token_id];
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
    let weight = |score: f64| {
        let difference = score - maximum;
        let scaled = if difference.is_finite() {
            difference / temperature
        } else {
            // The finite scores have opposite signs. Scale first to avoid overflow.
            score / temperature - maximum / temperature
        };
        scaled.exp()
    };
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
