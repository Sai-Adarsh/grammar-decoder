# Grammar-constrained decoding

A Rust sampler uses an incremental Earley parser to check the bytes of each candidate token. It samples only legal tokens at the configured temperature, advances the parser after selection, and admits the end-of-sequence token only when the grammar accepts the output. Running out of tokens or finding no viable candidate returns an error.

```sh
cargo run --release
```

The fixture assigns the highest logit to an illegal token and splits a UTF-8 code point and a JSON escape across token boundaries. Its grammar accepts one object with a fixed key order. `action` is `search` or `answer`, `query` is a JSON string, and `limit` is an integer from 1 to 10. The decoder returns the bytes and emitted token IDs.

To attach a model, implement `LogitSource` for next-token scores and model state updates. Pass a vocabulary indexed by token ID and identify the end-of-sequence token. For tokenizers with context-dependent detokenization, adapt the piece-to-byte decoding before passing it to the parser. Use `Grammar::new` to supply a different context-free grammar.

The validity guarantee covers successful decoding against the supplied grammar. JSON Schema compilation and Pydantic integration are outside the library. Validate cross-field and application rules separately.
