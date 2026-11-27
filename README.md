# Grammar-constrained decoding

A Rust sampler uses an incremental Earley parser to check the bytes of each candidate token. It samples only legal tokens at the configured temperature, advances the parser after selection, and admits the end-of-sequence token only when the grammar accepts the output. Running out of tokens or finding no viable candidate returns an error.

```sh
cargo run --locked --release
```

The fixture assigns the highest logit to an illegal token and splits a UTF-8 code point and a JSON escape across token boundaries. The grammar comes from [the tool call schema](examples/tool-call.schema.json). It accepts `search` or `answer` as the action, a query string, and an integer limit from 1 to 10.

## Decode with a JSON Schema

```sh
printf '%s\n' '{"jobs":[{"name":"index","timeout":1.5},{"name":"clean","dryRun":true}],"version":1}' |
  cargo run --locked --release -- --schema examples/batch.schema.json
```

The CLI reads a compact JSON candidate from stdin. A fixture assigns a score to each candidate byte, and the sampler applies the grammar before selecting it. The command prints the completed document after accepting the end-of-sequence token. Rejection reports the byte offset.

The compiler supports this subset of [JSON Schema Draft 2020-12](https://json-schema.org/draft/2020-12/json-schema-validation):

- Closed objects with `additionalProperties: false`, declared `properties`, and optional `required`. Fields appear in schema order and can be omitted unless required.
- Arrays with one `items` schema. An array may be empty.
- `string`, `number`, `integer`, `boolean`, and `null`. The integer grammar emits digits with an optional leading minus.
- `enum` or `const` with JSON literals. The only additional constraint allowed here is `type`, which must match every literal. Numeric literals use integer notation within the `i64` or `u64` range. Literal objects use their declared key order.

The optional root `$schema` must name Draft 2020-12. The compiler accepts `title`, `description`, and `$comment` as string annotations. It rejects unsupported constructs such as references, union types, and length or range constraints, and reports the location in the schema. The grammar emits compact JSON. A valid document with a different field order or formatting may fail to parse.

Before expanding rules, the compiler caps schemas at 64 nested JSON value levels, 512 values, and 8 KiB of combined UTF-8 key and string bytes. These limits include values in `enum` and `const`. The CLI accepts up to 64 KiB of schema input and 16 KiB of candidate input, counting whitespace.

## Model integration

Implement `LogitSource` for next-token scores and model state updates. Pass a vocabulary indexed by token ID and identify the end-of-sequence token. For tokenizers with context-dependent detokenization, adapt piece-to-byte decoding before passing it to the parser.

Pass a `serde_json::Value` to `json_schema::compile`, or use `Grammar::new` to supply another context-free grammar. Validate application-specific rules separately.
