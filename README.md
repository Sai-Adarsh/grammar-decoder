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

## Masking with a real vocabulary

The `Vocabulary` index shares byte prefixes across tokens. One traversal advances the Earley chart once per viable trie edge and rolls it back before visiting a sibling. Invalid prefixes skip whole subtrees. Duplicate byte spellings keep each token ID, and EOS is allowed only when the original parser state accepts. Reuse the index across requests with `decode_with_vocabulary`. When finite logits are sparse, the decoder checks only those candidates individually.

Download the [GPT-2 ByteLevel tokenizer](https://huggingface.co/openai-community/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json) at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` and run:

```sh
cargo run --locked --release -- --mask-bench /path/to/tokenizer.json
```

One Apple Silicon run used 50,257 token IDs and a 98,024-node trie. The two masks matched exactly in every state and every trial. Timing is the median of nine alternating passes and excludes index construction, tokenization, and model execution.

| Parser state | Allowed IDs | Individual byte attempts | Trie byte attempts | Individual µs | Trie µs |
|---|---:|---:|---:|---:|---:|
| Start | 2 | 50,259 | 259 | 338 | 2 |
| Action enum | 8 | 52,508 | 354 | 518 | 4 |
| Query text | 50,032 | 320,705 | 97,929 | 312,983 | 105,414 |
| Unicode escape | 1 | 50,271 | 264 | 388 | 3 |
| Complete | 1 | 50,256 | 256 | 382 | 2 |

The free-form string state still visits most of the trie. The measurements show where prefix sharing helps and where additional mask caching or a different parser representation would matter. This benchmark uses a real tokenizer vocabulary with predetermined parser states. The two decoding examples still use fixture logits.

## Model integration

Implement `LogitSource` for next-token scores and model state updates. Pass a vocabulary indexed by token ID and identify the end-of-sequence token. For tokenizers with context-dependent detokenization, adapt piece-to-byte decoding before passing it to the parser.

Pass a `serde_json::Value` to `json_schema::compile`, or use `Grammar::new` to supply another context-free grammar. Validate application-specific rules separately.

## Tests and coverage

```sh
cargo test --locked
```

The tests exercise the public parser, decoder, and schema compiler, then run all three CLI modes. The mask benchmark uses an offline fixture with 50,257 token IDs and checks that the two algorithms agree. The real GPT-2 performance experiment above is separate.

On macOS, use Rust's source-based instrumentation and Xcode's LLVM tools:

```sh
cov_dir=$(mktemp -d)
mkdir "$cov_dir/raw"
CARGO_TARGET_DIR="$cov_dir/build" RUSTFLAGS="-Cinstrument-coverage" \
  LLVM_PROFILE_FILE="$cov_dir/raw/%m-%p.profraw" cargo test --locked
xcrun llvm-profdata merge -sparse "$cov_dir"/raw/*.profraw -o "$cov_dir/merged.profdata"
decoder_tests=$(find "$cov_dir/build/debug/deps" -type f -name 'decoder-*' -perm -111)
cli_tests=$(find "$cov_dir/build/debug/deps" -type f -name 'cli-*' -perm -111)
xcrun llvm-cov report "$cov_dir/build/debug/grammar-decoder" \
  -object "$decoder_tests" -object "$cli_tests" -instr-profile "$cov_dir/merged.profdata" \
  src/lib.rs src/json_schema.rs src/json_schema/syntax.rs src/main.rs src/mask_benchmark.rs
```

Ten tests covered 798 of 808 production lines (98.76%) and 627 of 652 regions (96.17%). Stable Rust emitted no branch counters for this build. The remaining lines handle compiler invariants, errors from serializing in-memory JSON, and failures of the bundled demo and benchmark checks.
