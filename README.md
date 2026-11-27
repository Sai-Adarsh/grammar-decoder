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

## Share parser states across equivalent bytes

`Vocabulary::new` shares exact token-byte prefixes. `Vocabulary::for_grammar` can merge additional prefixes when the grammar treats different bytes identically. It partitions the 256 bytes by membership in every terminal byte set and uses the smallest byte in each class as its representative. Two equivalent prefixes have the same length and match the same terminals at each position, so they produce the same Earley chart. Each trie node retains all original token IDs that end there. Sampling and commit still use the original bytes.

The one-shot `decode` entry point builds the projected trie. With `decode_with_vocabulary`, reuse a prepared trie across requests. A different byte partition triggers per-token scanning with the original bytes. Sparse finite logits also use per-token checks. In either path EOS is legal only when the current parser state accepts. Each chart column indexes terminal transitions and items waiting on nonterminals. The parser retains at most 64 rolled-back columns to reuse their storage.

## Masking with the GPT-2 vocabulary

Download the [GPT-2 ByteLevel tokenizer](https://huggingface.co/openai-community/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json) at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` and run:

```sh
cargo run --locked --release -- --mask-bench /path/to/tokenizer.json
```

The measured tokenizer had SHA-256 `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`.

The benchmark compiles [the tool call schema](examples/tool-call.schema.json), then tests five prefixes. It times the original per-token reference, exact byte trie, and projected trie in the same release process. All three paths use separate parsers, one untimed warmup per state, and 12 timed passes in balanced order. Each trial compares the complete masks across all 50,257 token IDs. The CSV reports attempted chart transitions and medians using wall-clock `Instant`. Vocabulary builds, prefix setup, and exact equality checks are outside the mask timings. The projected build includes the byte partition. The existing microsecond columns remain available for the reference and exact byte trie.

The [recorded run](evidence/gpt2-byte-projection.csv) used rustc 1.87.0 on arm64 macOS 27.2.1. The exact trie uses the traversal from `4f85196`. Across seven independent release processes and all 420 timed trials, the projected trie and exact trie masks both matched the per-token reference bit for bit. The table takes the median of the seven process medians.

| Parser state | Allowed IDs | Exact trie attempts | Projected trie attempts | Exact trie median (ns) | Projected trie median (ns) |
|---|---:|---:|---:|---:|---:|
| Start | 2 | 259 | 53 | 1,708 | 1,021 |
| Action enum | 8 | 354 | 135 | 2,875 | 1,896 |
| Query text | 50,032 | 97,929 | 80,472 | 45,783,354 | 37,949,271 |
| Unicode escape | 1 | 264 | 54 | 1,645 | 896 |
| Complete | 1 | 256 | 50 | 1,208 | 791 |

For query text, the per-process medians ranged from 43.81 to 49.23 ms for the exact trie and from 35.60 to 39.72 ms for the projected trie. The benchmark built the exact trie first and the projected trie second. They had 98,024 and 80,538 nodes respectively, with median build times of 11.11 and 7.07 ms. The smallest timings are sensitive to timer overhead. These measurements cover mask generation on one host with fixed schema prefixes. The [earlier run](evidence/gpt2-earley-before-after.csv) measured the chart indexing changes separately.

## Model integration

Implement `LogitSource` for next-token scores and model state updates. Pass a vocabulary indexed by token ID and identify the end-of-sequence token. For tokenizers with context-dependent detokenization, adapt piece-to-byte decoding before passing it to the parser.

Pass a `serde_json::Value` to `json_schema::compile`, or use `Grammar::new` to supply another context-free grammar. Validate application-specific rules separately.

## Tests and coverage

```sh
cargo test --locked
```

The tests cover overlapping terminal byte sets, duplicate token IDs, split UTF-8, foreign grammar reuse, all 256 distinct byte classes, and the public decoding paths. An offline fixture with 50,257 IDs exercises the benchmark CLI and checks exact agreement across all three paths.

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

Fourteen tests covered 921 of 931 production lines (98.93%) and 678 of 705 regions (96.17%). Stable Rust emitted no branch counters for this build. The remaining lines handle compiler invariants, errors from serializing in-memory JSON, and failures of the bundled demo and benchmark checks.
