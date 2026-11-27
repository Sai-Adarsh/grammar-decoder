# Grammar-constrained decoding

A Rust sampler that constrains model output to a grammar, including JSON from supported schemas.

## Run the fixture

The demo prints a JSON tool call.

```sh
cargo run --locked --release
```

- [JSON Schema and supported features](docs/reference.md#decode-with-a-json-schema)
- [Run TinyStories with llama.cpp](docs/reference.md#run-a-real-model-with-llamacpp)
- [GPT-2 mask benchmark](docs/reference.md#masking-with-the-gpt-2-vocabulary)
- [Full reference](docs/reference.md)
