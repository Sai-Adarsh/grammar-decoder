//! Compile a closed subset of JSON Schema to the decoder's byte grammar.

mod syntax;

use crate::{Grammar, Symbol};
use serde_json::{Map, Value};
use std::collections::HashSet;
use std::fmt;
use syntax::{literal, Rules};

type Object = Map<String, Value>;

const MAX_VALUE_DEPTH: usize = 64;
const MAX_SCHEMA_VALUES: usize = 512;
const MAX_SCHEMA_TEXT_BYTES: usize = 8 * 1024;

#[derive(Debug)]
pub struct SchemaError {
    path: String,
    message: String,
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl std::error::Error for SchemaError {}

/// Build a grammar that emits compact JSON, with object keys in schema order.
///
/// Unsupported keywords and schema shapes fail with the location of the
/// offending constraint. See the README for the supported subset.
pub fn compile(schema: &Value) -> Result<Grammar, SchemaError> {
    SchemaBudget {
        values_left: MAX_SCHEMA_VALUES,
        text_left: MAX_SCHEMA_TEXT_BYTES,
    }
    .check(schema, 0)?;
    let mut rules = Rules::default();
    let root = compile_node(&mut rules, schema, "$", true)?;
    rules.finish(root).map_err(|message| error("$", message))
}

struct SchemaBudget {
    values_left: usize,
    text_left: usize,
}

impl SchemaBudget {
    fn check(&mut self, value: &Value, depth: usize) -> Result<(), SchemaError> {
        if depth >= MAX_VALUE_DEPTH {
            return Err(error("$", format!("schema exceeds {MAX_VALUE_DEPTH} JSON value levels")));
        }
        // Bound rule scaffolding by node count and terminal expansion by text size.
        // Check the whole tree, including literals, before serialization or rule creation.
        self.values_left = self.values_left.checked_sub(1)
            .ok_or_else(|| error("$", format!("schema exceeds {MAX_SCHEMA_VALUES} JSON values")))?;
        match value {
            Value::Object(fields) => {
                for (key, value) in fields {
                    self.charge_text(key.len())?;
                    self.check(value, depth + 1)?;
                }
            }
            Value::Array(items) => {
                for item in items {
                    self.check(item, depth + 1)?;
                }
            }
            Value::String(text) => self.charge_text(text.len())?,
            _ => {}
        }
        Ok(())
    }

    fn charge_text(&mut self, bytes: usize) -> Result<(), SchemaError> {
        self.text_left = self.text_left.checked_sub(bytes).ok_or_else(|| {
            error("$", format!("schema exceeds {MAX_SCHEMA_TEXT_BYTES} bytes of keys and strings"))
        })?;
        Ok(())
    }
}

fn compile_node(
    rules: &mut Rules,
    schema: &Value,
    path: &str,
    is_root: bool,
) -> Result<usize, SchemaError> {
    let schema = schema.as_object().ok_or_else(|| error(path, "expected a schema object"))?;
    if let Some(dialect) = schema.get("$schema") {
        if !is_root || dialect.as_str() != Some("https://json-schema.org/draft/2020-12/schema") {
            return Err(error(&child(path, "$schema"), "expected the Draft 2020-12 URI at the root"));
        }
    }

    let kind = schema.get("type").map(|value| Type::parse(value, path)).transpose()?;
    if schema.contains_key("enum") || schema.contains_key("const") {
        check_keywords(schema, path, &["type", "enum", "const"])?;
        let values = match (schema.get("enum"), schema.get("const")) {
            (Some(_), Some(_)) => return Err(error(path, "choose either enum or const")),
            (Some(choices), None) => choices.as_array()
                .filter(|values| !values.is_empty())
                .map(Vec::as_slice)
                .ok_or_else(|| error(&child(path, "enum"), "expected a nonempty array"))?,
            (None, Some(value)) => std::slice::from_ref(value),
            (None, None) => unreachable!("enum or const is present"),
        };
        let id = rules.reserve();
        for value in values {
            if !literal_has_exact_numbers(value) {
                return Err(error(path, "use i64 or u64 integer notation in enum and const values"));
            }
            if kind.is_some_and(|kind| !kind.matches(value)) {
                return Err(error(path, "enum or const value does not match the declared type"));
            }
            let json = serde_json::to_string(value).map_err(|err| error(path, err.to_string()))?;
            rules.add(id, literal(json.as_bytes()));
        }
        return Ok(id);
    }

    match kind {
        Some(Type::Object) => compile_object(rules, schema, path),
        Some(Type::Array) => {
            check_keywords(schema, path, &["type", "items"])?;
            let items_path = child(path, "items");
            let items = schema.get("items")
                .ok_or_else(|| error(&items_path, "array requires an items schema"))?;
            let item = compile_node(rules, items, &items_path, false)?;
            Ok(rules.array(item))
        }
        Some(kind) => {
            check_keywords(schema, path, &["type"])?;
            Ok(match kind {
                Type::String => rules.string(),
                Type::Integer => rules.number(false),
                Type::Number => rules.number(true),
                Type::Boolean => rules.choices([literal(b"true"), literal(b"false")]),
                Type::Null => rules.choices([literal(b"null")]),
                Type::Array | Type::Object => unreachable!("container types already handled"),
            })
        }
        None => {
            check_keywords(schema, path, &[])?;
            Err(error(path, "expected type, enum, or const"))
        }
    }
}

fn compile_object(
    rules: &mut Rules,
    schema: &Object,
    path: &str,
) -> Result<usize, SchemaError> {
    check_keywords(schema, path, &["type", "properties", "required", "additionalProperties"])?;
    if schema.get("additionalProperties").and_then(Value::as_bool) != Some(false) {
        return Err(error(&child(path, "additionalProperties"), "object requires false"));
    }

    let properties_path = child(path, "properties");
    let empty = Object::new();
    let properties = match schema.get("properties") {
        Some(value) => value.as_object().ok_or_else(|| error(&properties_path, "expected an object"))?,
        None => &empty,
    };
    let required_path = child(path, "required");
    let mut required = HashSet::new();
    if let Some(value) = schema.get("required") {
        let entries = value.as_array().ok_or_else(|| error(&required_path, "expected an array"))?;
        for (index, entry) in entries.iter().enumerate() {
            let entry_path = child(&required_path, &index.to_string());
            let name = entry.as_str().ok_or_else(|| error(&entry_path, "expected a property name"))?;
            if !properties.contains_key(name) {
                return Err(error(&entry_path, format!("property {name:?} is not declared")));
            }
            if !required.insert(name) {
                return Err(error(&entry_path, format!("property {name:?} appears twice")));
            }
        }
    }

    let mut fields = Vec::with_capacity(properties.len());
    for (name, schema) in properties {
        let path = child(&properties_path, name);
        let id = compile_node(rules, schema, &path, false)?;
        let key = serde_json::to_string(name).map_err(|err| error(&path, err.to_string()))?;
        let mut field = literal(key.as_bytes());
        field.extend([b':'.into(), Symbol::Rule(id)]);
        fields.push((field, required.contains(name.as_str())));
    }
    Ok(rules.object(fields))
}

fn check_keywords(schema: &Object, path: &str, allowed: &[&str]) -> Result<(), SchemaError> {
    for (key, value) in schema {
        match key.as_str() {
            "$schema" => {}
            "title" | "description" | "$comment" if value.is_string() => {}
            "title" | "description" | "$comment" => {
                return Err(error(&child(path, key), "expected a string"));
            }
            keyword if allowed.contains(&keyword) => {}
            _ => return Err(error(&child(path, key), "unsupported keyword here")),
        }
    }
    Ok(())
}

fn child(path: &str, key: &str) -> String {
    format!("{path}/{}", key.replace('~', "~0").replace('/', "~1"))
}

fn error(path: &str, message: impl Into<String>) -> SchemaError {
    SchemaError { path: path.into(), message: message.into() }
}

#[derive(Clone, Copy)]
enum Type {
    Object,
    Array,
    String,
    Integer,
    Number,
    Boolean,
    Null,
}

impl Type {
    fn parse(value: &Value, path: &str) -> Result<Self, SchemaError> {
        match value.as_str() {
            Some("object") => Ok(Self::Object),
            Some("array") => Ok(Self::Array),
            Some("string") => Ok(Self::String),
            Some("integer") => Ok(Self::Integer),
            Some("number") => Ok(Self::Number),
            Some("boolean") => Ok(Self::Boolean),
            Some("null") => Ok(Self::Null),
            _ => Err(error(&child(path, "type"), "expected one supported type name")),
        }
    }

    fn matches(self, value: &Value) -> bool {
        match self {
            Self::Object => value.is_object(),
            Self::Array => value.is_array(),
            Self::String => value.is_string(),
            Self::Integer => value.is_i64() || value.is_u64(),
            Self::Number => value.is_number(),
            Self::Boolean => value.is_boolean(),
            Self::Null => value.is_null(),
        }
    }
}

fn literal_has_exact_numbers(value: &Value) -> bool {
    // Decimal parsing can round the source text to f64 before compilation.
    // Restrict literal numbers to the lossless integer representations.
    match value {
        Value::Number(number) => number.is_i64() || number.is_u64(),
        Value::Array(values) => values.iter().all(literal_has_exact_numbers),
        Value::Object(fields) => fields.values().all(literal_has_exact_numbers),
        _ => true,
    }
}
