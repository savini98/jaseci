"""Partial-JSON streaming parser for structured outputs.

Extracts characters from a designated top-level string field of a JSON
object as chunks of the JSON text arrive. Used by `dispatch_streaming`
to support typed returns: when a `by llm()` function has a non-`str`
return type and `stream=True`, byLLM sends `response_format` to the
provider, the model emits a JSON object matching the schema token by
token, and the parser surfaces *only* the characters of the designated
text field (e.g. `answer`) to the caller — leaving the rest of the JSON
buffered so it can be parsed in full once streaming completes.

Design notes:

  - Top-level only. The parser tracks `{}` and `[]` depth to ensure
    matches against the target key happen at depth 1 (i.e. a property of
    the outermost object). Nested objects' keys named the same as the
    target are correctly skipped.
  - String-aware. Inside JSON strings the parser ignores braces and
    brackets, and respects backslash escapes including `\\"`.
  - Robust to partial chunks. The state machine survives any chunk
    boundary: target key split across chunks, escape sequence split
    across chunks, value boundary mid-stream.
  - Escape decoding. Standard JSON escapes (`\\n`, `\\t`, `\\"`, `\\\\`,
    `\\/`, `\\r`, `\\b`, `\\f`) are decoded to their character form.
    `\\u` escapes are passed through as raw text for now (rare in
    practice for streamed model outputs); a follow-up can decode them.

The buffer of received raw chunks is preserved on the extractor so
callers can `json.loads(extractor.raw)` once `is_done()` is true.
"""

"""Pick the streaming text field for a typed return.

Resolution order:
  1. Explicit `streaming_text_field` in call_params
  2. `"answer"` if the response_format schema has a top-level string property
     by that name (the convention used by `MemoryAwareReply` and the docs
     in `byllm-react-implementation-plan.md`)
  3. The first top-level `string` property in the schema
  4. `"answer"` as a last-resort fallback (extractor will simply never
     match and yield nothing — caller still gets `.parsed` after stream
     ends, but no token-level streaming)

`response_schema` is the JSON schema dict (the value of `response_format`'s
inner `json_schema.schema`, or whatever shape `mt_run.get_output_schema`
returns). When it can't be introspected, we fall back to "answer".
"""
def resolve_streaming_text_field(
    explicit: str | None,
    response_schema: dict | None
) -> str {
    if explicit {
        return explicit;
    }
    if response_schema is None {
        return "answer";
    }
    # The schema may be wrapped in {"type": "json_schema", "json_schema":
    # {"schema": {...}}} (OpenAI shape) or be the raw schema dict.
    schema = response_schema;
    if isinstance(schema, dict) and "json_schema" in schema {
        inner = schema.get("json_schema") or {};
        if isinstance(inner, dict) {
            schema = inner.get("schema") or schema;
        }
    }
    props = (
        schema.get("properties")
            if isinstance(schema, dict)
            else None
    );
    if not isinstance(props, dict) {
        return "answer";
    }
    if "answer" in props {
        candidate = props.get("answer");
        if isinstance(candidate, dict) and candidate.get("type") == "string" {
            return "answer";
        }
    }
    for (name, spec) in props.items() {
        if isinstance(spec, dict) and spec.get("type") == "string" {
            return name;
        }
    }
    return "answer";
}


obj StreamingFieldExtractor {
    """Extracts characters from a designated top-level JSON string field
    as JSON chunks arrive.

    Usage:
        ex = StreamingFieldExtractor(field_name="answer");
        for chunk in stream {
            tokens = ex.feed(chunk);
            if tokens {
                yield tokens;
            }
        }
        # After streaming completes, ex.raw holds the full JSON text;
        # caller should json.loads(ex.raw) to extract structured fields.
    """

    has field_name: str;

    # Internal state — do not access directly from outside.
    has _raw: str = "";
    has _state: str = "scanning";   # scanning | in_value | in_escape | done
    has _depth: int = 0;            # 0 = outside outer {}, 1 = at top level
    has _in_string: bool = False;   # are we inside a non-target string?
    has _string_escape: bool = False;
    has _key_buffer: str = "";      # last completed top-level key (when seen)
    has _matched_key: bool = False; # we matched the target key, awaiting `:`
    has _seeking_value_start: bool = False;  # we're past `:`, awaiting opening `"`

    def feed(chunk: str) -> str;
    def is_done() -> bool;
    def get_raw() -> str;
    def _decode_escape(ch: str) -> str;
}


impl StreamingFieldExtractor.feed(chunk: str) -> str {
    self._raw += chunk;
    out: list[str] = [];
    i = 0;
    n = len(chunk);

    while i < n {
        ch = chunk[i];

        if self._state == "done" {
            # Already finished extracting the target value; just buffer
            # the rest of the raw stream.
            break;
        }

        if self._state == "in_value" {
            # We're inside the target field's string value.
            if ch == "\\" {
                self._state = "in_escape";
                i += 1;
                continue;
            }
            if ch == "\"" {
                # End of target value.
                self._state = "done";
                i += 1;
                continue;
            }
            out.append(ch);
            i += 1;
            continue;
        }

        if self._state == "in_escape" {
            # Single-character escape decoding. \uXXXX is rare in
            # streamed model outputs and would require buffering 4 hex
            # digits — passed through as raw "\\u" + chars for now.
            decoded = self._decode_escape(ch);
            out.append(decoded);
            self._state = "in_value";
            i += 1;
            continue;
        }

        # state == "scanning": parse JSON top-level structure to find
        # the target key.
        if self._in_string {
            # Inside a non-target string (could be a key being read or
            # a different value's string). Track escapes and the closing
            # quote.
            if self._string_escape {
                self._string_escape = False;
                if self._depth == 1 and not self._matched_key {
                    self._key_buffer += ch;
                }
                i += 1;
                continue;
            }
            if ch == "\\" {
                self._string_escape = True;
                if self._depth == 1 and not self._matched_key {
                    self._key_buffer += ch;
                }
                i += 1;
                continue;
            }
            if ch == "\"" {
                self._in_string = False;
                # The string just ended. If we were buffering a top-level
                # key, this is the moment to check it against the target.
                if self._depth == 1 and not self._matched_key {
                    if self._key_buffer == self.field_name {
                        # Mark; the next non-whitespace char should be ':'.
                        self._matched_key = True;
                    }
                    self._key_buffer = "";
                }
                i += 1;
                continue;
            }
            if self._depth == 1 and not self._matched_key {
                self._key_buffer += ch;
            }
            i += 1;
            continue;
        }

        # Not inside any string.
        if ch == "\"" {
            self._in_string = True;
            self._key_buffer = "";
            # If we just saw `:` and were waiting for the value's opening
            # quote, this is it — start streaming chars.
            if self._seeking_value_start {
                self._seeking_value_start = False;
                self._matched_key = False;
                self._state = "in_value";
                self._in_string = False;
            }
            i += 1;
            continue;
        }
        if ch == "{" {
            self._depth += 1;
            i += 1;
            continue;
        }
        if ch == "}" {
            self._depth -= 1;
            i += 1;
            continue;
        }
        if ch == "[" {
            self._depth += 1;
            i += 1;
            continue;
        }
        if ch == "]" {
            self._depth -= 1;
            i += 1;
            continue;
        }
        if ch == ":" {
            if self._matched_key {
                # We matched the target key; the next `"` opens the
                # value we want to stream.
                self._seeking_value_start = True;
            }
            i += 1;
            continue;
        }
        # Whitespace, commas, numbers, true/false/null tokens at top
        # level — irrelevant to key matching; just advance.
        i += 1;
    }

    return "".join(out);
}


impl StreamingFieldExtractor.is_done() -> bool {
    return self._state == "done";
}


impl StreamingFieldExtractor.get_raw() -> str {
    return self._raw;
}


import from typing { Iterator }


obj StructuredStream {
    """Iterable wrapper around a structured-streaming generator.

    Returned from the public `by llm()` call site when a function declares
    a typed return AND `stream=True`. Iterating yields the streamed tokens
    of the designated text field (default `answer`). Once iteration is
    exhausted (or `parsed` is accessed first), the parsed structured
    instance — including any non-streamed fields like `citations` — is
    available via `.parsed`.

    The underlying generator's body is what populates
    `runtime.streaming_result` on completion, so this wrapper simply
    drains-then-fetches.
    """

    has _generator: Iterator[str];
    has _runtime: object;          # MTRuntime, but typed loosely to avoid cycles
    has _exhausted: bool = False;
    has _drain_count: int = 0;     # debug counter; tokens dropped during drain

    def __iter__() -> Iterator[str] {
        return self;
    }

    def __next__() -> str {
        try {
            return next(self._generator);
        } except StopIteration as e {
            self._exhausted = True;
            raise e;
        }
    }

    @property
    def parsed -> object {
        if not self._exhausted {
            try {
                while True {
                    next(self._generator);
                    self._drain_count += 1;
                }
            } except StopIteration {
                self._exhausted = True;
            }
        }
        rt = self._runtime;
        if rt is None {
            return None;
        }
        return getattr(rt, "streaming_result", None);
    }
}


impl StreamingFieldExtractor._decode_escape(ch: str) -> str {
    if ch == "n" {
        return "\n";
    }
    if ch == "t" {
        return "\t";
    }
    if ch == "r" {
        return "\r";
    }
    if ch == "b" {
        return "\b";
    }
    if ch == "f" {
        return "\f";
    }
    if ch == "\"" {
        return "\"";
    }
    if ch == "\\" {
        return "\\";
    }
    if ch == "/" {
        return "/";
    }
    # \uXXXX and any unrecognised escape: pass through as the raw
    # backslash + char. Callers parsing the post-stream raw JSON will
    # still see correct content.
    return "\\" + ch;
}
