"""Standalone smoke test for byllm.streaming_parser.

Exercises the StreamingFieldExtractor against canned chunk streams that
mimic what an OpenAI/Anthropic structured-output stream looks like:
JSON tokens arriving as fragments where the target string field's
characters are split across chunk boundaries.

Run via:  jac run byllm/streaming_parser_test.jac
"""

import json;

import from byllm.streaming_parser { StreamingFieldExtractor }


"""Run the extractor over a list of chunks and return the streamed text
plus the parsed JSON object."""
def run_extractor(chunks: list[str], field: str) -> dict {
    ex = StreamingFieldExtractor(field_name=field);
    streamed = "";
    for c in chunks {
        out = ex.feed(c);
        if out {
            streamed += out;
        }
    }
    return {
        "streamed": streamed,
        "is_done": ex.is_done(),
        "raw": ex.get_raw(),
        "parsed": json.loads(ex.get_raw())
    };
}


with entry {
    failures = 0;
    total = 0;

    # ----- Test 1: simple single-chunk JSON, target field at top level -----
    total += 1;
    r = run_extractor(
        ['{"answer": "hello world", "citations": ["tr_2"]}'],
        "answer"
    );
    if r["streamed"] == "hello world" and r["is_done"] and r["parsed"]["citations"] == ["tr_2"] {
        print("PASS  test 1 (single chunk)");
    } else {
        failures += 1;
        print(f"FAIL  test 1 (single chunk): {r}");
    }

    # ----- Test 2: chunked at every character -----
    total += 1;
    raw = '{"answer": "split me up", "citations": []}';
    chunks = [c for c in raw];
    r = run_extractor(chunks, "answer");
    if r["streamed"] == "split me up" and r["is_done"] {
        print("PASS  test 2 (one-char chunks)");
    } else {
        failures += 1;
        print(f"FAIL  test 2 (one-char chunks): streamed={r['streamed']!r} done={r['is_done']}");
    }

    # ----- Test 3: target key appears as substring of another key first -----
    total += 1;
    r = run_extractor(
        ['{"pre_answer": "decoy", "answer": "real value", "post": "tail"}'],
        "answer"
    );
    if r["streamed"] == "real value" and r["is_done"] {
        print("PASS  test 3 (key-substring decoy)");
    } else {
        failures += 1;
        print(f"FAIL  test 3 (key-substring decoy): streamed={r['streamed']!r}");
    }

    # ----- Test 4: target field name appears inside a NESTED object key -----
    total += 1;
    r = run_extractor(
        ['{"meta": {"answer": "nested decoy"}, "answer": "outer real"}'],
        "answer"
    );
    if r["streamed"] == "outer real" and r["is_done"] {
        print("PASS  test 4 (nested decoy)");
    } else {
        failures += 1;
        print(f"FAIL  test 4 (nested decoy): streamed={r['streamed']!r}");
    }

    # ----- Test 5: escape sequences inside the value -----
    total += 1;
    r = run_extractor(
        ['{"answer": "line one\\nline two with \\"quotes\\" and \\\\back"}'],
        "answer"
    );
    expected = "line one\nline two with \"quotes\" and \\back";
    if r["streamed"] == expected and r["is_done"] {
        print("PASS  test 5 (escapes)");
    } else {
        failures += 1;
        print(f"FAIL  test 5 (escapes): streamed={r['streamed']!r} expected={expected!r}");
    }

    # ----- Test 6: escape sequence split across chunk boundary -----
    total += 1;
    r = run_extractor(
        ['{"answer": "before\\', 'nafter"}'],
        "answer"
    );
    if r["streamed"] == "before\nafter" and r["is_done"] {
        print("PASS  test 6 (escape across chunk boundary)");
    } else {
        failures += 1;
        print(f"FAIL  test 6 (escape across chunk boundary): streamed={r['streamed']!r}");
    }

    # ----- Test 7: target key split across chunk boundary -----
    total += 1;
    r = run_extractor(
        ['{"ans', 'wer": "key cut"}'],
        "answer"
    );
    if r["streamed"] == "key cut" and r["is_done"] {
        print("PASS  test 7 (key across chunk boundary)");
    } else {
        failures += 1;
        print(f"FAIL  test 7 (key across chunk boundary): streamed={r['streamed']!r}");
    }

    # ----- Test 8: citations field comes BEFORE the answer field -----
    total += 1;
    r = run_extractor(
        ['{"citations": ["tr_2", "tr_5"], "answer": "after array"}'],
        "answer"
    );
    if r["streamed"] == "after array" and r["is_done"] and r["parsed"]["citations"] == ["tr_2", "tr_5"] {
        print("PASS  test 8 (array before target)");
    } else {
        failures += 1;
        print(f"FAIL  test 8 (array before target): {r}");
    }

    # ----- Test 9: opaque whitespace and formatting -----
    total += 1;
    r = run_extractor(
        ['{\n  "answer"  :  "spaced out"  ,\n  "n": 1\n}'],
        "answer"
    );
    if r["streamed"] == "spaced out" and r["is_done"] {
        print("PASS  test 9 (whitespace tolerant)");
    } else {
        failures += 1;
        print(f"FAIL  test 9 (whitespace tolerant): streamed={r['streamed']!r}");
    }

    # ----- Test 10: not-yet-done state after a partial chunk that
    # didn't reach the value -----
    total += 1;
    ex = StreamingFieldExtractor(field_name="answer");
    out_so_far = ex.feed('{"citations": ["tr_2"], "ans');
    if out_so_far == "" and not ex.is_done() {
        ex.feed('wer": "completed"}');
        if ex.is_done() {
            print("PASS  test 10 (delayed completion)");
        } else {
            failures += 1;
            print("FAIL  test 10 (delayed completion): not done after second feed");
        }
    } else {
        failures += 1;
        print(f"FAIL  test 10 (delayed completion): premature output={out_so_far!r}");
    }

    print(f"\n{total - failures}/{total} passed");
    if failures > 0 {
        raise AssertionError(f"{failures} test(s) failed");
    }
}
