"""Smoke test: smart default for streaming_text_field.

When no `streaming_text_field` is configured, byLLM resolves it from the
response_format schema:
  1. Use `answer` if the schema has a top-level string property by that name
  2. Otherwise pick the first top-level string property in the schema
  3. Fall back to "answer" if introspection fails

This test verifies (2): a response shape that uses `summary` instead of
`answer` should still stream tokens correctly without any explicit config.
"""

import from byllm.lib { Model }


obj BriefingReply {
    has summary: str;
    has key_points: list[str];
}


glob llm = Model(
    model_name="mockllm",
    config={
        "outputs": [
            '{"summary": "rate limit raised to 750 req/min for prod", '
            '"key_points": ["750 req/min cap", "rolled out 2026-Q1"]}'
        ]
    },
);


def brief(topic: str) -> BriefingReply by llm(stream=True);


with entry {
    print("=== smart default streaming_text_field ===\n");
    result = brief("rate limit policy");

    streamed = "";
    for token in result {
        streamed += token;
    }
    print(f"Streamed text: {streamed!r}");

    parsed = result.parsed;
    print(f"parsed.summary:      {parsed.summary!r}");
    print(f"parsed.key_points:   {parsed.key_points!r}");

    expected_text = "rate limit raised to 750 req/min for prod";
    expected_points = ["750 req/min cap", "rolled out 2026-Q1"];

    failures = 0;
    if streamed == expected_text {
        print("PASS  resolver picked 'summary' (first str field, no 'answer' present)");
    } else {
        failures += 1;
        print(f"FAIL  streamed mismatch. expected={expected_text!r} got={streamed!r}");
    }
    if parsed.summary == expected_text {
        print("PASS  parsed.summary matches");
    } else {
        failures += 1;
        print(f"FAIL  parsed.summary mismatch: {parsed.summary!r}");
    }
    if parsed.key_points == expected_points {
        print("PASS  parsed.key_points matches");
    } else {
        failures += 1;
        print(f"FAIL  parsed.key_points mismatch: {parsed.key_points!r}");
    }

    if failures > 0 {
        raise AssertionError(f"{failures} assertion(s) failed");
    }
    print("\nAll assertions passed.");
}
