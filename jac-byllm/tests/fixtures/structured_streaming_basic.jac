"""End-to-end smoke test for typed+streaming via MockLLM.

Declares a `by llm()` function with a typed return AND `stream=True`.
MockLLM is configured to emit a JSON object matching the schema; its
streaming dispatcher chunks the JSON and feeds chunks through the
StreamingFieldExtractor wired into both BaseLLM.dispatch_streaming and
MockLLM.dispatch_streaming.

Validates:
- The returned object is a StructuredStream (iterable + .parsed)
- The streamed tokens are *only* the characters of the designated text
  field ("answer"), not raw JSON syntax.
- After streaming completes, the parsed AgentReply object is available
  via `.parsed`, with all non-streamed fields populated.
"""

import from byllm.lib { Model }


obj AgentReply {
    has answer: str;
    has citations: list[str];
}


glob llm = Model(
    model_name="mockllm",
    config={
        "outputs": [
            '{"answer": "the rate limit is 750 req/min", '
            '"citations": ["tr_2", "tr_5"]}'
        ]
    },
);


"""Answer a question and return both the streamed answer text and the
list of cited tool-result IDs."""
def my_agent(question: str) -> AgentReply by llm(stream=True);


with entry {
    print("=== structured streaming via MockLLM ===\n");
    result = my_agent("what is the rate limit?");
    print(f"type(result) = {type(result).__name__}");

    streamed = "";
    for token in result {
        streamed += token;
    }
    print(f"\nStreamed text: {streamed!r}");

    # After iteration completes, the parsed object should be available.
    parsed = result.parsed;
    print(f"parsed type:        {type(parsed).__name__}");
    print(f"parsed.answer:      {parsed.answer!r}");
    print(f"parsed.citations:   {parsed.citations!r}");

    expected_text = "the rate limit is 750 req/min";
    expected_cites = ["tr_2", "tr_5"];

    failures = 0;
    if streamed == expected_text {
        print("PASS  streamed text matches");
    } else {
        failures += 1;
        print(f"FAIL  streamed mismatch. expected={expected_text!r} got={streamed!r}");
    }
    if type(parsed).__name__ == "AgentReply" {
        print("PASS  parsed is an AgentReply instance");
    } else {
        failures += 1;
        print(f"FAIL  parsed type wrong: {type(parsed).__name__}");
    }
    if parsed.answer == expected_text {
        print("PASS  parsed.answer matches");
    } else {
        failures += 1;
        print(f"FAIL  parsed.answer mismatch. got={parsed.answer!r}");
    }
    if parsed.citations == expected_cites {
        print("PASS  parsed.citations matches");
    } else {
        failures += 1;
        print(f"FAIL  parsed.citations mismatch. got={parsed.citations!r}");
    }

    if failures > 0 {
        raise AssertionError(f"{failures} assertion(s) failed");
    }
    print("\nAll assertions passed.");
}
