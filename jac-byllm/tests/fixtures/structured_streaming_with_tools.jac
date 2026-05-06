"""End-to-end smoke test for typed+streaming WITH tools via MockLLM.

The harder case: the agent uses tools (the model calls read_file once),
then produces a final structured response. The streaming + structured
return + tools combination is what the research vision requires —
"answer + citations in the same call" alongside genuine tool use.

Validates:
- The agent can declare `def f(...) -> SomeObj by llm(tools=[...], stream=True)`
- The ReAct loop runs (tool call executes)
- finish_tool fires; _stream_final_answer detects typed mode, clears
  tools, and streams the structured response via response_format
- StructuredStream wraps the result; iteration yields answer tokens;
  `.parsed` returns the full typed object with citations
"""

import from byllm.lib { Model, MockToolCall }


# Fake tool the agent will use.
def fetch_doc(doc_id: str) -> str {
    return f"Contents of doc {doc_id}: rate_limit=750, timeout=800ms";
}


# Required by MockLLM's outputs script — the model "calls" finish_tool
# at the end of the ReAct loop, and the function signature has to match.
def finish_tool(answer: str, citations: list[str]) -> str {
    return answer;
}


obj StructuredAnswer {
    has answer: str;
    has citations: list[str];
}


glob llm = Model(
    model_name="mockllm",
    config={
        "outputs": [
            # Iteration 1: model decides to call fetch_doc
            MockToolCall(tool=fetch_doc, args={"doc_id": "tr_2"}),
            # Iteration 2: model decides it's done; calls finish_tool
            MockToolCall(
                tool=finish_tool,
                args={
                    "answer": "the rate limit is 750 req/min",
                    "citations": ["tr_2"]
                }
            ),
            # _stream_final_answer's streaming follow-up: typed mode strips
            # tools and re-prompts, MockLLM emits the structured JSON which
            # gets extracted by StreamingFieldExtractor.
            '{"answer": "the rate limit is 750 req/min", '
            '"citations": ["tr_2"]}'
        ]
    },
);


"""Investigate and produce a structured answer with citations."""
def investigate(question: str) -> StructuredAnswer by llm(
    tools=[fetch_doc], stream=True
);


with entry {
    print("=== typed+stream+tools via MockLLM ===\n");
    result = investigate("what is the rate limit?");
    print(f"type(result) = {type(result).__name__}");

    streamed = "";
    for token in result {
        streamed += token;
    }
    print(f"\nStreamed text: {streamed!r}");

    parsed = result.parsed;
    print(f"parsed type:        {type(parsed).__name__}");
    print(f"parsed.answer:      {parsed.answer!r}");
    print(f"parsed.citations:   {parsed.citations!r}");

    expected_text = "the rate limit is 750 req/min";
    expected_cites = ["tr_2"];

    failures = 0;
    if streamed == expected_text {
        print("PASS  streamed text matches");
    } else {
        failures += 1;
        print(f"FAIL  streamed mismatch. expected={expected_text!r} got={streamed!r}");
    }
    if type(parsed).__name__ == "StructuredAnswer" {
        print("PASS  parsed is StructuredAnswer instance");
    } else {
        failures += 1;
        print(f"FAIL  parsed type wrong: {type(parsed).__name__}");
    }
    if parsed.answer == expected_text {
        print("PASS  parsed.answer matches");
    } else {
        failures += 1;
        print(f"FAIL  parsed.answer mismatch: {parsed.answer!r}");
    }
    if parsed.citations == expected_cites {
        print("PASS  parsed.citations matches");
    } else {
        failures += 1;
        print(f"FAIL  parsed.citations mismatch: {parsed.citations!r}");
    }

    if failures > 0 {
        raise AssertionError(f"{failures} assertion(s) failed");
    }
    print("\nAll assertions passed.");
}
