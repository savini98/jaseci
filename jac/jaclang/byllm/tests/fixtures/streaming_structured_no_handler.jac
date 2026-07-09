"""Structured streaming without a handler returns a StructuredStream.

Streaming a non-`str` return without a stream_handler used to be rejected
up front; it is now supported: the call returns a StructuredStream whose
tokens are the designated text field's characters and whose `.parsed`
carries the fully-typed instance once the stream is consumed. The
handler-driven path (see streaming_visit_by) is unchanged: with a
stream_handler the raw structured tokens go to the handler instead.
"""
import from jaclang.byllm.lib { MockLLM }
import from jaclang.byllm.streaming_parser { StructuredStream }

obj Pick {
    has answer: str = "";
    has choices: list[int] = [];
}

glob llm: MockLLM = MockLLM(
         model_name="mockllm", config={"outputs": ['{"answer": "b", "choices": [1]}']}
     );

def pick(options: str) -> Pick by llm(stream=True);

with entry {
    result = pick("0=a, 1=b");
    assert isinstance(result, StructuredStream) , (
        f"expected StructuredStream, got {type(result).__name__}"
    );
    streamed = "".join([tok for tok in result]);
    assert streamed == "b" , f"unexpected streamed text: {streamed!r}";
    parsed = result.parsed;
    assert isinstance(parsed, Pick) , f"parsed is {type(parsed).__name__}";
    assert parsed.choices == [1] , f"parsed.choices = {parsed.choices}";
    print("STREAMING_STRUCTURED_NO_HANDLER_PASS");
}
