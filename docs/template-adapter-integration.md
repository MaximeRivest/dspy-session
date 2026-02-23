# TemplateAdapter + dspy-session Integration

This guide shows how `dspy-session` and `dspy-template-adapter` fit together.

- `dspy-session` manages turn state and `dspy.History` accumulation
- `TemplateAdapter` controls exact prompt layout

## 1) Recommended baseline (`{"role": "history"}`)

```python
import dspy
from dspy_session import sessionify
from dspy_template_adapter import TemplateAdapter, Predict

class ChatSig(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

adapter = TemplateAdapter(
    messages=[
        {"role": "system", "content": "You are concise."},
        {"role": "history"},
        {"role": "user", "content": "{question}"},
    ],
    parse_mode="full_text",
)

chat = Predict(ChatSig, adapter=adapter)
session = sessionify(chat, max_turns=10)

# dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
# out1 = session(question="What is DSPy?")
# out2 = session(question="And sessionify?")
```

This is the most explicit and reliable placement strategy.

---

## 2) Inline history with `{history()}` (new helper)

`TemplateAdapter` now supports a built-in template function:

- `{history()}`
- `{history(style='json')}`
- `{history(style='yaml')}`
- `{history(style='xml')}`

### Everything in system

```python
class ChatSig(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

adapter = TemplateAdapter(
    messages=[
        {
            "role": "system",
            "content": "Conversation so far:\n{history(style='yaml')}",
        },
        {"role": "user", "content": "{question}"},
    ],
    parse_mode="full_text",
)
```

### Everything in user

```python
adapter = TemplateAdapter(
    messages=[
        {"role": "system", "content": "Answer directly."},
        {
            "role": "user",
            "content": "Conversation:\n{history(style='json')}\n\nQuestion:\n{question}",
        },
    ],
    parse_mode="full_text",
)
```

### Half-and-half (contrived)

```python
adapter = TemplateAdapter(
    messages=[
        {
            "role": "system",
            "content": "Long-term memory:\n{history(style='xml')}",
        },
        {
            "role": "user",
            "content": "Recent context:\n{history(style='yaml')}\n\nQ: {question}",
        },
    ],
    parse_mode="full_text",
)
```

These are intentionally unusual, but they demonstrate total prompt control.

---

## 3) Preview without LM calls

```python
preview_msgs = adapter.preview(
    signature=session.module.signature,
    inputs={
        "question": "Follow-up question",
        "history": session.session_history,
    },
)
```

Use this to verify exact prompt structure before runtime.

---

## 4) Notes

- If template contains `{"role": "history"}`, history is expanded at that message position.
- If template uses `{history(...)}`, history is consumed inline and not auto-injected again.
- Session remains adapter-agnostic; TemplateAdapter is purely prompt-shaping.
