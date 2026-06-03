# Test suite

The suite is split into three practical levels:

- `unit` — pure project logic and small Torch modules in isolation;
- `integration` — several XTrim components working together while external tools are mocked;
- `smoke` — a fast pipeline run from config parsing to history/Pareto artifacts without real YOLO, ADB, or NCNN binaries.

Run everything:

```bash
pytest
```

Run by level:

```bash
pytest -m unit
pytest -m integration
pytest -m smoke
```

Run with coverage:

```bash
pytest --cov=xtrim --cov-report=term-missing
```

Tests intentionally do not require a connected Android device, Ultralytics weights, ONNX Runtime, or NCNN binaries. Those are external integration points and are replaced with deterministic fakes/mocks here.
