# Known Issues — trajectory_forge

> Recorded after commit `70fe0c1`. All issues below are **unresolved** in the current codebase.
> Fix these before running large-scale trajectory generation.

---

## ~~CRITICAL~~ RESOLVED

### ~~1. Wrong parameter ranges → catastrophic image blackening~~ FIXED

**Status**: **RESOLVED** — image_engine SCALES updated, temperature range fixed.

**Root cause was**: `image_engine/rapidraw_basic_color/basic.py` had `SCALES["exposure"] = 0.8`
and `SCALES["brightness"] = 0.8`（original RapidRAW ±5 UI range），but trajectory_forge tools
used ±100 UI range. UI `exposure = -14` → `2^(-14/0.8) = 2^(-17.5)` → near-zero image.

**Fix applied**:
1. `image_engine/rapidraw_basic_color/basic.py`: SCALES `exposure` 0.8→16.0, `brightness` 0.8→16.0
   (UI ±100 / 16 = ±6.25 internal, same as original ±5 / 0.8 = ±6.25)
2. `tools/tool_registry.py` + `config/tools.json`: `temperature` range (-500,500) → (-100,100)
3. Tool parameter ranges for exposure/brightness remain (-100, 100), now correct with SCALE=16.0

---
### 3. `:.1f` format crash when `delta_e` key is missing

**File**: `agents/prompts.py`, line ~117

**Problem**:
```python
de_str = f" → DeltaE={q.get('delta_e', '?'):.1f}" if q else ""
```
If `q = {}` (non-empty dict but missing `delta_e`), `q.get('delta_e', '?')` returns the string `'?'`, and `:.1f` raises `TypeError: unsupported format character`.

**Fix**:
```python
de_str = f" → DeltaE={q['delta_e']:.1f}" if q and "delta_e" in q else ""
```

---

### 4. `mllm_agent.py` retries non-retryable 4xx errors

**File**: `agents/mllm_agent.py`, `call()` method

**Problem**: The `except Exception` clause sleeps and retries on every error, including 400/401/422 which are permanent. Each retry wastes `retry_delay` seconds (default 5 s) with no hope of success.

**Fix**: Detect HTTP status code and skip retry for 4xx:
```python
except Exception as e:
    status = getattr(e, "status_code", None)
    if status is not None and 400 <= status < 500:
        raise  # non-retryable
    if attempt < max_retries - 1:
        logger.warning(...)
        time.sleep(retry_delay)
    else:
        raise
```

---

## MINOR

### 5. `run_generate.py` crashes if `thumbnail_size` is an int in config

**File**: `run_generate.py`

**Problem**:
```python
thumbnail_size=int(gen_cfg.get("thumbnail_size", [512, 512])[0])
```
If `pipeline.yaml` has `thumbnail_size: 512` (an integer, not a list), `[0]` subscript raises `TypeError`.

**Fix**:
```python
_ts = gen_cfg.get("thumbnail_size", 512)
thumbnail_size = int(_ts[0] if isinstance(_ts, list) else _ts)
```

---

### 6. `trajectory_generator.py` saves `step_0_input.jpg` twice

**File**: `pipeline/trajectory_generator.py`

**Problem**: Before the loop (line 124) the source image is saved as `step_0_input.jpg`. Then inside turn 0 of the loop (line 207–209), `current_img` (which is still `src_img`) is saved again to the same path, overwriting the first save.

**Fix**: Remove the pre-loop save (lines 122–124):
```python
# DELETE these lines:
if step_dir and save_images:
    _save_step_image(src_img, step_dir, "step_0_input", image_quality)
```
The loop already saves `step_{turn}_input` at the start of each turn.

---

### 7. Unused `asdict` import in `trajectory_generator.py`

**File**: `pipeline/trajectory_generator.py`, line 13

```python
from dataclasses import asdict  # never used in this file
```

Remove this import.

---

### 8. Misleading docstring in `image_engine_adapter.py`

**File**: `tools/image_engine_adapter.py`, `merge_tool_call()` docstring

**Problem**: The docstring says *"Parameters from the new tool call are **added** (not replaced)"*, but the implementation **overwrites** (replaces) every parameter in the affected group unconditionally.

**Fix**: Update docstring to say "overwrite" or "replace":
```
Parameters from the new tool call overwrite the corresponding fields in
accumulated for the tool's parameter group.
```

---

## Summary table

| #   | Severity     | File                                          | Issue                                      |
| --- | ------------ | --------------------------------------------- | ------------------------------------------ |
| 1   | **CRITICAL** | `tools/tool_registry.py`, `config/tools.json` | Wrong parameter ranges → image blackening  |
| 2   | **MEDIUM**   | `agents/prompts.py`                           | Delta stats unit ambiguity misleads model  |
| 3   | **MEDIUM**   | `agents/prompts.py`                           | `:.1f` crash on missing `delta_e` key      |
| 4   | **MEDIUM**   | `agents/mllm_agent.py`                        | Retries non-retryable 4xx errors           |
| 5   | **MINOR**    | `run_generate.py`                             | `thumbnail_size` int/list type crash       |
| 6   | **MINOR**    | `pipeline/trajectory_generator.py`            | Duplicate `step_0_input` save              |
| 7   | **MINOR**    | `pipeline/trajectory_generator.py`            | Unused `asdict` import                     |
| 8   | **MINOR**    | `tools/image_engine_adapter.py`               | Misleading "added" vs "replaced" docstring |
