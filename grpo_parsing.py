

import json
import re


def _try_parse_json_object(text: str, start: int) -> dict | None:

    depth = 0
    in_str = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start:idx + 1]
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None

    return None


def _strip_common_wrappers(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("<|im_end|>", "")
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    return cleaned


def _try_parse_json_object_with_completion(text: str, start: int) -> dict | None:

    depth = 0
    in_str = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start:idx + 1]
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None

    if depth <= 0:
        return None

    chunk = text[start:].rstrip() + ("}" * depth)
    try:
        obj = json.loads(chunk)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def extract_json_object(text: str) -> dict | None:

    if not text:
        return None

    start = text.find("{")
    while start != -1:
        obj = _try_parse_json_object(text, start)
        if obj is not None:
            return obj
        start = text.find("{", start + 1)

    return None


def extract_json_object_loose(text: str) -> dict | None:

    if not text:
        return None
    cleaned = _strip_common_wrappers(text)
    start = cleaned.find("{")
    while start != -1:
        obj = _try_parse_json_object_with_completion(cleaned, start)
        if obj is not None:
            return obj
        start = cleaned.find("{", start + 1)
    return None


def parse_final_answer(text: str) -> str | None:

    if not text:
        return None
    cleaned = _strip_common_wrappers(text)
    obj = extract_json_object(cleaned)
    if not obj:
        obj = extract_json_object_loose(cleaned)
    if not obj:
        return None

    if "name" in obj and "arguments" in obj:
        return None
    if "tool_calls" in obj:
        return None
    if obj.get("type") in {"tool_call", "tool"}:
        return None

    if "answer" in obj:
        return str(obj["answer"]).strip()
    if "final_answer" in obj:
        return str(obj["final_answer"]).strip()
    if obj.get("type") in {"final", "final_answer"} and "content" in obj:
        return str(obj["content"]).strip()

    return None


def parse_answer(text: str) -> str:

    if not text:
        return ""

    text = text.strip()

    json_answer = parse_final_answer(text)
    if json_answer:
        return json_answer

    if re.search(r"(query\s*plan|计划|plan:)", text, re.IGNORECASE):
        return ""


    if len(text) < 50:
        return text


    option_match = re.search(r"\b([A-E])\b", text)
    if option_match:
        return option_match.group(1)


    region_patterns = [
        r'(右上角|左上角|右下角|左下角|中心|上半部|下半部|左半部|右半部)',
        r'(右上|左上|右下|左下)',
    ]
    for pattern in region_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)


    num_patterns = [
        r'(\d+\.?\d*)\s*m/s',
        r'(\d+\.?\d*)\s*米/秒',
        r'(\d+\.?\d*)%',
        r'约?\s*(\d+\.?\d*)\s*个?网格',
        r'答案[是为：:]\s*(\d+\.?\d*)',
        r'^(\d+\.?\d*)$',  
    ]
    for pattern in num_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)


    trend_patterns = [
        r'(持续增加|持续减少|波动变化|保持稳定|先增后减|先减后增)',
    ]
    for pattern in trend_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)


    yesno_patterns = [
        r'(存在涡旋|不存在涡旋|存在回流|不存在回流)',
        r'^(是|否|有|无)',
    ]
    for pattern in yesno_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)


    answer_patterns = [
        r'答案[是为：:]\s*(.+?)(?:\n|$|。)',
        r'结论[是为：:]\s*(.+?)(?:\n|$|。)',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()[:50]


    first_line = text.split('\n')[0].strip()
    return first_line[:50] if len(first_line) > 50 else first_line


def parse_tool_calls(text: str) -> list[dict]:


    if not text:
        return []


    tag_matches = re.findall(r"<tool_call>.*?</tool_call>", text, re.DOTALL)
    candidates = tag_matches if tag_matches else [text]
    tool_calls: list[dict] = []


    for cand in candidates:
        obj = extract_json_object(cand)  
        if not obj:
            continue
        normalized = _normalize_tool_calls(obj)  
        if normalized:
            tool_calls.extend(normalized)

    return tool_calls


def _normalize_query_plan_obj(obj: dict | None) -> dict | None:
    if not obj or not isinstance(obj, dict):
        return None

    raw_tools = None
    if "tools" in obj:
        raw_tools = obj.get("tools")
    elif "steps" in obj:
        raw_tools = obj.get("steps")
    elif "plan" in obj:
        raw_tools = obj.get("plan")
    elif "name" in obj and any(k in obj for k in ("args", "arguments", "parameters")):
        raw_tools = [obj]

    if not isinstance(raw_tools, list) or not raw_tools:
        return None

    normalized_tools = []
    for item in raw_tools:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("tool")
        args = item.get("args") or item.get("arguments") or item.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        if not name:
            continue
        normalized_tools.append({"name": str(name), "args": args})

    if not normalized_tools:
        return None

    return {"tools": normalized_tools}


def parse_query_plan(text: str, *, allow_incomplete: bool = False) -> dict | None:


    if not text:
        return None
    cleaned = _strip_common_wrappers(text)
    strict_obj = extract_json_object(cleaned)
    plan = _normalize_query_plan_obj(strict_obj)
    if plan or not allow_incomplete:
        return plan
    loose_obj = extract_json_object_loose(cleaned)
    return _normalize_query_plan_obj(loose_obj)


def parse_tool_call(text: str) -> dict:

    tool_calls = parse_tool_calls(text)
    return tool_calls[0] if tool_calls else {}


def _normalize_tool_calls(obj: dict) -> list[dict]:
    if not isinstance(obj, dict):
        return []

    if "tool_calls" in obj and isinstance(obj["tool_calls"], list):
        tool_calls = []
        for item in obj["tool_calls"]:
            tool_call = _normalize_tool_call(item)
            if tool_call:
                tool_calls.append(tool_call)
        if tool_calls:
            return tool_calls

    tool_call = _normalize_tool_call(obj)
    return [tool_call] if tool_call else []


def _normalize_tool_call(obj: dict) -> dict:
    if not isinstance(obj, dict):
        return {}

    if "tool_calls" in obj and isinstance(obj["tool_calls"], list) and obj["tool_calls"]:
        obj = obj["tool_calls"][0]
    if "tool_call" in obj and isinstance(obj["tool_call"], dict):
        obj = obj["tool_call"]
    if "tool" in obj and isinstance(obj["tool"], dict):
        obj = obj["tool"]
    if "function" in obj and isinstance(obj["function"], dict):
        obj = obj["function"]

    name = obj.get("name") or obj.get("tool_name")
    args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}

    if not isinstance(args, dict):
        args = {}

    if not name:
        return {}

    return {"name": str(name), "arguments": args}
