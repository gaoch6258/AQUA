import json
from pathlib import Path

from datasets import Dataset


def resolve_dataset_paths(dataset_value: str) -> tuple[str, str]:


    dataset_path = Path(dataset_value)
    if not dataset_path.exists():
        candidate = Path(__file__).resolve().parent / "dataset" / dataset_value
        if candidate.exists():
            dataset_path = candidate

    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集不存在: {dataset_path}")

    qa_dir = dataset_path / "QA" / "cases"
    if not qa_dir.exists():
        raise FileNotFoundError(f"QA目录不存在: {qa_dir}")

    gaussians_dir = dataset_path / "gs"
    if not gaussians_dir.exists():
        legacy_dir = dataset_path / "hdf5_file"
        if legacy_dir.exists():
            gaussians_dir = legacy_dir
        else:
            raise FileNotFoundError(f"高斯目录不存在: {gaussians_dir}")

    return str(gaussians_dir), str(qa_dir)


def _matches_case_name(path: Path, case_name: str) -> bool:
    stem = path.stem
    return stem == case_name or stem.startswith(f"{case_name}_")


def load_qa_dataset(
    qa_dir: str,
    gaussians_dir: str,
    limit: int | None = None,
    case_names: set[str] | None = None,
) -> Dataset:


    qa_dir = Path(qa_dir)
    gaussians_dir = Path(gaussians_dir)

    if not qa_dir.exists():
        raise FileNotFoundError(f"QA目录不存在: {qa_dir}")
    if not gaussians_dir.exists():
        raise FileNotFoundError(f"高斯目录不存在: {gaussians_dir}")

    data = {
        "prompt": [],
        "gaussians_path": [],
        "qa_path": [],
        "case_name": [],
    }

    hdf5_files = list(gaussians_dir.rglob("*.hdf5"))
    hdf5_files += list(gaussians_dir.rglob("*.h5"))
    count = 0

    for json_file in sorted(qa_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            qa_obj = json.load(f)

        case_name = None
        if isinstance(qa_obj, dict):
            case_name = qa_obj.get("case")
        if not case_name:
            case_name = json_file.stem

        if case_names and case_name not in case_names:
            continue

        gaussians_file = None
        candidates = [h5 for h5 in hdf5_files if _matches_case_name(h5, case_name)]
        if candidates:
            gaussians_file = sorted(candidates)[0]
        if gaussians_file is None:
            print(f"警告: QA case 与高斯文件名不匹配，跳过 {json_file.name}")
            continue

        qa_list = qa_obj.get("qas") if isinstance(qa_obj, dict) else qa_obj
        qa_list = qa_list or []
        for item in qa_list:
            question = item.get("question") if isinstance(item, dict) else None
            if not question:
                continue
            data["prompt"].append(question)
            data["gaussians_path"].append(str(gaussians_file))
            data["qa_path"].append(str(json_file))
            data["case_name"].append(str(case_name))
            count += 1
            if limit and count >= limit:
                break

        if limit and count >= limit:
            break

    print(f"加载了 {len(data['prompt'])} 个问题")
    print(f"涉及 {len(set(data['gaussians_path']))} 个HDF5文件")
    return Dataset.from_dict(data)


def _build_prompt_to_paths(dataset: Dataset) -> dict[str, tuple[str, str]]:
    prompt_to_paths: dict[str, tuple[str, str]] = {}
    for i in range(len(dataset)):
        prompt = dataset[i]["prompt"]
        if prompt in prompt_to_paths:
            raise ValueError(f"检测到重复问题，无法唯一映射: {prompt[:80]}")
        prompt_to_paths[prompt] = (
            dataset[i]["gaussians_path"],
            dataset[i]["qa_path"],
        )
    return prompt_to_paths


def build_prompt_to_qa(dataset: Dataset) -> dict[str, dict]:


    qa_cache: dict[str, dict[str, dict]] = {}
    prompt_to_qa: dict[str, dict] = {}

    for i in range(len(dataset)):
        prompt = dataset[i]["prompt"]
        qa_path = dataset[i]["qa_path"]
        if qa_path not in qa_cache:
            with open(qa_path, "r", encoding="utf-8") as f:
                qa_obj = json.load(f)
            qa_list = qa_obj.get("qas") if isinstance(qa_obj, dict) else qa_obj
            qa_list = qa_list or []
            case_name = qa_obj.get("case") if isinstance(qa_obj, dict) else None
            if not case_name:
                case_name = Path(qa_path).stem
            qa_cache[qa_path] = {
                item.get("question"): {
                    **item,
                    "_meta_case": case_name,
                    "_meta_q_index": idx + 1,
                }
                for idx, item in enumerate(qa_list)
                if isinstance(item, dict) and item.get("question")
            }

        qa_item = qa_cache[qa_path].get(prompt)
        if qa_item:
            prompt_to_qa[prompt] = qa_item

    return prompt_to_qa
