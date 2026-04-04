from pathlib import Path


def read_source_file(file_path: Path) -> str:
    try:
        raw_data = file_path.read_bytes()
    except FileNotFoundError:
        return ""

    if not raw_data:
        return ""

    for enc in ["utf-8-sig", "utf-8", "utf-16", "utf-32", "shift_jis", "cp1252", "latin-1"]:
        try:
            return raw_data.decode(enc, errors="strict")
        except (UnicodeDecodeError, LookupError):
            continue

    return raw_data.decode("utf-8", errors="replace")
