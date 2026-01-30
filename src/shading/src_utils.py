import codecs
import inspect
import os


def read_source_file(relative_filename):
    abs_path = os.path.abspath((inspect.stack()[1])[1])
    parent_directory = os.path.dirname(abs_path)
    file_path = os.path.join(parent_directory, relative_filename)

    with open(file_path, "rb") as f:
        raw_data = f.read()

    if not raw_data:
        return ""

    if raw_data.startswith(codecs.BOM_UTF32_BE):
        return raw_data[len(codecs.BOM_UTF32_BE):].decode("utf-32-be")
    if raw_data.startswith(codecs.BOM_UTF32_LE):
        return raw_data[len(codecs.BOM_UTF32_LE):].decode("utf-32-le")
    if raw_data.startswith(codecs.BOM_UTF16_BE):
        return raw_data[len(codecs.BOM_UTF16_BE):].decode("utf-16-be")
    if raw_data.startswith(codecs.BOM_UTF16_LE):
        return raw_data[len(codecs.BOM_UTF16_LE):].decode("utf-16-le")
    if raw_data.startswith(codecs.BOM_UTF8):
        return raw_data[len(codecs.BOM_UTF8):].decode("utf-8")

    encodings_to_try = ["utf-8", "shift_jis", "cp1252", "latin-1"]

    for enc in encodings_to_try:
        try:
            return raw_data.decode(enc, errors="strict")
        except (UnicodeDecodeError, LookupError):
            continue

    return raw_data.decode("utf-8", errors="replace")
