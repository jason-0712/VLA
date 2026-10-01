# Experiment Registry

Append one row to `registry.csv` for every launched run, including failed and aborted runs. Use `status` values such as `planned`, `running`, `completed`, `aborted`, or `invalid`.

Store large outputs outside Git. Paths in the registry must point to immutable run directories. If a field is not yet available, leave it empty rather than inventing a value.

Recommended ID format: `E###-short-description`, for example `E001-clean-full-ft`.
