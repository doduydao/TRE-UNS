# debug_model

This folder contains debug and error-analysis scripts for the reasoning model.

## Structure

```text
debug_model/
├── analyze_violations.py
├── final_report_generator.py
└── generate_error_csvs_detailed.py
```

## Installation

There are no separate dependencies. Use the same environment as `Reasoning/tre_reasoner`.

## Usage

The scripts here are for post-hoc analysis:

- collect prediction errors,
- extract logic violations,
- generate summary reports.

Run them directly with Python when you need to debug a specific experiment.
