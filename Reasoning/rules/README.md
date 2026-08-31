# rules

This folder contains the logic rule files loaded by the reasoning layer.

## Structure

```text
rules/
├── rules_MATRES.txt
├── rules_MATRES_v2.txt
├── rules_TBD.txt
├── rules_TBDv2.txt
├── rules_TDD.txt
└── rules_i2b2.txt
```

## Installation

There are no separate dependencies. These files are read directly by `Reasoning/tre_reasoner`.

## Usage

Select the rule file in `Reasoning/script/config/*.conf` via the `rule_file` field.

## Notes

- These are inputs to the reasoning engine, not code.
- If you rename or add rule files, update the corresponding config.
