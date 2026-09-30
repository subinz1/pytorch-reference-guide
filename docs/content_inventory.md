# Content Inventory

The guide has two complementary entry points:

- **50 curriculum modules** (`01_foundations/` through `50_torch_sparse/`)
  with one notebook per numbered module.
- **2 operational guides**: [CRCR & Downstream CI](../40_crcr_downstream_ci/)
  and [Targeted Test Selection](../41_targeted_tests/). These use historical
  directory prefixes and are intentionally outside the numbered curriculum.

The practical reference collection contains 28 cards in
[`docs/reference_cards/`](reference_cards/). It is organized by a reader's
immediate problem rather than the curriculum order.

## Numbering

The canonical curriculum number is the number shown in the course table and
notebook index. When a directory prefix could be ambiguous, link to the
directory name rather than referring to the prefix alone. For example,
`40_image_classifier/` is curriculum Module 40, while
`40_crcr_downstream_ci/` is an operational guide.

## Maintaining the inventory

Run the check from the repository root whenever a module, notebook, reference
card, or example is added:

```bash
python tools/check_content_inventory.py
```

The check reports the current inventory and fails if the stable curriculum or
notebook sequence is incomplete. Update this document and the README summary
when the content shape changes intentionally.
