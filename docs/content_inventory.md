# Content Inventory

The guide has two complementary entry points:

- **55 curriculum modules** (`01_foundations/` through
  `55_multimodal_retrieval/`). The 50 notebooks cover the foundational and
  core curriculum; application modules 51–55 currently provide runnable
  scripts and focused implementation guides.
- **2 operational guides**: [CRCR & Downstream CI](../40_crcr_downstream_ci/)
  and [Targeted Test Selection](../41_targeted_tests/). These use historical
  directory prefixes and are intentionally outside the numbered curriculum.

The practical reference collection contains 28 cards in
[`docs/reference_cards/`](reference_cards/). It is organized by a reader's
immediate problem rather than the curriculum order.

## Numbering

The canonical curriculum number is the number shown in the course table. The
notebook index intentionally covers the stable 01–50 sequence. When a directory
prefix could be ambiguous, link to the directory name rather than referring to
the prefix alone. For example,
`40_image_classifier/` is curriculum Module 40, while
`40_crcr_downstream_ci/` is an operational guide.

## Maintaining the inventory

Run the check from the repository root whenever a module, notebook, reference
card, or example is added:

```bash
python tools/check_content_inventory.py
```

The check reports the current inventory and fails if the curriculum, stable
notebook sequence, reference cards, or examples are incomplete. Update this
document and the README summary when the content shape changes intentionally.
