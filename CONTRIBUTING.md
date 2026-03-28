# Contributing

## Local checks

Run the lightweight test suite before pushing:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Useful shortcuts:

```bash
make test
make pipeline-dry-run
make docs-example
```

## Working with local artifacts

This repository is set up to keep large artifacts out of version control:

- `data/`
- `artifacts/`
- `checkpoints/`
- `exports/`
- `logs/`
- `.onnx_export_vendor/`

Keep generated outputs local. The tracked repository should mostly contain source, docs, tests, and small configuration files.

## Before the first GitHub push

The current working tree is clean for future commits, but the existing Git history still contains old large blobs. If this is going to a brand new GitHub repository, the safest upload path is:

1. create a fresh repository from the current working tree, or
2. rewrite history with `git filter-repo` before pushing

That avoids publishing the old 2.29 GiB packed history.

## Pages

The static site in `docs/` is ready for GitHub Pages. After the repository is on GitHub, set Pages to deploy from GitHub Actions and the included workflow will publish the site.
