# Audit References

## Sionna RT Technical Report

The Sionna RT technical report is normative for the RT audit.

- arXiv abstract page: https://arxiv.org/abs/2504.21719
- arXiv PDF: https://arxiv.org/pdf/2504.21719
- arXiv TeX source: https://arxiv.org/e-print/2504.21719
- Hosted HTML: https://nvlabs.github.io/sionna/rt/tech-report/index.html

The arXiv page identifies the report as `arXiv:2504.21719`, title
`Sionna RT: Technical Report`, submitted on 2025-04-30 and last revised as v2 on
2025-11-21.

Do not commit downloaded PDF or source archives unless explicitly needed. For
local analysis, use:

```bash
mkdir -p audit/references/local
curl -L https://arxiv.org/pdf/2504.21719 -o audit/references/local/sionna-rt-tech-report.pdf
curl -L https://arxiv.org/e-print/2504.21719 -o audit/references/local/sionna-rt-tech-report-source.tar
```

Suggested extraction workflow:

```bash
mkdir -p audit/references/local/sionna-rt-tech-report-source
tar -xf audit/references/local/sionna-rt-tech-report-source.tar \
  -C audit/references/local/sionna-rt-tech-report-source
```

If arXiv serves a compressed tarball with a different extension, inspect it with
`file` and extract accordingly.

