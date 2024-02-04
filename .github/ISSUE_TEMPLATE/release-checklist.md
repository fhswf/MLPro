---
name: Release-Checklist
about: List of all necessary steps before, during and after a release.
title: Release vX.Y.Z
labels: admin
assignees: ''

---

Release Checklist
-------------------

**1 Preparation**
- [ ] 1.1 Inform the team on slack to stop merging to main
- [ ] 1.2 Rename label "next release" to "vX.Y.Z"
- [ ] 1.3 Create new label "next release"
- [ ] 1.4 Relabel all open issues of release "vX.Y.Z" to "next release"
- [ ] 1.5 Update version in ./setup.py
- [ ] 1.6 Update version in ./src/setup.py
- [ ] 1.7 Update version in ./src/conda/meta.yaml
- [ ] 1.8 Update version in ./doc/rtd/conf.py
- [ ] 1.9 Build and check RTD documentation
  - [ ] 1.9.1 All class diagrams there?
  - [ ] 1.9.2 All auto-generated code descriptions there?
  - [ ] 1.9.3 Logo there?
- [ ] 1.10 Check action log for errors


**2 Release**
- [ ] 2.1 Create a new release
- [ ] 2.2 Generate/complete release notes
- [ ] 2.3 Commit new release and observe the action log
- [ ] 2.4 Activate new release in [ReadTheDocs](https://readthedocs.org) as user mlpro-admin


**3 Postprocessing**
- [ ] 3.1 Check the [RTD documentation](https://mlpro.readthedocs.io)
  - [ ] 3.1.1 All class diagrams there?
  - [ ] 3.1.2 All auto-generated code descriptions there?
  - [ ] 3.1.3 Logo there?
- [ ] 3.2 Check [MLPro in PyPI](https://pypi.org/project/mlpro/)
- [ ] 3.3 Check [MLPro in Anaconda](https://anaconda.org/mlpro/mlpro/)
- [ ] 3.4 Update all open branches from main
- [ ] 3.5 Inform the team on slack
