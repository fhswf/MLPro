---
name: Release-Checklist
about: List of all necessary steps before, during and after a release.
title: Release vX.Y.Z
labels: admin
assignees: detlefarend, steveyuwono

---

Release Checklist
-------------------

**1 Preparation**
- [ ] 1.1 Inform the team on slack to stop merging to main
- [ ] 1.2 Create a new version number in project custom field 'version'
- [ ] 1.3 Assign new version number to all related issues and remove label 'next release'
- [ ] 1.4 Updates in branch main 
    - [ ] 1.4.1 Update version in ./setup.py
    - [ ] 1.4.2 Update version in ./src/setup.py
    - [ ] 1.4.3 Update version in ./src/conda/meta.yaml
    - [ ] 1.4.4 Update version in ./doc/rtd/conf.py
    - [ ] 1.4.5 Build and check RTD documentation
        - [ ] 1.4.5.1 All class diagrams there?
        - [ ] 1.4.5.2 All auto-generated code descriptions there?
        - [ ] 1.4.5.3 Logo there?
    - [ ] 1.4.6 Commit all changes and observe the action log

**2 Release**
- [ ] 2.1 Create a new [release](https://github.com/fhswf/MLPro/releases)
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
