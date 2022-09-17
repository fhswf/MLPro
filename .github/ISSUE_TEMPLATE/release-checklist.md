---
name: Release-Checklist
about: List of all necessary steps before, during and after a release.
title: Release vX.Y.Z
labels: admin
assignees: detlefarend

---

Release Checklist
-----------------

**1 Preparation**
- [ ] 1.1 Inform the team on slack to stop merging to main
- [ ] 1.2 Rename label "next release" to "vX.Y.Z"
- [ ] 1.3 Create new label "next release"
- [ ] 1.4 Relabel all open issues of release "vX.Y.Z" to "next release"
- [ ] 1.5 Update version in ./setup.py
- [ ] 1.6 Update version in ./src/setup.py
- [ ] 1.7 Update version in ./src/conda/meta.yaml
- [ ] 1.8 Check documentation on RTD
  - [ ] 1.8.1 All class diagrams there?
  - [ ] 1.8.2 All auto-generated code descriptions there?
  - [ ] 1.8.3 Logo there?
- [ ] 1.9 Check action log for errors


**2 Release**
- [ ] 2.1 Create a new release
- [ ] 2.2 Generate/beautify release notes
- [ ] 2.3 Commit new release


**3 Postprocessing**
- [ ] 3.1 Check documentation on RTD
  - [ ] 3.1.1 All class diagrams there?
  - [ ] 3.1.2 All auto-generated code descriptions there?
  - [ ] 3.1.3 Logo there?
- [ ] 3.2 Check entry on PyPI
- [ ] 3.3 Check entry on Anaconda
- [ ] 3.4 Inform the team on slack
- [ ] 3.5 Significant progress? Write a project log entry on ResearchGate.
