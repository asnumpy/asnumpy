# Changelog

All notable changes to asnumpy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

## [Unreleased]

## [0.3.0] - 2026-05-18

### Added
- Adopt setuptools-scm for automatic git-tag-based versioning (single source of truth)
- Adopt towncrier for fragment-based CHANGELOG management
- Inject `SKBUILD_PROJECT_VERSION` into CMake `project()` for C++ side version awareness
- Create initial CHANGELOG.md

### Changed
- Reorganize repository into src-layout (`src/asnumpy/`)
- Split native C++ code into separate source trees (`csrc/`, `bindings/`, `include/`)
- Merge PR #112 restructured codebase into master

### Removed
- Delete empty `bindings/python/bind_version.cpp` shell
- Remove unused `_core.version` submodule from pybind11 bindings
