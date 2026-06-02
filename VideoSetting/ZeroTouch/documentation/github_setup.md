# GitHub Setup & Large File Management

This document explains why the initial GitHub push failed and how it was resolved by excluding large, unnecessary files.

## The Problem: GH001 Error
GitHub has a strict file size limit of **100MB** per file and a total recommended repository size. The push failed because the `venv/` (virtual environment) and `dist/` (build output) folders were included in the commit history. 

Key culprits:
- `dist/ZeroTouch/_internal/tensorflow/python/_pywrap_tensorflow_common.dll` (~1GB)
- `venv/Lib/site-packages/clang/native/libclang.dll` (~80MB)

## The Solution: `.gitignore`
A `.gitignore` file was added to the project root to ensure that these files are never tracked by Git. Virtual environments and build artifacts should be generated locally and not stored in the repository.

## How to Clean the Repository
If large files are already committed, they remain in the hidden `.git` history even after deleting them. To fix this on a fresh repository, follow these steps:

1. **Create/Verify `.gitignore`**.
2. **Remove the existing `.git` folder**:
   ```powershell
   Remove-Item -Recurse -Force .git
   ```
3. **Re-initialize and Commit**:
   ```powershell
   git init
   git add .
   git commit -m "Initial commit (clean)"
   git branch -M main
   git remote add origin https://github.com/Igndv/ZeroTouch.git
   git push -u origin main --force
   ```

## Best Practices
- **Always use a venv**: But never commit it.
- **Ignore build folders**: `dist/` and `build/` are for deployment, not source control.
- **Git LFS**: If you truly need to store files >100MB, use [Git Large File Storage](https://git-lfs.github.com).
