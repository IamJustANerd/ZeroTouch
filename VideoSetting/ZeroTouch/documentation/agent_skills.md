# Agent Skills

This document describes the custom skills initialized for this project, located in `C:\Users\ignas\.agents\skills`.

## Available Skills

### 1. Caveman Mode (`/cavemen`)
- **Location**: `C:\Users\ignas\.agents\skills\caveman\SKILL.md`
- **Description**: An ultra-compressed communication mode that reduces token usage by approximately 75% while maintaining technical accuracy.
- **Usage**:
  - `/cavemen lite`: Professional but tight, no filler.
  - `/cavemen full` (Default): Classic caveman style, fragments okay.
  - `/cavemen ultra`: Maximum compression, abbreviated prose.
  - `/cavemen wenyan-lite|full|ultra`: Classical Chinese (Wenyan) styles.
- **Persistence**: Once activated, it remains active until "stop caveman" or "normal mode" is requested.

### 2. Find Skills (`/find_skills`)
- **Location**: `C:\Users\ignas\.agents\skills\find-skills\SKILL.md`
- **Description**: Helps discover and install new agent skills from the ecosystem.
- **Key Commands**:
  - `npx skills find [query]`: Search for skills.
  - `npx skills add <package>`: Install a skill.
  - `npx skills check`: Check for updates.
- **Browse**: [skills.sh](https://skills.sh/)

## Initialization Details
- **Path**: `C:\Users\ignas\.agents\skills`
- **Status**: Initialized and Ready.
- **Mapping**:
  - `/cavemen` -> Triggers `caveman` skill.
  - `/find_skills` -> Triggers `find-skills` skill.
