# CLAUDE.md

This file provides guidance for AI assistants (Claude and others) working in this repository.

## Repository Overview

**Repository**: `loveoftheai/opc`
**Remote**: GitHub at `loveoftheai/opc`

This repository is in its initial state. Update this section as the project evolves with a description of what the project does, its purpose, and intended audience.

## Development Branch Conventions

- AI-driven work is done on feature branches following the pattern: `claude/<description>-<id>`
- Example: `claude/add-claude-documentation-vmVtr`
- Never push directly to `main` or `master` without explicit permission
- Always use `git push -u origin <branch-name>` when pushing a branch for the first time

## Git Workflow

1. Check out or create the designated feature branch before making changes
2. Make focused, atomic commits with clear, descriptive messages
3. Push to the remote branch when changes are complete
4. Do not create pull requests unless explicitly asked by the user

### Commit Message Style

- Use the imperative mood: "Add feature" not "Added feature"
- Keep the subject line under 72 characters
- Describe *what* and *why*, not *how*
- Reference issue numbers where applicable (e.g., `Fixes #42`)

## Working with GitHub

- Use GitHub MCP tools (`mcp__github__*`) for all GitHub interactions
- Repository scope is restricted to `loveoftheai/opc` — do not interact with other repositories
- Post comments sparingly — only when a reply is genuinely necessary
- Do not open, close, or merge PRs without explicit user instruction

## File Editing Conventions

- Prefer editing existing files over creating new ones
- Do not create documentation files (README, CLAUDE.md, etc.) unless explicitly requested
- Do not add comments, docstrings, or type annotations to code you did not change
- Do not introduce speculative abstractions or features beyond what is asked
- Avoid backwards-compatibility hacks; remove unused code cleanly

## Security

- Never commit secrets, credentials, `.env` files, or API keys
- Validate input only at system boundaries (user input, external APIs)
- Do not introduce command injection, XSS, SQL injection, or other OWASP Top 10 vulnerabilities
- If insecure code is written, fix it immediately before proceeding

## Project Structure

> This section should be updated as the project grows.

```
opc/
├── CLAUDE.md          # This file — AI assistant guidance
└── (project files TBD)
```

## Technology Stack

> To be documented once the project is initialized with source code.

- **Language**: TBD
- **Framework**: TBD
- **Package Manager**: TBD
- **Test Runner**: TBD

## Running the Project

> To be documented once a build system is established.

```bash
# Install dependencies
# <command TBD>

# Run tests
# <command TBD>

# Start development server
# <command TBD>
```

## Environment Variables

> Document required environment variables here as they are introduced.

| Variable | Required | Description |
|----------|----------|-------------|
| (none yet) | — | — |

## Key Conventions

> Update this section with project-specific conventions as they emerge (naming schemes, code style, architectural patterns, etc.).

- Follow the language/framework's standard style guide
- Keep functions small and focused
- Prefer explicit over implicit
- Write tests for new functionality
