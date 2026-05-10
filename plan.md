# Plan: Convert to Jupyter Notebooks Devcontainer

## Goal
Adapt the existing zero-config devcontainer template into a Jupyter Notebooks project environment, with both a development devcontainer and a production JupyterLab Docker deployment.

## Decisions Made

| Decision | Choice |
|---|---|
| Devcontainer base image | `mcr.microsoft.com/devcontainers/python:3` |
| Jupyter interface | Both JupyterLab + VS Code Notebooks |
| Package manager | pip |
| GPU/CUDA | Not needed |
| Node.js | Remove (Python-only) |
| Extra Python packages | None — keep minimal |
| Production Dockerfile | `docker/Dockerfile.prod` |
| Production base image | `quay.io/jupyter/base-notebook` |
| Docker Compose | Yes — `docker/docker-compose.yml` |
| Jupyter auth (production) | No auth (dev-only environment) |

---

## Files to Modify

### 1. `.devcontainer/Dockerfile`
**FROM** `mcr.microsoft.com/devcontainers/base:ubuntu`
**TO** `mcr.microsoft.com/devcontainers/python:3`

Add after `FROM`:
```dockerfile
RUN pip install --upgrade pip && pip install jupyterlab notebook
```

### 2. `.devcontainer/devcontainer.json`
- Remove `ghcr.io/devcontainers/features/node:1` feature
- Add port forwarding for JupyterLab:
  - `"forwardPorts": [8888]`
  - `"portsAttributes"` for port 8888 (label: `"JupyterLab"`, protocol: `"http"`)
- Add VS Code extensions:
  - `ms-python.python`, `ms-python.vscode-pylance`
  - `ms-toolsai.jupyter`, `ms-toolsai.jupyter-keymap`, `ms-toolsai.jupyter-renderers`

### 3. `.gitignore`
Add:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
.venv/
venv/

# Jupyter
.ipynb_checkpoints/
```

### 4. `.vscode/extensions.json`
Add to `recommendations`:
```json
"ms-python.python",
"ms-toolsai.jupyter"
```

### 5. `AGENTS.md`
Populate with project conventions (Python, pip, JupyterLab, production build/run).

---

## Files to Create

### 6. `docker/Dockerfile.prod`
Production JupyterLab image based on `quay.io/jupyter/base-notebook`:
- Image already includes JupyterLab pre-installed
- Set `NB_USER` / `NB_UID` if needed
- Disable authentication (`--NotebookApp.token=''`)
- No additional Python packages by default (add via requirements.txt later)

### 7. `docker/docker-compose.yml`
Compose file to run the production container:
- Build from `docker/Dockerfile.prod` (context: repo root)
- Port mapping: `8888:8888`
- Volumes:
  - `./notebooks:/home/jovyan/work` — notebooks directory
  - `./data:/home/jovyan/data` — data directory
- Environment: disable token/password auth
- Command: `start-notebook.sh` with no-auth flags

---

## Summary

| File | Action |
|---|---|
| `.devcontainer/Dockerfile` | Modify |
| `.devcontainer/devcontainer.json` | Modify |
| `.gitignore` | Modify |
| `.vscode/extensions.json` | Modify |
| `AGENTS.md` | Modify |
| `docker/Dockerfile.prod` | **Create** |
| `docker/docker-compose.yml` | **Create** |
