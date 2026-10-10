# Development environment setup

Tested on **Windows 11 + WSL2 (Ubuntu 24.04)**, 32 GB RAM, NVIDIA Quadro RTX 4000 (8 GB VRAM).
Everything runs inside WSL except Docker Desktop and Ollama, which run on the Windows host.

> All commands below run in the **Ubuntu (WSL) terminal** unless marked *PowerShell*.

## 1. WSL configuration (Windows host)

Create `C:\Users\<you>\.wslconfig` — plain text, UTF-8 **without BOM**, first line must be `[wsl2]`:

```ini
[wsl2]
memory=22GB
swap=8GB
networkingMode=mirrored
```

Apply it (*PowerShell*): `wsl --shutdown`, then reopen Ubuntu.
`networkingMode=mirrored` lets WSL reach Windows services (Ollama) on `localhost`.

## 2. Docker Desktop

In Docker Desktop: **Settings → General → Use the WSL 2 based engine**, then
**Settings → Resources → WSL Integration → enable Ubuntu**, *Apply & restart*, and reopen the terminal.

## 3. Tools inside WSL

uv, pre-commit and Terraform are pinned (NFR-07). These versions must stay in sync with the `env` block of
`.github/workflows/ci.yml`; change both together.

```bash
# Base tools
sudo apt update && sudo apt install -y curl unzip build-essential gh

# uv (Python toolchain; project environments stay per repository)
curl -LsSf https://astral.sh/uv/0.12.23/install.sh | sh
source $HOME/.local/bin/env

# AWS CLI v2 (official installer; the Ubuntu apt package is outdated)
cd /tmp && curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip -q awscliv2.zip && sudo ./aws/install && rm -rf aws awscliv2.zip && cd ~

# Terraform (official HashiCorp repository)
wget -qO- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install -y terraform=1.16.5-1
sudo apt-mark hold terraform   # keep apt upgrade from moving it

# pre-commit (isolated tool environment)
uv tool install pre-commit==4.6.2
```

### LocalStack CLI (`lstk`)

The legacy `localstack` CLI and the `awslocal` / `tflocal` wrappers are deprecated; use `lstk`.
`lstk` is pinned to **1.3.0**, the version the local setup was verified with (NFR-07). Download that release,
check it against the published checksums, and install **all three files** — `lstk` looks for its bundled
extensions (e.g. `doctor`) next to its own binary:

```bash
cd /tmp
gh release download v1.3.0 --repo localstack/lstk \
  --pattern 'lstk_1.3.0_linux_amd64.tar.gz' --pattern 'checksums.txt'
sha256sum --check --ignore-missing checksums.txt    # must print "lstk_1.3.0_linux_amd64.tar.gz: OK"
mkdir -p lstk-extract && tar -xzf lstk_1.3.0_linux_amd64.tar.gz -C lstk-extract
cp lstk-extract/lstk lstk-extract/bundled-extensions lstk-extract/lstk-extensions.toml ~/.local/bin/
cd ~ && lstk --version                              # lstk 1.3.0
```

`lstk` announces newer versions on start; do not run `lstk update`, which installs the latest release.
Change the pinned version deliberately, here and in the NFR-07 note of `docs/requirements.md`.

## 4. Accounts and authentication

```bash
gh auth login                 # GitHub.com → HTTPS → login with a web browser
lstk doctor                   # pre-flight checks
lstk                          # login (browser), choose AWS, start LocalStack, create the "localstack" AWS profile
aws configure set region eu-central-1 --profile localstack
```

From WSL the browser does not open automatically: copy the printed URL into a Windows browser,
check that the one-time code matches, approve, **then** press a key in the terminal.

Create a free LocalStack account (plan: **Hobby**, non-commercial) before running `lstk`.

After this first login, start and stop LocalStack with `make up` and `make down`. `make up` runs the pinned
image `LOCALSTACK_IMAGE` from the `Makefile` (NFR-07) through `lstk start --image`, which does not change your
lstk configuration; a bare `lstk` or `lstk start` uses the tag from that configuration (`latest` by default).

## 5. Verification

```bash
for t in git docker aws terraform uv lstk gh pre-commit claude; do
  printf "%-12s" "$t"; command -v "$t" || echo "MISSING"
done
free -h | head -2                                   # ~21-22 GiB total
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
curl -s http://localhost:11434/api/version          # Ollama on the Windows host
aws sts get-caller-identity --profile localstack    # account 000000000000
lstk status
```

Every path must start with `/usr` or `/home/<you>` — never `/mnt/c/...` (that is a Windows binary).

## 6. Project setup

```bash
make setup      # uv sync + pre-commit install
make lint       # all pre-commit hooks on every file
make test       # unit tests with coverage
```

No linter needs a manual install: ruff comes from `uv.lock`, gitleaks and hadolint are installed by pre-commit,
and tflint runs from a pinned container image. **Docker Desktop must be running to commit Terraform files**
(the tflint hook); `terraform fmt` uses the local Terraform.

## Troubleshooting — issues actually met

| Symptom | Cause | Fix |
| --- | --- | --- |
| `wsl: '=' expected in .wslconfig:1` | A stray first line (e.g. `ini` copied from a code fence) or a BOM | Rewrite the file so line 1 is exactly `[wsl2]`, UTF-8 without BOM |
| `free -h` shows half the RAM | `.wslconfig` not applied | Fix the file, `wsl --shutdown` |
| `docker` resolves to `/mnt/c/...` or "could not be found in this WSL 2 distro" | WSL integration disabled | Enable it in Docker Desktop, reopen the terminal |
| Ollama "not reachable" from WSL | Mirrored networking off, or Ollama server not running | Set `networkingMode=mirrored`; start the Ollama app on Windows |
| `lstk doctor`: unknown command | Only the `lstk` binary was installed | Copy `bundled-extensions` and `lstk-extensions.toml` next to it |
| `localhost.localstack.cloud` does not resolve | Router DNS rebind protection | Use `http://localhost:4566` and S3 path-style addressing |
| `lstk` auth "not confirmed" | A key was pressed before approving in the browser | Rerun `lstk`, approve first, then press a key |
| Profiles in different regions | Defaults written by tools | Align on `eu-central-1` |
