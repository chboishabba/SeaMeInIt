#!/usr/bin/env bash
# Bootstrap a Python virtual environment and execute SeaMeInIt helpers.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON:-python3}"
BOOTSTRAP_MARKER="${VENV_DIR}/.smii_bootstrapped"

create_venv() {
    echo "Creating virtual environment at ${VENV_DIR}" >&2
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
}

REQUIREMENTS_FILE="${SMII_REQUIREMENTS_FILE:-${ROOT_DIR}/requirements-dev.txt}"

bootstrap_dependencies() {
    echo "Installing SeaMeInIt dependencies" >&2

    if [ ! -f "${REQUIREMENTS_FILE}" ]; then
        cat <<EOF >&2
Could not find requirements file at ${REQUIREMENTS_FILE}.
Create the file or set SMII_REQUIREMENTS_FILE to a valid path.
EOF
        exit 1
    fi

    (\
        cd "${ROOT_DIR}" && \
        "${VENV_DIR}/bin/python" -m pip install --upgrade pip && \
        "${VENV_DIR}/bin/python" -m pip install -r "${REQUIREMENTS_FILE}"
    )
    touch "${BOOTSTRAP_MARKER}"
}

ensure_environment() {
    if [ "${SMII_SKIP_BOOTSTRAP:-}" = "1" ]; then
        return
    fi

    if [ ! -d "${VENV_DIR}" ]; then
        create_venv
    fi

    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"

    if [ ! -f "${BOOTSTRAP_MARKER}" ]; then
        bootstrap_dependencies
    fi
}

print_usage() {
    cat <<USAGE
Usage: $(basename "$0") <command> [args...]

Bootstraps the SeaMeInIt development environment (virtualenv + editable install)
and dispatches to a common helper. Recognised commands:

  interactive   Launch the clearance CLI (python -m smii interactive)
  afflec-demo   Run the Ben Afflec fixture demo (python -m smii afflec-demo)
  pytest        Execute the test suite (python -m pytest)
  download-smplx
                Fetch SMPL-X assets (python tools/download_smplx.py)

Any other command is executed verbatim inside the virtual environment.

The requirements manifest used during bootstrapping defaults to
"requirements-dev.txt" at the repository root. Override this by setting the
SMII_REQUIREMENTS_FILE environment variable.

Pass a leading "--" to forward dash-prefixed arguments to the target command,
e.g. $(basename "$0") pytest -- --maxfail=1.
USAGE
}

run_command() {
    if [ $# -eq 0 ]; then
        print_usage
        exit 1
    fi

    if [ "${1:-}" = "--" ]; then
        shift || true
    fi

    if [ $# -eq 0 ]; then
        echo "No command provided after -- sentinel." >&2
        print_usage
        exit 1
    fi

    local cmd="$1"
    shift || true

    case "${cmd}" in
        interactive|afflec-demo)
            exec -- python -m smii "${cmd}" "$@"
            ;;
        smii)
            exec -- python -m smii "$@"
            ;;
        pytest)
            exec -- python -m pytest "$@"
            ;;
        download-smplx)
            exec -- python "${ROOT_DIR}/tools/download_smplx.py" "$@"
            ;;
        *)
            exec -- "${cmd}" "$@"
            ;;
    esac
}

main() {
    if [ $# -eq 0 ] || [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
        print_usage
        exit 0
    fi

    ensure_environment
    run_command "$@"
}

main "$@"
