#!/bin/bash

# Setup script for lidar-fault-localization
# This script is designed to work on multiple Linux distributions

set -euo pipefail

echo "Setting up lidar-fault-localization..."
echo "======================================"

# Function to detect OS and package manager
detect_os() {
    if command -v apt-get &> /dev/null; then
        echo "ubuntu"
    elif command -v dnf &> /dev/null; then
        echo "fedora"
    elif command -v yum &> /dev/null; then
        echo "rhel"
    elif command -v pacman &> /dev/null; then
        echo "arch"
    elif command -v zypper &> /dev/null; then
        echo "opensuse"
    else
        echo "unknown"
    fi
}

# Function to install system dependencies
install_system_deps() {
    local os=$1
    echo "Installing system dependencies for $os..."

    case $os in
        ubuntu)
            sudo apt-get update
            sudo apt-get install -y python3-venv python3-pip python3-tk python3-dev build-essential cmake git ninja-build
            # KISS-ICP specific dependencies
            sudo apt-get install -y libeigen3-dev libtbb-dev
            ;;
        fedora)
            sudo dnf install -y python3 python3-pip python3-tkinter python3-devel cmake git ninja-build gcc gcc-c++
            # KISS-ICP specific dependencies
            sudo dnf install -y eigen3-devel tbb-devel
            ;;
        rhel)
            sudo yum install -y python3 python3-pip python3-tkinter python3-devel cmake git ninja-build gcc gcc-c++
            # KISS-ICP specific dependencies (may need EPEL)
            sudo yum install -y eigen3-devel tbb-devel || echo "Warning: Some KISS-ICP dependencies may not be available in base RHEL repos"
            ;;
        arch)
            sudo pacman -S --noconfirm python python-pip tk cmake git ninja base-devel
            # KISS-ICP specific dependencies
            sudo pacman -S --noconfirm eigen tbb
            ;;
        opensuse)
            sudo zypper install -y python3 python3-pip python3-tk cmake git ninja gcc gcc-c++
            # KISS-ICP specific dependencies
            sudo zypper install -y libeigen3-devel tbb-devel
            ;;
        *)
            echo "Warning: Could not detect package manager. Please ensure you have:"
            echo "  - Python 3.10+ with venv support"
            echo "  - pip"
            echo "  - build tools (cmake, gcc, ninja, etc.)"
            echo "  - git"
            echo "  - tkinter for Python"
            return 1
            ;;
    esac
}

# Function to find Python executable
find_python() {
    # Try different Python executables
    for cmd in python3.12 python3.11 python3.10 python3 python; do
        if command -v "$cmd" &> /dev/null; then
            # Check version
            if "$cmd" -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

# Function to validate installation
validate_installation() {
    echo "Validating installation..."

    # Check if virtual environment is activated
    if [[ "$VIRTUAL_ENV" != *"lidar-fault-localization"* ]]; then
        echo "Error: Virtual environment not activated"
        return 1
    fi

    # Check if key packages are installed
    python -c "import numpy, pandas, matplotlib, scipy, evo" 2>/dev/null || {
        echo "Error: Key Python packages not installed"
        return 1
    }

    # Check if KISS-ICP is installed from kiss_icp_modifications
    python -c "import kiss_icp" 2>/dev/null || {
        echo "Error: KISS-ICP not properly installed"
        return 1
    }
    
    # Check if lfl package is installed
    python -c "import lfl, lfi, lfa" 2>/dev/null || {
        echo "Error: LFL package not properly installed"
        return 1
    }

    # Check if modifications are present (should have fault_model parameter)
    python -c "from kiss_icp.datasets.kitti import KITTIOdometryDataset; import inspect; sig = inspect.signature(KITTIOdometryDataset.__init__); assert 'fault_model' in sig.parameters" 2>/dev/null || {
        echo "Error: KISS-ICP modifications not applied correctly"
        return 1
    }

    echo "✓ Installation validated successfully"
    return 0
}

# Main setup process
main() {
    # Detect OS
    OS=$(detect_os)
    echo "Detected OS: $OS"

    # Install system dependencies
    if ! install_system_deps "$OS"; then
        echo "Warning: System dependency installation may have failed"
    fi

    # Find suitable Python
    echo "Finding Python 3.10+..."
    PYTHON_CMD=$(find_python)
    if [ $? -ne 0 ]; then
        echo "Error: Python 3.10+ not found"
        echo "Please install Python 3.10 or higher"
        exit 1
    fi

    PYTHON_VERSION=$("$PYTHON_CMD" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo "Using Python $PYTHON_VERSION at $PYTHON_CMD"

    # Check if virtual environment already exists
    if [ -d "venv" ]; then
        echo "Virtual environment already exists. Removing..."
        rm -rf venv
    fi

    # Create virtual environment
    echo "Creating virtual environment..."
    "$PYTHON_CMD" -m venv venv

    # Activate virtual environment
    echo "Activating virtual environment..."
    # shellcheck source=/dev/null
    source venv/bin/activate

    # Upgrade pip and install wheel
    echo "Upgrading pip..."
    pip install --upgrade pip wheel setuptools

    # Check Python minor version and compatibility for some pinned packages
    echo "Checking Python compatibility for pinned packages..."
    # Extract major and minor version as integers
    PY_MAJOR=$("$PYTHON_CMD" -c 'import sys; print(sys.version_info.major)')
    PY_MINOR=$("$PYTHON_CMD" -c 'import sys; print(sys.version_info.minor)')

    # If Python < 3.11, patch requirements that require Python >=3.11 or unavailable versions
    if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
        echo "Detected Python < 3.11; applying compatibility fixes to requirements.txt"
        # Backup original requirements
        cp requirements.txt requirements.txt.bak || true

        # Replace unavailable/newer mcap-ros2-support with a published compatible version
        if grep -q "^mcap-ros2-support==" requirements.txt; then
            echo "Patching mcap-ros2-support pin to a compatible published version (0.5.7)"
            sed -i 's/^mcap-ros2-support==.*$/mcap-ros2-support==0.5.7/' requirements.txt
        fi

        # (Optional) Add other compatibility replacements here if needed in future
    fi

    # Install pinned Python dependencies (includes MCAP/ROS2 support & viz stack)
    echo "Installing Python requirements..."
    pip install -r requirements.txt

    # Install project in development mode (uses pyproject.toml)
    echo "Installing lfl editable package..."
    pip install -e .

    # Setup KISS-ICP using local modifications
    echo "Setting up KISS-ICP from local modifications..."
    
    # Check if modifications directory exists
    if [ ! -d "kiss_icp_modifications" ]; then
        echo "Error: kiss_icp_modifications directory not found"
        echo "Please ensure the repository was cloned with all files"
        exit 1
    fi

    # Create a .pth file to make kiss_icp_modifications discoverable
    # This is a simple and reliable way to add it to sys.path without hardcoding
    SITE_PACKAGES="venv/lib/python$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages"
    
    if [ -d "$SITE_PACKAGES" ]; then
        # Get absolute path to kiss_icp_modifications
        KISS_ICP_PATH="$(cd kiss_icp_modifications && pwd)"
        echo "$KISS_ICP_PATH" > "$SITE_PACKAGES/kiss_icp_modifications.pth"
        echo "✓ Added kiss_icp_modifications to Python path"
    else
        echo "Warning: Could not find site-packages directory"
    fi
    
    echo "✓ KISS-ICP configured from kiss_icp_modifications/"

    # Optional extras (uncomment if you need Open3D visualizer)
    # pip install \"open3d>=0.18\"

    # Validate installation
    if validate_installation; then
        echo ""
        echo "🎉 Setup complete!"
        echo ""
        echo "Installation Summary:"
        echo "  - KISS-ICP installed from: kiss_icp_modifications/ (portable)"
        echo "  - LFL package installed in editable mode"
        echo "  - All dependencies installed"
        echo ""
        echo "Next steps:"
        echo "1. Download KITTI dataset to data/kitti/"
        echo "   - Visit: http://www.cvlibs.net/datasets/kitti/eval_odometry.php"
        echo "   - Download 'data_odometry_velodyne.zip' and 'data_odometry_poses.zip'"
        echo "   - Extract to data/kitti/"
        echo ""
        echo "2. Activate environment: source venv/bin/activate"
        echo "3. Test installation: python3 -m lfl.cli --help"
        echo "4. Run example: python3 -m lfl.runner --fault_model fog --sequences 03"
        echo ""
        echo "For detailed instructions, see README.md"
    else
        echo "❌ Installation validation failed"
        echo "Please check the error messages above"
        exit 1
    fi
}

# Run main function
main "$@"
