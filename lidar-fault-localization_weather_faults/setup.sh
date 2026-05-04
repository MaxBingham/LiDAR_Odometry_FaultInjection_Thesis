#!/bin/bash

# Setup script for lidar-fault-localization
# This script is designed to work on multiple Linux distributions

set -e

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
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Error: Virtual environment not activated"
        return 1
    fi

    # Check if key packages are installed
    python -c "import numpy, pandas, matplotlib, scipy, evo" 2>/dev/null || {
        echo "Error: Key Python packages not installed"
        return 1
    }

    # Check if KISS-ICP modifications are applied
    if [ ! -f "external/kiss-icp/python/kiss_icp/datasets/kitti.py" ]; then
        echo "Error: KISS-ICP not properly set up"
        return 1
    fi

    # Check if modifications are present
    if ! grep -q "fault_model" "external/kiss-icp/python/kiss_icp/datasets/kitti.py"; then
        echo "Error: KISS-ICP modifications not applied"
        return 1
    fi

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
    source venv/bin/activate

    # Upgrade pip and install wheel
    echo "Upgrading pip..."
    pip install --upgrade pip wheel setuptools

    # Install project in development mode
    echo "Installing project dependencies..."
    pip install -e .

    # Install CMake 3.24+ for KISS-ICP build
    echo "Installing CMake 3.24+..."
    pip install --upgrade "cmake>=3.24"

    # Setup KISS-ICP
    echo "Setting up KISS-ICP..."
    if [ ! -d "external/kiss-icp" ]; then
        echo "Cloning KISS-ICP..."
        git clone https://github.com/PRBonn/kiss-icp.git external/kiss-icp
    else
        echo "KISS-ICP directory already exists"
    fi

    # Check if modifications directory exists
    if [ ! -d "kiss_icp_modifications" ]; then
        echo "Error: kiss_icp_modifications directory not found"
        echo "Please ensure the repository was cloned with all files"
        exit 1
    fi

    # Apply modifications
    echo "Applying KISS-ICP modifications..."
    if [ -d "external/kiss-icp/python/kiss_icp" ]; then
        cp -r kiss_icp_modifications/* external/kiss-icp/python/
        echo "✓ Modifications applied"
    else
        echo "Error: KISS-ICP python package not found"
        exit 1
    fi

    # Install KISS-ICP
    echo "Installing KISS-ICP..."
    # Install build dependencies first
    pip install scikit-build-core pybind11
    # Use --no-build-isolation so venv cmake is found during build
    export PATH="$(pwd)/venv/bin:$PATH"
    cd external/kiss-icp
    pip install --no-build-isolation -e python
    cd ../..

    # Validate installation
    if validate_installation; then
        echo ""
        echo "🎉 Setup complete!"
        echo ""
        echo "Next steps:"
        echo "1. Download KITTI dataset to data/kitti/"
        echo "   - Visit: http://www.cvlibs.net/datasets/kitti/eval_odometry.php"
        echo "   - Download 'data_odometry_velodyne.zip' and 'data_odometry_poses.zip'"
        echo "   - Extract to data/kitti/"
        echo ""
        echo "2. Activate environment: source venv/bin/activate"
        echo "3. Test installation: lfl_pipeline --help"
        echo "4. Run example: lfl_pipeline --sequence 07"
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