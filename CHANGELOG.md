# Changelog

## [Unreleased] - 2025-10-31

### Repository Cleanup and Restructuring

#### Removed
- **St_CABBIO03.py** - Legacy file that duplicated functionality in main.py (99KB)
  - This was an old version of the dashboard application that is no longer needed
  - All functionality is now consolidated in main.py

#### Renamed
- **teste/** → **test_data/** - Improved directory naming for clarity
  - Contains test data files for simulations
  
- **Body/estimacion_parametros/** → **Body/parameter_estimation/** - Consistent English naming
  - Maintains consistency with other directory names
  - Updated all imports in main.py to reflect the change

#### Added
- **Package Structure** - Added `__init__.py` files to all packages:
  - `Body/__init__.py` - Main package documentation
  - `Body/modeling/__init__.py` - Modeling modules documentation
  - `Body/parameter_estimation/__init__.py` - Parameter estimation documentation
  - `Body/estimation/__init__.py` - State estimation documentation
  - `Body/control/__init__.py` - Control modules documentation
  - `Body/control/regulatorio/__init__.py` - Regulatory control documentation
  - `Body/control/avanzado/__init__.py` - Advanced control documentation

- **setup.py** - Proper Python package configuration
  - Enables installation via `pip install -e .`
  - Defines package metadata and dependencies
  - Supports development and production installations

- **.gitignore updates** - Added build/dist exclusions
  - Excludes `build/`, `dist/`, `*.egg-info/` directories
  - Prevents package build artifacts from being committed

#### Modified
- **README.md** - Updated with:
  - Detailed repository structure tree
  - Improved installation instructions (Quick Start + Manual)
  - Package-based installation option
  - More comprehensive directory descriptions

- **main.py** - Updated imports:
  - Changed `from Body.estimacion_parametros` to `from Body.parameter_estimation`
  - All three parameter estimation modules updated

### Testing
- All package imports verified successfully
- All module functions tested and working
- Application runs without errors
- No breaking changes to functionality

### Impact
- **No functional changes** - All features work exactly as before
- **Improved code organization** - Clearer package structure
- **Better maintainability** - Standard Python package conventions
- **Easier installation** - Can now be installed as a package
- **Reduced repository size** - Removed 99KB of duplicate code
