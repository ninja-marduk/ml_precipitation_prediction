"""
Workflow utility: log versions of installed libraries and script checksums.

Usage in workflow scripts:
    from _versions import log_environment, log_script_version
    log_environment(logger, ['numpy', 'xarray', 'netCDF4', 'tensorflow'])
    log_script_version(logger, __file__)
"""

import hashlib
import importlib
import platform
import sys
from pathlib import Path


# Script version stamp - update with each release
WORKFLOW_VERSION = '2.1.0-multiseed'
WORKFLOW_DATE = '2026-04-19'


def log_script_version(logger, script_path: str) -> None:
    """Log the script filename, MD5 checksum, and workflow version.

    This allows BSC operators to verify that the deployed script matches
    the expected version from the development machine.
    """
    p = Path(script_path)
    try:
        md5 = hashlib.md5(p.read_bytes()).hexdigest()[:12]
    except Exception:
        md5 = 'unknown'
    logger.info(f'  Script    : {p.name}')
    logger.info(f'  MD5       : {md5}')
    logger.info(f'  Version   : {WORKFLOW_VERSION} ({WORKFLOW_DATE})')
    logger.info(f'  Full path : {p.resolve()}')


def get_version(pkg_name: str) -> str:
    """Return the installed version of a package, or 'NOT INSTALLED'."""
    try:
        # Try importlib.metadata first (more reliable for distributions)
        try:
            from importlib.metadata import version as _v
            return _v(pkg_name)
        except Exception:
            pass
        # Fallback: import the module and read __version__
        mod = importlib.import_module(pkg_name)
        return getattr(mod, '__version__', 'unknown')
    except Exception:
        return 'NOT INSTALLED'


def log_environment(logger, packages: list = None) -> None:
    """
    Log Python version, platform, and versions of relevant packages.

    Args:
        logger: Python logger instance
        packages: List of package names to check. If None, uses common defaults.
    """
    if packages is None:
        packages = [
            'numpy', 'pandas', 'scipy', 'xarray', 'netCDF4', 'h5netcdf',
            'sklearn', 'matplotlib',
        ]

    logger.info('-' * 60)
    logger.info('  Environment')
    logger.info('-' * 60)
    logger.info(f'  Python    : {sys.version.split()[0]}')
    logger.info(f'  Platform  : {platform.system()} {platform.release()}')
    logger.info(f'  Machine   : {platform.machine()}')

    # Map alternate package names → import names where they differ
    name_map = {
        'sklearn': 'sklearn',
        'scikit-learn': 'sklearn',
        'PyTorch': 'torch',
        'torch-geometric': 'torch_geometric',
        'pyg': 'torch_geometric',
        'TensorFlow': 'tensorflow',
    }

    for pkg in packages:
        import_name = name_map.get(pkg, pkg)
        ver = get_version(import_name)
        # Pretty display name
        display = pkg if pkg not in ('sklearn',) else 'scikit-learn'
        logger.info(f'  {display:10s}: {ver}')
    logger.info('-' * 60)
