# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Base directory
base_dir = os.path.abspath('.')

# Collect data files from qfluentwidgets
qfluentwidgets_data = collect_data_files('qfluentwidgets')

added_files = [
    (os.path.join(base_dir, 'clamp3'), 'clamp3'),
    (os.path.join(base_dir, 'tag_embeddings.npy'), '.'),
    (os.path.join(base_dir, 'library_config.json'), '.'),
] + qfluentwidgets_data

a = Analysis(
    ['app.py'],
    pathex=[base_dir],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'sklearn.neighbors._typedefs',
        'sklearn.utils._cython_blas',
        'sklearn.neighbors._quad_tree',
        'sklearn.tree._utils',
        'scipy.special.cython_special',
        'scipy.linalg.cython_blas',
        'scipy.linalg.cython_lapack',
        'pandas',
        'win32timezone',
        'search',
        'indexer',
        'database',
        'autotagger',
        'waveform',
        'galaxy',
        'onboarding',
        'model_wrapper',
        'validate_core'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'notebook'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LocalVibe',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # Set to True if you want to see logs for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='app.ico' # Uncomment if you have an icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LocalVibe',
)