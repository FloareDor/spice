# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[('E:\\cmu-e\\spice\\clamp3', 'clamp3'), ('tag_embeddings.npy', '.')],
    hiddenimports=['sklearn.neighbors._typedefs', 'sklearn.utils._cython_blas', 'sklearn.neighbors._quad_tree', 'sklearn.tree._utils', 'scipy.special.cython_special', 'scipy.linalg.cython_blas', 'scipy.linalg.cython_lapack', 'pandas', 'win32timezone', 'search', 'indexer', 'database', 'autotagger', 'waveform', 'validate_core'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'notebook'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

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
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LocalVibe',
)
