"""Minimal smoke tests to ensure package imports and subpackages resolve.

These tests avoid loading heavy model dependencies.
"""

def test_package_imports_version():
    import transparent_background_mcp as tbm
    assert hasattr(tbm, "__version__")
    from transparent_background_mcp import __version__
    assert isinstance(__version__, str)


def test_models_import():
    from transparent_background_mcp.models import BEN2Model
    assert BEN2Model.__name__ == "BEN2Model"

