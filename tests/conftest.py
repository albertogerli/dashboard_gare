"""Pytest configuration for dashboard_gare tests."""


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (loading full CSV, etc.)")
