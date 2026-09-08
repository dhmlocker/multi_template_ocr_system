import importlib


def test_ui_pages_expose_render_functions_without_importing_streamlit_at_module_import():
    modules = [
        'ui.recognition_page', 'ui.template_page', 'ui.history_page',
        'ui.experiments_page', 'ui.settings_page'
    ]
    for name in modules:
        mod = importlib.import_module(name)
        assert callable(getattr(mod, 'render'))
