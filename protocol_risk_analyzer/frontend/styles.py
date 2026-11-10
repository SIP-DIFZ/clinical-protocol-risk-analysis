_STYLE_RENDER_IN_PROGRESS = False


def render_styles() -> None:
    global _STYLE_RENDER_IN_PROGRESS
    if _STYLE_RENDER_IN_PROGRESS:
        return
    _STYLE_RENDER_IN_PROGRESS = True
    try:
        import __main__ as running_main

        legacy_render = getattr(running_main, "_render_styles", None)
        if callable(legacy_render):
            legacy_render()
            return

        from protocol_risk_analyzer import web_app

        web_app._render_styles()
    finally:
        _STYLE_RENDER_IN_PROGRESS = False


__all__ = ["render_styles"]
