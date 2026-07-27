"""
Plugin and hook system for extensible RAG pipeline.
"""

import logging
from typing import Callable, Any
from dataclasses import dataclass, field

logger = logging.getLogger("rag.plugins")


@dataclass
class HookResult:
    data: Any = None
    modified: bool = False
    abort: bool = False
    abort_reason: str = ""


class HookRegistry:
    """Registry for pipeline hooks that can modify behavior at key points."""

    HOOKS = [
        "pre_retrieve",      # Before retrieval
        "post_retrieve",     # After retrieval, before verification
        "pre_verify",        # Before verification
        "post_verify",       # After verification
        "pre_generate",      # Before generation
        "post_generate",     # After generation
        "pre_index",         # Before indexing documents
        "post_index",        # After indexing
        "on_error",          # On pipeline error
    ]

    def __init__(self):
        self._hooks: dict[str, list[Callable]] = {h: [] for h in self.HOOKS}

    def register(self, hook_name: str, callback: Callable):
        if hook_name not in self._hooks:
            logger.warning(f"Unknown hook: {hook_name}")
            return
        self._hooks[hook_name].append(callback)
        logger.info(f"Registered hook: {hook_name} -> {callback.__name__}")

    def unregister(self, hook_name: str, callback: Callable):
        if hook_name in self._hooks:
            self._hooks[hook_name] = [h for h in self._hooks[hook_name] if h != callback]

    def run(self, hook_name: str, data: Any = None) -> HookResult:
        result = HookResult(data=data)
        for callback in self._hooks.get(hook_name, []):
            try:
                out = callback(result.data)
                if out is not None:
                    result.data = out
                    result.modified = True
            except Exception as e:
                logger.error(f"Hook {hook_name}/{callback.__name__} failed: {e}")
        return result

    def list_hooks(self) -> dict:
        return {name: [f.__name__ for f in callbacks] for name, callbacks in self._hooks.items()}


class Plugin:
    """Base class for RAG plugins."""

    name: str = "base"
    description: str = ""

    def __init__(self):
        self.hooks = HookRegistry()

    def register(self, hook_registry: HookRegistry):
        pass

    def unregister(self, hook_registry: HookRegistry):
        pass


class PluginManager:
    def __init__(self):
        self.plugins: dict[str, Plugin] = {}
        self.hooks = HookRegistry()

    def load_plugin(self, plugin: Plugin):
        plugin.register(self.hooks)
        self.plugins[plugin.name] = plugin
        logger.info(f"Loaded plugin: {plugin.name}")

    def unload_plugin(self, name: str):
        plugin = self.plugins.pop(name, None)
        if plugin:
            plugin.unregister(self.hooks)
            logger.info(f"Unloaded plugin: {name}")

    def run_hook(self, hook_name: str, data: Any = None) -> HookResult:
        return self.hooks.run(hook_name, data)

    def list_plugins(self) -> list[dict]:
        return [{"name": p.name, "description": p.description} for p in self.plugins.values()]
