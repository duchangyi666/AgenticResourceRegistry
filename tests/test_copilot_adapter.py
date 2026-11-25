# SPDX-FileCopyrightText: 2026 杜昌亿 <218478651+duchangyi666@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0

"""Gap coverage for the VS Code Copilot adapter in observal_cli/harness/copilot.py.

tests/test_copilot_integration.py already covers detect_hooks installed and
missing, scan_home finding an MCP, scan_home with no .vscode directory, and
scan_project finding an MCP. This file covers the remaining gaps:

* mcpServers as well as servers, URL servers, source labels, malformed JSON,
  and non-dict server entries
* discover_home and discover_project reading .vscode/mcp.json
* scan results that stay empty of skills and hooks
* headless_command NotSupportedError and the METHOD_FEATURE_MAP gate
* hook spec events and markers, and detect_hooks from a .vscode config dir
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from observal_cli.discovery.models import DiagnosticCode, DiscoveryScope
from observal_cli.harness import DiscoveredMcp, HookSpec, NotSupportedError, ScanResult
from observal_cli.harness.base import _check_feature
from observal_cli.harness.copilot import CopilotAdapter
from observal_cli.harness.protocol import HeadlessRequest
from observal_shared.harness_registry import HARNESS_REGISTRY

if TYPE_CHECKING:
    from pathlib import Path

_OBSERVAL_HOOK_COMMAND = "python -m observal_cli.hooks.session_push --harness copilot"
_SCOPES = ("home", "project")


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point Path.home at the test tmp dir so scans never read a real home."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)


def _write_mcp(root: Path, payload: object) -> None:
    path = root / ".vscode" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_text(text)


def _scan(root: Path, scope: str) -> ScanResult:
    adapter = CopilotAdapter()
    if scope == "home":
        return adapter.scan_home(home=root)
    return adapter.scan_project(root)


def _source(scope: str) -> str:
    return "copilot:global" if scope == "home" else "copilot:project"


def _description(scope: str, name: str) -> str:
    prefix = "Copilot MCP" if scope == "home" else "Copilot project MCP"
    return f"{prefix}: {name}"


def _write_observal_hook(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "observal.json").write_text(
        json.dumps(
            {
                "version": 1,
                "hooks": {
                    "SessionStart": [{"type": "command", "command": _OBSERVAL_HOOK_COMMAND}],
                },
            }
        )
    )


def _write_ignored_components(root: Path) -> None:
    """Registry skill, hook, and agent paths the Copilot scan does not read."""
    for relative in (
        ".github/skills/helper",
        ".copilot/skills/user-skill",
    ):
        skill = root / relative
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\ndescription: Ignored skill\n---\nBody\n")
    for relative in (".github/hooks", ".copilot/hooks"):
        hooks = root / relative
        hooks.mkdir(parents=True)
        (hooks / "observal.json").write_text("{}\n")
    agents = root / ".github" / "agents"
    agents.mkdir(parents=True)
    (agents / "helper.agent.md").write_text("---\ndescription: Ignored agent\n---\nPrompt\n")


class TestCopilotMcpScan:
    """Legacy scan_home and scan_project MCP edge cases."""

    def test_omitted_home_reads_patched_path_home(self, tmp_path: Path) -> None:
        _write_mcp(tmp_path, {"servers": {"default-home": {"url": "https://mcp.example.test/home"}}})

        result = CopilotAdapter().scan_home()

        assert [mcp.name for mcp in result.mcps] == ["default-home"]
        assert result.mcps[0].source == "copilot:global"
        assert result.mcps[0].url == "https://mcp.example.test/home"

    @pytest.mark.parametrize("scope", _SCOPES)
    def test_mcp_servers_key_records_url_server_and_source(self, tmp_path: Path, scope: str) -> None:
        _write_mcp(
            tmp_path,
            {"mcpServers": {"remote": {"url": "https://mcp.example.test/mcp"}}},
        )

        result = _scan(tmp_path, scope)

        assert len(result.mcps) == 1
        mcp = result.mcps[0]
        assert mcp.name == "remote"
        assert mcp.command is None
        assert mcp.args == []
        assert mcp.url == "https://mcp.example.test/mcp"
        assert mcp.source == _source(scope)
        assert mcp.description == _description(scope, "remote")

    @pytest.mark.parametrize("scope", _SCOPES)
    def test_servers_key_skips_non_dict_entries(self, tmp_path: Path, scope: str) -> None:
        _write_mcp(
            tmp_path,
            {
                "servers": {
                    "kept": {"command": "node", "args": ["server.js"]},
                    "string-entry": "npx server",
                    "list-entry": ["node", "server.js"],
                    "null-entry": None,
                }
            },
        )

        result = _scan(tmp_path, scope)

        assert [mcp.name for mcp in result.mcps] == ["kept"]
        kept = result.mcps[0]
        assert kept.command == "node"
        assert kept.args == ["server.js"]
        assert kept.url is None
        assert kept.source == _source(scope)
        assert kept.description == _description(scope, "kept")

    @pytest.mark.parametrize("scope", _SCOPES)
    def test_servers_key_takes_precedence_over_mcp_servers(self, tmp_path: Path, scope: str) -> None:
        _write_mcp(
            tmp_path,
            {
                "servers": {"from-servers": {"command": "node", "args": ["a.js"]}},
                "mcpServers": {"from-mcp-servers": {"command": "python", "args": ["b.py"]}},
            },
        )

        result = _scan(tmp_path, scope)

        assert [mcp.name for mcp in result.mcps] == ["from-servers"]
        assert result.mcps[0].command == "node"
        assert result.mcps[0].source == _source(scope)

    @pytest.mark.parametrize("scope", _SCOPES)
    def test_malformed_json_returns_empty_scan_result(self, tmp_path: Path, scope: str) -> None:
        _write_mcp(tmp_path, "{not json")

        result = _scan(tmp_path, scope)

        assert result == ScanResult()

    @pytest.mark.parametrize("scope", _SCOPES)
    def test_scan_leaves_skills_and_hooks_empty(self, tmp_path: Path, scope: str) -> None:
        _write_ignored_components(tmp_path)
        _write_mcp(tmp_path, {"servers": {"kept": {"command": "node", "args": ["server.js"]}}})

        result = _scan(tmp_path, scope)

        assert [mcp.name for mcp in result.mcps] == ["kept"]
        assert result.skills == []
        assert result.hooks == []
        assert result.agents == []


class TestCopilotDiscovery:
    """Read-only inventory discovery from .vscode/mcp.json."""

    def test_discover_home_uses_patched_path_home_when_omitted(self, tmp_path: Path) -> None:
        _write_mcp(tmp_path, {"mcpServers": {"default-home": {"url": "https://mcp.example.test/home"}}})

        result = CopilotAdapter().discover_home()

        assert len(result.evidence) == 1
        evidence = result.evidence[0]
        assert isinstance(evidence.component, DiscoveredMcp)
        assert evidence.component.name == "default-home"
        assert evidence.component.source == "copilot:global"
        assert evidence.scope is DiscoveryScope.USER
        assert evidence.harness == "copilot"
        assert evidence.display_path == "~/.vscode/mcp.json"
        assert result.diagnostics == []

    @pytest.mark.parametrize(
        ("scope", "key", "expected_source", "expected_scope", "display_path"),
        [
            ("home", "servers", "copilot:global", DiscoveryScope.USER, "~/.vscode/mcp.json"),
            ("project", "mcpServers", "copilot:project", DiscoveryScope.PROJECT, "<project>/.vscode/mcp.json"),
        ],
    )
    def test_discover_reads_vscode_mcp_json(
        self,
        tmp_path: Path,
        scope: str,
        key: str,
        expected_source: str,
        expected_scope: DiscoveryScope,
        display_path: str,
    ) -> None:
        root = tmp_path if scope == "home" else tmp_path / "repo"
        _write_ignored_components(root)
        _write_mcp(
            root,
            {
                key: {
                    "remote": {"url": "https://mcp.example.test/mcp"},
                    "skipped": "not-a-server",
                }
            },
        )
        adapter = CopilotAdapter()

        result = adapter.discover_home(home=root) if scope == "home" else adapter.discover_project(root)

        assert len(result.evidence) == 1
        evidence = result.evidence[0]
        assert evidence.scope is expected_scope
        assert evidence.harness == "copilot"
        assert evidence.display_path == display_path
        assert isinstance(evidence.component, DiscoveredMcp)
        mcp = evidence.component
        assert mcp.name == "remote"
        assert mcp.command is None
        assert mcp.args == []
        assert mcp.url == "https://mcp.example.test/mcp"
        assert mcp.source == expected_source
        assert mcp.description == "Copilot MCP: remote"
        assert [item.code for item in result.diagnostics] == [DiagnosticCode.METADATA_MALFORMED]
        assert [item.message for item in result.diagnostics] == ["MCP server entry must be an object"]


class TestCopilotHookSpec:
    """Hook spec and .vscode config-dir detection."""

    def test_hook_spec_lists_vscode_events_and_markers(self) -> None:
        spec = CopilotAdapter().get_hook_spec()

        assert isinstance(spec, HookSpec)
        assert spec.events == [
            "SessionStart",
            "UserPromptSubmit",
            "PreToolUse",
            "PostToolUse",
            "Stop",
        ]
        assert spec.format == "command"
        assert spec.markers == ["observal", "OBSERVAL"]

    def test_vscode_config_dir_resolves_hooks_from_github_hooks(self, tmp_path: Path) -> None:
        _write_observal_hook(tmp_path / ".github" / "hooks")
        vscode = tmp_path / ".vscode"
        vscode.mkdir()

        assert CopilotAdapter().detect_hooks(vscode) == "installed"

    def test_vscode_config_dir_ignores_marker_inside_vscode(self, tmp_path: Path) -> None:
        vscode = tmp_path / ".vscode"
        _write_observal_hook(vscode)

        assert CopilotAdapter().detect_hooks(vscode) == "missing"

    def test_vscode_config_dir_falls_back_to_user_copilot_hooks(self, tmp_path: Path) -> None:
        _write_observal_hook(tmp_path / ".copilot" / "hooks")
        vscode = tmp_path / ".vscode"
        vscode.mkdir()

        assert CopilotAdapter().detect_hooks(vscode) == "installed"


class TestCopilotSupportGates:
    """Capability and headless_run gates."""

    def test_headless_command_raises_without_headless_run(self, tmp_path: Path) -> None:
        assert HARNESS_REGISTRY["copilot"]["headless_run"] is False
        request = HeadlessRequest(
            agent_name="reviewer",
            instructions="Review the diff.",
            message="Look at the tests.",
            workdir=tmp_path,
            scratch_dir=tmp_path / "scratch",
            session_id="session-1",
        )

        with pytest.raises(NotSupportedError, match="copilot does not support headless_command") as exc_info:
            CopilotAdapter().headless_command(request)

        assert exc_info.value.harness_name == "copilot"
        assert exc_info.value.method_name == "headless_command"

    def test_feature_gate_rejects_methods_when_capability_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The registry gate lives in ``_check_feature``, which BaseAdapter calls.

        CopilotAdapter replaces scan_home and get_hook_spec without calling that
        helper, so dropping a capability does not make those overrides raise.
        """
        spec = HARNESS_REGISTRY["copilot"]
        capabilities = spec["capabilities"]
        assert {"mcp_servers", "hooks", "skills", "prompts"} <= set(capabilities)
        for method in ("scan_home", "scan_project", "get_hook_spec", "detect_hooks"):
            _check_feature("copilot", method)

        monkeypatch.setitem(spec, "capabilities", set(capabilities) - {"mcp_servers"})
        for method in ("scan_home", "scan_project"):
            with pytest.raises(NotSupportedError, match=rf"copilot does not support {method}"):
                _check_feature("copilot", method)
        _check_feature("copilot", "get_hook_spec")

        monkeypatch.setitem(spec, "capabilities", set(capabilities) - {"hooks"})
        for method in ("get_hook_spec", "detect_hooks", "generate_hook_config"):
            with pytest.raises(NotSupportedError, match=rf"copilot does not support {method}"):
                _check_feature("copilot", method)
        _check_feature("copilot", "scan_home")
