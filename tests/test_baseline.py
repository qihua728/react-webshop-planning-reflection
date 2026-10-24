from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from webshop_baseline.agent import ReActAgent, TrajectoryStep
from webshop_baseline.agents.factory import AGENT_TYPES
from webshop_baseline.agents.full import PlanReflectAgent
from webshop_baseline.agents.planning import PlanningAgent
from webshop_baseline.agents.reflection import ReflectionAgent
from webshop_baseline.config import (
    AgentConfig,
    BaselineConfig,
    ExperimentConfig,
    LLMConfig,
    WebShopConfig,
    load_config,
)
from webshop_baseline.env import InvalidAction, WebShopError, normalize_action
from webshop_baseline.env import StepResult
from webshop_baseline.evaluate import read_jsonl, summarize
from webshop_baseline.runner import run_experiment


class ActionTests(unittest.TestCase):
    def test_normalizes_action_prefix(self) -> None:
        self.assertEqual(normalize_action("Action: search[black shoes]"), "search[black shoes]")

    def test_rejects_free_text(self) -> None:
        with self.assertRaises(InvalidAction):
            normalize_action("I should search for shoes")


class EvaluationTests(unittest.TestCase):
    def test_summary(self) -> None:
        records = [
            {
                "method": "react_baseline",
                "status": "completed",
                "reward": 1.0,
                "success": True,
                "steps": 4,
                "llm_calls": 4,
                "invalid_actions": 0,
            },
            {
                "method": "react_baseline",
                "status": "max_steps",
                "reward": 0.5,
                "success": False,
                "steps": 6,
                "llm_calls": 6,
                "invalid_actions": 1,
            },
        ]
        result = summarize(records, "react_baseline")
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["average_reward"], 0.75)
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["invalid_action_rate"], 0.1)
        self.assertEqual(result["invalid_auxiliary_outputs"], 0)
        self.assertEqual(result["auxiliary_format_error_rate"], 0.0)

    def test_summary_reports_auxiliary_format_errors(self) -> None:
        records = [
            {
                "method": "react_reflection",
                "status": "completed",
                "reward": 1.0,
                "success": True,
                "steps": 2,
                "action_calls": 2,
                "llm_calls": 4,
                "planning_calls": 0,
                "reflection_calls": 2,
                "invalid_auxiliary_outputs": 1,
                "invalid_actions": 0,
            }
        ]
        result = summarize(records, "react_reflection")
        self.assertEqual(result["invalid_auxiliary_outputs"], 1)
        self.assertEqual(result["auxiliary_format_error_rate"], 0.5)

    def test_summary_detects_format_errors_in_legacy_records(self) -> None:
        records = [
            {
                "method": "react_plan_reflect",
                "status": "completed",
                "reward": 1.0,
                "success": True,
                "steps": 1,
                "action_calls": 1,
                "llm_calls": 3,
                "planning_calls": 1,
                "reflection_calls": 1,
                "plan": (
                    "plan[plan[constraints: item; search: item; "
                    "verify: match; fallback: broad]]"
                ),
                "reflections": [
                    {
                        "text": "reflection[problem: stuck; evidence: repeated click]",
                    }
                ],
                "invalid_actions": 0,
            }
        ]
        result = summarize(records, "react_plan_reflect")
        self.assertEqual(result["invalid_auxiliary_outputs"], 2)
        self.assertEqual(result["auxiliary_format_error_rate"], 1.0)

    def test_read_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(json.dumps({"reward": 1}) + "\n", encoding="utf-8")
            self.assertEqual(read_jsonl(path), [{"reward": 1}])


class _FakeEnvironment:
    def reset(self, task_id: str) -> StepResult:
        return StepResult("Instruction: buy the matching item\n[Search]", 0.0, False)

    def step(self, task_id: str, action: str) -> StepResult:
        if action == "search[item]":
            return StepResult("[ABC123] Item", 0.0, False)
        if action == "click[ABC123]":
            return StepResult("Your score: 1", 1.0, True)
        raise InvalidAction("unexpected action")


class _UnavailableEnvironment:
    def __init__(self, **_: object) -> None:
        pass

    def reset(self, task_id: str) -> StepResult:
        raise WebShopError(f"{task_id} unavailable")


class _FakeModel:
    def __init__(self) -> None:
        self.actions = iter(["search[item]", "click[ABC123]"])

    def generate_action(self, prompt: str) -> str:
        return next(self.actions)


class _AuxiliaryFakeModel:
    def __init__(self, actions: list[str]) -> None:
        self.actions = iter(actions)
        self.auxiliary_tags: list[str] = []

    def generate_action(self, prompt: str) -> str:
        return next(self.actions)

    def generate_text(self, prompt: str, output_tag: str) -> str:
        self.auxiliary_tags.append(output_tag)
        if output_tag == "plan":
            return "plan[constraints: item; search: item; verify: match; fallback: broader query]"
        return "reflection[problem: invalid click; evidence: rejected action; revised_strategy: search first]"


class AgentTests(unittest.TestCase):
    def test_agent_records_a_complete_trajectory(self) -> None:
        agent = ReActAgent(
            env=_FakeEnvironment(),  # type: ignore[arg-type]
            llm=_FakeModel(),
            demonstration="Example\nAction: search[example]",
            max_steps=3,
            max_prompt_chars=1000,
        )
        result = agent.run("fixed_0")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["reward"], 1.0)
        self.assertEqual(result["steps"], 2)
        self.assertEqual(len(result["trajectory"]), 2)
        self.assertEqual(result["planning_calls"], 0)
        self.assertEqual(result["reflection_calls"], 0)
        self.assertEqual(result["total_llm_calls"], 2)

    def test_planning_agent_records_plan_and_extra_call(self) -> None:
        model = _AuxiliaryFakeModel(["search[item]", "click[ABC123]"])
        agent = PlanningAgent(
            env=_FakeEnvironment(),  # type: ignore[arg-type]
            llm=model,  # type: ignore[arg-type]
            demonstration="Example\nAction: search[example]",
            planning_prompt="Plan from: {observation}",
            max_steps=3,
            max_prompt_chars=1000,
        )
        result = agent.run("fixed_0")
        self.assertTrue(result["plan"].startswith("plan["))
        self.assertEqual(result["planning_calls"], 1)
        self.assertEqual(result["reflection_calls"], 0)
        self.assertEqual(result["total_llm_calls"], 3)
        self.assertTrue(result["plan_format_valid"])
        self.assertEqual(result["invalid_auxiliary_outputs"], 0)
        self.assertEqual(model.auxiliary_tags, ["plan"])

    def test_reflection_agent_uses_invalid_action_trigger(self) -> None:
        model = _AuxiliaryFakeModel(["click[bad]", "search[item]", "click[ABC123]"])
        agent = ReflectionAgent(
            env=_FakeEnvironment(),  # type: ignore[arg-type]
            llm=model,  # type: ignore[arg-type]
            demonstration="Example\nAction: search[example]",
            reflection_prompt=(
                "Instruction: {observation}\nPlan: {plan}\nTriggers: {triggers}\n"
                "Recent: {recent_trajectory}"
            ),
            max_steps=4,
            max_prompt_chars=2000,
        )
        result = agent.run("fixed_0")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["invalid_actions"], 1)
        self.assertEqual(result["reflection_calls"], 1)
        self.assertEqual(result["total_llm_calls"], 4)
        self.assertIn("invalid_action", result["reflections"][0]["triggers"])
        self.assertTrue(result["reflections"][0]["format_valid"])
        self.assertEqual(result["invalid_auxiliary_outputs"], 0)
        self.assertEqual(model.auxiliary_tags, ["reflection"])

    def test_full_agent_combines_plan_and_reflection(self) -> None:
        model = _AuxiliaryFakeModel(["click[bad]", "search[item]", "click[ABC123]"])
        agent = PlanReflectAgent(
            env=_FakeEnvironment(),  # type: ignore[arg-type]
            llm=model,  # type: ignore[arg-type]
            demonstration="Example\nAction: search[example]",
            planning_prompt="Plan from: {observation}",
            reflection_prompt=(
                "Instruction: {observation}\nPlan: {plan}\nTriggers: {triggers}\n"
                "Recent: {recent_trajectory}"
            ),
            max_steps=4,
            max_prompt_chars=2000,
        )
        result = agent.run("fixed_0")
        self.assertEqual(result["planning_calls"], 1)
        self.assertEqual(result["reflection_calls"], 1)
        self.assertEqual(result["total_llm_calls"], 5)
        self.assertEqual(model.auxiliary_tags, ["plan", "reflection"])

    def test_auxiliary_normalizer_removes_nested_tag(self) -> None:
        normalized, valid = PlanningAgent._normalize_auxiliary(
            (
                "plan[plan[constraints: item; search: item; "
                "verify: exact match; fallback: broader query]]"
            ),
            "plan",
        )
        self.assertEqual(
            normalized,
            "plan[constraints: item; search: item; verify: exact match; fallback: broader query]",
        )
        self.assertTrue(valid)

    def test_auxiliary_normalizer_flags_missing_fields(self) -> None:
        normalized, valid = ReflectionAgent._normalize_auxiliary(
            "reflection[problem: stuck; evidence: repeated click]",
            "reflection",
        )
        self.assertEqual(
            normalized,
            "reflection[problem: stuck; evidence: repeated click]",
        )
        self.assertFalse(valid)

    def test_page_guidance_blocks_search_away_from_initial_page(self) -> None:
        guidance = ReActAgent._page_guidance(
            "[Back to Search]\n[ABC123]\nItem",
            [],
        )
        self.assertIn("search[...] is not valid on this page", guidance)
        self.assertIn("exact target is visibly enclosed in brackets", guidance)

    def test_page_guidance_warns_about_repeated_clicks(self) -> None:
        trajectory = [
            {
                "step": 1,
                "action": "click[small]",
                "observation": "clicked",
                "reward": 0.0,
                "done": False,
                "invalid_action": False,
                "latency_seconds": 0.0,
            },
            {
                "step": 2,
                "action": "click[small]",
                "observation": "clicked",
                "reward": 0.0,
                "done": False,
                "invalid_action": False,
                "latency_seconds": 0.0,
            },
        ]
        guidance = ReActAgent._page_guidance(
            "[small]\n[Buy Now]",
            [TrajectoryStep(**item) for item in trajectory],
        )
        self.assertIn("repeated without progress", guidance)


class RunnerTests(unittest.TestCase):
    def test_preflight_failure_writes_no_results_and_skips_llm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prompts").mkdir()
            (root / "prompts" / "react.txt").write_text(
                "Example\nAction: search[item]",
                encoding="utf-8",
            )
            config = BaselineConfig(
                method="react_baseline",
                webshop=WebShopConfig(base_url="http://localhost:3000"),
                llm=LLMConfig(
                    provider="deepseek",
                    api_key_env="DEEPSEEK_API_KEY",
                    base_url="https://api.deepseek.com",
                    model="test-model",
                ),
                agent=AgentConfig(prompt_file="prompts/react.txt"),
                experiment=ExperimentConfig(
                    results_file="results/test.jsonl",
                    summary_file="results/test_summary.json",
                ),
                project_root=root,
            )
            with (
                patch("webshop_baseline.runner.WebShopEnv", _UnavailableEnvironment),
                patch("webshop_baseline.runner.DeepSeekClient") as client,
            ):
                with self.assertRaisesRegex(WebShopError, "preflight failed"):
                    run_experiment(config)
                client.assert_not_called()
            self.assertFalse((root / "results" / "test.jsonl").exists())


class ExperimentLayoutTests(unittest.TestCase):
    def test_factory_exposes_four_partitioned_agents(self) -> None:
        self.assertIs(AGENT_TYPES["react_baseline"], ReActAgent)
        self.assertIs(AGENT_TYPES["react_planning"], PlanningAgent)
        self.assertIs(AGENT_TYPES["react_reflection"], ReflectionAgent)
        self.assertIs(AGENT_TYPES["react_plan_reflect"], PlanReflectAgent)

    def test_all_experiment_configs_load(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config_names = ["baseline_frozen.yaml", "planning.yaml", "reflection.yaml", "full.yaml"]
        methods = [
            load_config(project_root / "configs" / name).method for name in config_names
        ]
        self.assertEqual(
            methods,
            ["react_baseline", "react_planning", "react_reflection", "react_plan_reflect"],
        )
        for name in config_names:
            config = load_config(project_root / "configs" / name)
            self.assertTrue(config.resolve(config.agent.prompt_file).is_file())
            self.assertEqual(
                config.webshop.git_commit,
                "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd",
            )
            self.assertGreater(config.llm.auxiliary_max_tokens, config.llm.max_tokens)
            if config.agent.planning_prompt_file:
                self.assertTrue(config.resolve(config.agent.planning_prompt_file).is_file())
            if config.agent.reflection_prompt_file:
                self.assertTrue(config.resolve(config.agent.reflection_prompt_file).is_file())

    def test_formal_configs_use_full_held_out_split(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config_names = [
            "formal_baseline.yaml",
            "formal_planning.yaml",
            "formal_reflection.yaml",
            "formal_full.yaml",
        ]
        for name in config_names:
            config = load_config(project_root / "configs" / name)
            self.assertEqual(config.experiment.start_index, 10)
            self.assertEqual(config.experiment.count, 100)
            self.assertEqual(config.experiment.dataset_mode, "full")
            self.assertTrue(config.experiment.dataset_version)
            self.assertTrue(config.experiment.results_file.startswith("results/formal/"))
            self.assertEqual(config.webshop.patch_version, "full-streaming-ijson-v1")


if __name__ == "__main__":
    unittest.main()
