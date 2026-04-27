import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from src.agents.base import Agent, Transition
from src.agents.registry import AGENTS, register_agent


class TestAgentABC(unittest.TestCase):

    def test_cannot_instantiate_abstract_agent(self):
        with self.assertRaises(TypeError) as ctx:
            Agent()
        self.assertIn("abstract", str(ctx.exception).lower())

    def test_subclass_missing_methods_cannot_instantiate(self):
        class IncompleteAgent(Agent):
            def act(self, obs, *, greedy=False):
                return 0

        with self.assertRaises(TypeError):
            IncompleteAgent()

    def test_subclass_complete_can_instantiate(self):
        class CompleteAgent(Agent):
            def act(self, obs, *, greedy=False): return 0
            def observe(self, transition): pass
            def learn(self): return {}
            def save(self, path): pass
            def load(self, path): pass

        agent = CompleteAgent()
        self.assertIsInstance(agent, Agent)


class TestRegistry(unittest.TestCase):

    def test_random_not_registered_yet(self):
        if "random" in AGENTS:
            self.skipTest("random agent already registered (T1.1 done)")
        with self.assertRaises(KeyError):
            _ = AGENTS["random"]

    def test_decorator_registers_class(self):
        @register_agent("test_dummy_agent")
        class DummyAgent(Agent):
            def act(self, obs, *, greedy=False): return 0
            def observe(self, transition): pass
            def learn(self): return {}
            def save(self, path): pass
            def load(self, path): pass

        try:
            self.assertIn("test_dummy_agent", AGENTS)
            self.assertIs(AGENTS["test_dummy_agent"], DummyAgent)
        finally:
            AGENTS.pop("test_dummy_agent", None)  # cleanup

    def test_decorator_rejects_non_agent_class(self):
        with self.assertRaises(TypeError):
            @register_agent("not_an_agent")
            class NotAnAgent:
                pass

    def test_decorator_rejects_duplicate_name(self):
        @register_agent("test_dup_a")
        class A(Agent):
            def act(self, obs, *, greedy=False): return 0
            def observe(self, transition): pass
            def learn(self): return {}
            def save(self, path): pass
            def load(self, path): pass

        try:
            with self.assertRaises(ValueError):
                @register_agent("test_dup_a")
                class B(Agent):
                    def act(self, obs, *, greedy=False): return 0
                    def observe(self, transition): pass
                    def learn(self): return {}
                    def save(self, path): pass
                    def load(self, path): pass
        finally:
            AGENTS.pop("test_dup_a", None)


class TestTransition(unittest.TestCase):

    def test_transition_is_frozen(self):
        t = Transition(
            obs=np.zeros(8, dtype=np.float32),
            action=0,
            reward=1.0,
            next_obs=np.zeros(8, dtype=np.float32),
            terminated=False,
            truncated=False,
        )
        with self.assertRaises(Exception):
            t.action = 1

if __name__ == "__main__":
    unittest.main()