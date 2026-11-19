"""Progressive skill loading.

Paige does not read every skill's full instructions up front. It may load a
skill automatically if relevant, then still choose not to run it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Skill(BaseModel):
    id: str
    name: str
    description: str
    triggers: list[str] = Field(default_factory=list)
    instructions: str
    scripts: list[str] = Field(default_factory=list)


class SkillRegistry:
    def __init__(self, skills: list[Skill] | None = None) -> None:
        self._skills = {s.id: s for s in (skills or [])}
        self._loaded: set[str] = set()

    def register(self, skill: Skill) -> None:
        self._skills[skill.id] = skill

    def catalog(self) -> list[dict[str, str]]:
        """Lightweight index only — not full instructions."""
        return [
            {"id": s.id, "name": s.name, "description": s.description}
            for s in self._skills.values()
        ]

    def select_candidates(self, text: str, limit: int = 3) -> list[Skill]:
        lowered = text.lower()
        scored: list[tuple[int, Skill]] = []
        for skill in self._skills.values():
            score = 0
            for trig in skill.triggers:
                if trig.lower() in lowered:
                    score += 2
            if skill.name.lower() in lowered:
                score += 1
            if score:
                scored.append((score, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:limit]]

    def load(self, skill_id: str) -> Skill | None:
        skill = self._skills.get(skill_id)
        if skill:
            self._loaded.add(skill_id)
        return skill

    def loaded_instructions(self) -> list[str]:
        return [self._skills[sid].instructions for sid in self._loaded if sid in self._skills]


DEFAULT_SKILLS = [
    Skill(
        id="kafka-consumer-lag",
        name="Kafka Consumer Lag",
        description="Triage consumer lag and offset / poison-pill issues.",
        triggers=["consumer lag", "kafka", "offset", "lag"],
        instructions=(
            "1. Confirm consumer group lag metric spike.\n"
            "2. Check for recent deploy of the consumer.\n"
            "3. Search logs for deserialization / poison messages.\n"
            "4. If duplicate records via REST race, follow cleanup playbook.\n"
            "5. Do not restart brokers without approval."
        ),
    ),
    Skill(
        id="dns-resolution",
        name="DNS Resolution Failure",
        description="Investigate intermittent DNS / NXDOMAIN failures.",
        triggers=["dns", "nxdomain", "servfail", "resolution"],
        instructions=(
            "1. Verify resolver errors in app logs.\n"
            "2. Check upstream dependency hostname changes.\n"
            "3. Correlate with recent network / DNS config changes.\n"
            "4. Prefer evidence from failing client, not just DNS server metrics."
        ),
    ),
    Skill(
        id="bad-deploy",
        name="Bad Deploy",
        description="Correlate incidents with recent deployments.",
        triggers=["deploy", "release", "rollback", "version"],
        instructions=(
            "1. List change events in the incident window.\n"
            "2. Diff error fingerprints before/after deploy.\n"
            "3. Search commits touching the failing component.\n"
            "4. Recommend rollback only when direct evidence ties the deploy."
        ),
    ),
]
