from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import os
import re
import threading
from typing import Any

from src.agents.base import BaseAgent
from src.contribution.schemas import (
    GroupContext,
    GroupContributionTable,
    ReviewFragment,
    TypeSynthesis,
    TypeSynthesisParagraph,
    TypeSynthesisSentence,
)
from src.generation.final_review_integrator import FinalReviewIntegrator
from src.generation.fragment_writer import FragmentWriter
from src.generation.type_synthesis_integrator import TypeSynthesisIntegrator
from src.organization.group_fragment_context_builder import GroupFragmentContextBuilder
from src.utils.io import read_json, write_json, write_text
from src.utils.llm_client import LLMClient


class SynthesisAgent(BaseAgent):
    name = "SynthesisAgent"
    paper_id_re = re.compile(r"(?<![A-Za-z0-9_])P\d{3,}(?!\d)")
    type_synthesis_max_chars = 1600
    type_synthesis_max_paragraphs = 4
    issue_type_synthesis_max_chars = 1600
    issue_type_synthesis_max_paragraphs = 4
    type_sentence_max_chars = 220
    type_sentence_max_citations = 3
    group_fragment_max_chars = 900
    final_section_titles = {
        "issue": ("研究关注与议题分布", "研究关注与研究议题分布"),
        "problem": ("核心研究问题",),
        "method": ("主要方法路径",),
        "finding": ("研究发现与认识",),
    }
    enumerative_sequences = (
        ("一是", "二是"),
        ("首先", "其次"),
        ("第一类", "第二类"),
        ("一类", "另一类"),
        ("第一方面", "第二方面"),
    )
    issue_citation_dump_phrases = (
        "相关研究涉及",
        "相关研究包括",
        "相关支撑文献包括",
        "支撑文献包括",
        "支撑论文包括",
    )
    _type_response_local = threading.local()

    def run(
        self,
        query: str,
        tables: list[GroupContributionTable],
        prompts: dict[str, str] | None = None,
        group_contexts: list[GroupContext] | None = None,
    ):
        fragments = self.generate_fragments(query, tables, prompts=prompts, group_contexts=group_contexts)
        type_syntheses = self.generate_type_syntheses(query, fragments, prompts=prompts)
        review = self.integrate(query, type_syntheses, prompts=prompts)
        return fragments, review

    def generate_fragments(
        self,
        query: str,
        tables: list[GroupContributionTable],
        prompts: dict[str, str] | None = None,
        group_contexts: list[GroupContext] | None = None,
        validation_feedback_by_group: dict[str, str] | None = None,
        revision_contexts_by_group: dict[str, dict] | None = None,
    ) -> list[ReviewFragment]:
        task = self.state.start_task(self.name, ["group_contexts.json", f"query:{query}"])
        selected_tables = self._select_tables(tables)
        contexts_by_group = {item.group_id: item for item in group_contexts or []}
        llm_client = LLMClient()
        prompts = prompts or {}
        writer = FragmentWriter()
        workers = self._worker_count("LLM_SYNTHESIS_WORKERS", default=10)

        def write_one(table: GroupContributionTable) -> tuple[ReviewFragment, int, dict[str, str] | None]:
            revision_context = (revision_contexts_by_group or {}).get(table.group_id)
            if not llm_client.enabled:
                if revision_context is not None:
                    raise RuntimeError("Feedback-based revision requires a configured LLM.")
                return writer.write(table), 0, None
            validation_feedback = (validation_feedback_by_group or {}).get(table.group_id, "")
            attempt_errors: list[str] = []
            for attempt in range(1, 3):
                try:
                    # A client per worker prevents future client-level state
                    # changes from becoming a cross-thread concern.
                    fragment = self._write_fragment_llm(
                        LLMClient(),
                        prompts,
                        table,
                        contexts_by_group.get(table.group_id),
                        validation_feedback=validation_feedback,
                        revision_context=revision_context,
                    )
                    return fragment, attempt, None
                except Exception as exc:
                    validation_feedback = str(exc)
                    attempt_errors.append(validation_feedback)
            if revision_context is not None:
                raise RuntimeError(
                    f"Feedback revision failed for {table.group_id}: " + " | ".join(attempt_errors)
                )
            return (
                writer.write(table),
                len(attempt_errors),
                {
                    "stage": "fragment",
                    "group_id": table.group_id,
                    "error": " | ".join(
                        f"attempt {index}: {value}" for index, value in enumerate(attempt_errors, 1)
                    ),
                },
            )

        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(selected_tables)))) as executor:
            outcomes = list(executor.map(write_one, selected_tables))
        fragments = [fragment for fragment, _, _ in outcomes]
        fragment_attempts = {
            table.group_id: attempts
            for table, (_, attempts, _) in zip(selected_tables, outcomes)
            if attempts
        }
        errors = [error for _, _, error in outcomes if error is not None]
        llm_fragment_count = sum(1 for _, attempts, error in outcomes if attempts and error is None)

        llm_status = {
            "llm_enabled": llm_client.enabled,
            "llm_model": llm_client.model,
            "worker_count": min(workers, len(selected_tables)),
            "fragment_mode": "llm" if llm_fragment_count else "rule_based",
            "llm_fragment_count": llm_fragment_count,
            "fragment_count": len(fragments),
            "fragment_attempts": fragment_attempts,
            "fallback_errors": errors,
        }
        outputs = [
            write_json(self.output_dir / "review_fragments.json", fragments),
            write_json(self.output_dir / "synthesis_llm_status.json", llm_status),
        ]
        self.state.finish_task(task, outputs, issues=errors)
        self.state.add_message(self.name, "Coordinator", "group_fragments_completed", llm_status, task.task_id)
        return fragments

    @staticmethod
    def _worker_count(variable: str, default: int) -> int:
        try:
            return max(1, int(os.getenv(variable) or default))
        except ValueError:
            return default

    @staticmethod
    def _type_integration_timeout_seconds() -> int:
        """Return the per-request timeout used only for type integration."""
        try:
            return max(1, int(os.getenv("LLM_TYPE_INTEGRATION_TIMEOUT_SECONDS") or 120))
        except ValueError:
            return 120

    def _type_integration_llm_client(self) -> LLMClient:
        return LLMClient(timeout_seconds=self._type_integration_timeout_seconds())

    def generate_type_syntheses(
        self,
        query: str,
        fragments: list[ReviewFragment],
        prompts: dict[str, str] | None = None,
    ) -> list[TypeSynthesis]:
        task = self.state.start_task(self.name, ["review_fragments.json", f"query:{query}"])
        prompts = prompts or {}
        llm_client = self._type_integration_llm_client()
        group_types = ["issue", "problem", "method", "finding"]
        source_by_type = {
            group_type: [fragment for fragment in fragments if fragment.group_type == group_type]
            for group_type in group_types
        }
        selected_types = [group_type for group_type in group_types if source_by_type[group_type]]
        workers = self._worker_count("LLM_TYPE_INTEGRATION_WORKERS", default=1)

        # Each type has independent source fragments and an independent LLM
        # client. map() preserves the fixed issue/problem/method/finding order.
        def synthesize_one(group_type: str) -> tuple[TypeSynthesis, str, int, list[dict[str, str | int]], dict | None]:
            return self._synthesize_one_type(
                query,
                group_type,
                source_by_type[group_type],
                prompts,
            )

        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(selected_types)))) as executor:
            outcomes = list(executor.map(synthesize_one, selected_types))

        type_syntheses = [item[0] for item in outcomes]
        type_modes = {group_type: mode for group_type, (_, mode, _, _, _) in zip(selected_types, outcomes)}
        type_attempts = {group_type: attempts for group_type, (_, _, attempts, _, _) in zip(selected_types, outcomes)}
        validation_retry_errors = [error for _, _, _, retry_errors, _ in outcomes for error in retry_errors]
        errors = [error for _, _, _, _, error in outcomes if error is not None]

        status = {
            "llm_enabled": llm_client.enabled,
            "llm_model": llm_client.model,
            "worker_count": min(workers, len(selected_types)),
            "type_modes": type_modes,
            "type_attempts": type_attempts,
            "validation_retry_errors": validation_retry_errors,
            "synthesis_quality_issues": [
                issue
                for item in type_syntheses
                for issue in item.quality_issues
            ],
            "type_synthesis_count": len(type_syntheses),
            "source_fragment_count": len(fragments),
            "covered_fragment_count": sum(len(item.covered_fragment_ids) for item in type_syntheses),
            "uncovered_fragment_ids": [
                fragment_id
                for item in type_syntheses
                for fragment_id in item.uncovered_fragment_ids
            ],
            "fallback_errors": errors,
        }
        outputs = [
            write_json(self.output_dir / "type_syntheses.json", type_syntheses),
            write_json(self.output_dir / "type_integration_llm_status.json", status),
        ]
        self.state.finish_task(task, outputs, issues=errors)
        self.state.add_message(self.name, "Coordinator", "type_synthesis_completed", status, task.task_id)
        return type_syntheses

    def _synthesize_one_type(
        self,
        query: str,
        group_type: str,
        source_fragments: list[ReviewFragment],
        prompts: dict[str, str],
    ) -> tuple[TypeSynthesis, str, int, list[dict[str, str | int]], dict | None]:
        fallback = TypeSynthesisIntegrator()
        if len(source_fragments) == 1:
            return fallback.integrate(group_type, source_fragments), "passthrough", 0, [], None

        llm_client = self._type_integration_llm_client()
        if not llm_client.enabled:
            return fallback.integrate(group_type, source_fragments), "rule_based", 0, [], None

        retry_errors: list[dict[str, str | int]] = []
        repair_data: dict[str, Any] | None = None
        attempt_errors: list[str] = []
        maximum_attempts = 3
        for attempt in range(1, maximum_attempts + 1):
            self._type_response_local.data = None
            try:
                synthesis = (
                    self._repair_type_integration_llm(llm_client, group_type, source_fragments, repair_data, attempt_errors[-1])
                    if repair_data is not None
                    else self._integrate_type_llm(llm_client, prompts, query, group_type, source_fragments)
                )
                mode = "llm" if attempt == 1 else "llm_after_format_repair"
                if synthesis.quality_issues:
                    mode += "_with_warnings"
                return synthesis, mode, attempt, retry_errors, None
            except Exception as exc:
                message = str(exc)
                attempt_errors.append(message)
                retry_errors.append({"group_type": group_type, "attempt": attempt, "error": message})
                raw_data = getattr(self._type_response_local, "data", None)
                # A structured response that only violates the strict contract
                # can be repaired without resending every source fragment.
                repair_data = raw_data if isinstance(raw_data, dict) else None

        error = {
            "stage": "type_integration",
            "group_type": group_type,
            "error": " | ".join(f"attempt {index}: {message}" for index, message in enumerate(attempt_errors, start=1)),
        }
        return fallback.integrate(group_type, source_fragments), "rule_based", maximum_attempts, retry_errors, error

    def regenerate_group_fragments(
        self,
        query: str,
        tables: list[GroupContributionTable],
        current_fragments: list[ReviewFragment],
        target_group_ids: set[str],
        prompts: dict[str, str] | None = None,
        group_contexts: list[GroupContext] | None = None,
        validation_feedback_by_group: dict[str, str] | None = None,
        revision_contexts_by_group: dict[str, dict] | None = None,
    ) -> list[ReviewFragment]:
        """Rewrite only groups changed by an incremental retrieval round."""
        if not target_group_ids:
            return current_fragments
        target_tables = [table for table in tables if table.group_id in target_group_ids]
        regenerated = self.generate_fragments(
            query,
            target_tables,
            prompts=prompts,
            group_contexts=group_contexts,
            validation_feedback_by_group=validation_feedback_by_group,
            revision_contexts_by_group=revision_contexts_by_group,
        )
        fragments_by_group = {fragment.group_id: fragment for fragment in current_fragments}
        fragments_by_group.update({fragment.group_id: fragment for fragment in regenerated})
        ordered = [
            fragments_by_group[table.group_id]
            for table in self._select_tables(tables)
            if table.group_id in fragments_by_group
        ]
        write_json(self.output_dir / "review_fragments.json", ordered)
        status_path = self.output_dir / "synthesis_llm_status.json"
        status = read_json(status_path, default={}) or {}
        status.update(
            {
                "fragment_count": len(ordered),
                "regenerated_group_ids": sorted(target_group_ids),
                "regenerated_fragment_count": len(regenerated),
            }
        )
        write_json(status_path, status)
        return ordered

    def integrate(self, query: str, type_syntheses: list[TypeSynthesis], prompts: dict[str, str] | None = None) -> str:
        task = self.state.start_task(self.name, ["type_syntheses.json", f"query:{query}"])
        llm_client = LLMClient()
        errors: list[dict[str, str]] = []
        integration_mode = "deterministic_assembly"
        review = FinalReviewIntegrator().integrate(query, type_syntheses)
        final_quality_issues = self._final_section_citation_issues(review, type_syntheses)
        outputs = [
            write_text(self.output_dir / "final_review.md", review),
            write_json(
                self.output_dir / "final_integration_llm_status.json",
                {
                    "llm_enabled": llm_client.enabled,
                    "llm_model": llm_client.model,
                    "integration_mode": integration_mode,
                    "type_text_rewritten": False,
                    "covered_type_synthesis_ids": [item.type_synthesis_id for item in type_syntheses],
                    "uncovered_type_synthesis_ids": [],
                    "synthesis_quality_issues": final_quality_issues,
                    "fallback_errors": errors,
                },
            ),
        ]
        self.state.finish_task(task, outputs, issues=errors)
        self.state.add_message(self.name, "Coordinator", "final_review_integrated", {"integration_mode": integration_mode}, task.task_id)
        return review

    def _write_fragment_llm(
        self,
        llm_client: LLMClient,
        prompts: dict[str, str],
        table: GroupContributionTable,
        group_context: GroupContext | None,
        validation_feedback: str = "",
        revision_context: dict | None = None,
    ) -> ReviewFragment:
        prompt_name = (
            "other_group_synthesis"
            if table.group_role == "other"
            else f"{table.group_type}_group_synthesis"
        )
        system_prompt = (
            prompts["group_feedback_revision"] if revision_context is not None
            else (prompts.get(prompt_name) or "")
        )
        if table.group_role != "other" and (
            table.group_quality_flag != "normal" or len(table.contributions) <= 1
        ):
            system_prompt += "\n\n当前组证据较弱或仅含单篇论文；必须使用保守表述，不能推导领域共识或稳定趋势，不虚构跨文献比较。"
        if table.group_role == "other":
            system_prompt += (
                "\n\n程序硬约束：这是未形成稳定语义社区的单例汇总写作容器，不是共同主题组。"
                "只能选择与 review_topic 最相关且信息量最高的 3–5 个知识单元作简要补充；"
                "若不足 3 个则按实际数量处理。不得声称组内单元具有共同方向、共识或趋势。"
                "text 必须是一个不超过 900 个字符的紧凑自然段，分别简述选中的补充方向；"
                "不寻找共同关系，不输出 Markdown 或内部工程标识。"
            )
        else:
            system_prompt += (
                "\n\n程序优先约束：text 必须是一个紧凑自然段，围绕 review_topic 与组标签说明组内知识单元的"
                "共同关系和边界；不逐篇罗列，不输出 Markdown 或内部工程标识，长度不得超过 900 个字符。"
            )
        if revision_context is not None:
            if not all(revision_context.get(key) for key in (
                "current_draft", "information_gaps", "supplementary_papers"
            )):
                raise ValueError("Feedback revision requires the current draft, information gaps and supplementary papers.")
        if validation_feedback:
            system_prompt += (
                "\n\n上一次输出未通过验收，原因如下：\n"
                + validation_feedback
                + "\n请重新输出完整 JSON，并只修复上述问题。"
            )
        system_prompt += "\n\n请输出 JSON：{\"title\":\"...\",\"text\":\"...\",\"cited_papers\":[\"P001\"]}"
        payload = {
            "review_topic": group_context.review_topic if group_context is not None else "",
            "group_metadata": {
                "group_id": table.group_id,
                "group_type": table.group_type,
                "group_label": table.group_label,
                "group_summary": table.group_summary,
                "quality_flag": table.group_quality_flag,
                "group_role": table.group_role,
            },
            "synthesis_context_card": (
                GroupFragmentContextBuilder().build(group_context) if group_context is not None else {"units": []}
            )
        }
        if revision_context is not None:
            payload["feedback_revision"] = revision_context
        data = llm_client.chat_json(system_prompt, payload, temperature=0.0)
        allowed_papers = (
            {unit.paper_id for unit in group_context.units if unit.paper_id}
            if group_context is not None
            else {contribution.paper_id for contribution in table.contributions}
        )
        text = self._require_text(data.get("text"), table.group_id)
        if len(text) > self.group_fragment_max_chars:
            raise ValueError(
                f"Group synthesis is too long for {table.group_id}: {len(text)} characters; "
                f"maximum={self.group_fragment_max_chars}."
            )
        if len([part for part in re.split(r"\n\s*\n", text) if part.strip()]) > 1:
            raise ValueError(f"Group synthesis must contain one compact paragraph for {table.group_id}.")
        if table.group_type == "issue":
            self._validate_issue_citation_style(text)
        cited_in_text = set(self.paper_id_re.findall(text))
        unknown_papers = cited_in_text - allowed_papers
        if unknown_papers:
            raise ValueError(f"Group synthesis introduced paper_ids outside the target group: {sorted(unknown_papers)}.")
        if not cited_in_text:
            raise ValueError(f"Group synthesis omitted inline paper_id citations for {table.group_id}.")
        if table.group_role == "other" and len(cited_in_text) > 5:
            raise ValueError(
                f"Other-group synthesis cited {len(cited_in_text)} papers for {table.group_id}; maximum=5."
            )
        return ReviewFragment(
            fragment_id=f"FRAG-{table.group_id}",
            group_id=table.group_id,
            group_type=table.group_type,
            title=str(data.get("title") or table.group_label),
            text=text,
            cited_papers=sorted(cited_in_text),
            group_role=table.group_role,
        )

    def _integrate_type_llm(
        self,
        llm_client: LLMClient,
        prompts: dict[str, str],
        query: str,
        group_type: str,
        fragments: list[ReviewFragment],
        validation_feedback: str = "",
        response_data: dict[str, Any] | None = None,
    ) -> TypeSynthesis:
        prompt_name = f"{group_type}_type_integration"
        system_prompt = (
            (prompts.get(prompt_name) or "")
            + "\n\n请只输出 JSON："
            + '{"title":"...","paragraphs":['
            + '{"role":"synthesis|exposition|appraisal","relation_focus":"...","sentences":['
            + '{"text":"不含论文编号和括号引用的单句","source_fragment_ids":["FRAG-..."],'
            + '"cited_papers":["P001"]}]}],'
            + '"omitted_fragment_ids":[{"fragment_id":"FRAG-...","reason":"与已覆盖片段重复"}]}'
        )
        if any(fragment.group_role == "other" for fragment in fragments):
            system_prompt += (
                "\n\n输入中 group_role=other 的片段只是未形成稳定社区的补充材料。"
                "不得把它作为统一研究方向或主体分类；只提取其中确有增量的重要内容，"
                "在正文中最多用 1–2 句补充。若与正常语义组重复，可以在 omitted_fragment_ids "
                "中明确说明重复关系。"
            )
        if validation_feedback:
            system_prompt += (
                "\n\n上一次输出未通过程序验收，原因如下：\n"
                + validation_feedback
                + "\n请重新生成完整 JSON，不要只修补局部字段。每个 sentence 的 text 不得自行写 paper_id"
                + "或括号引用；cited_papers 只选择 1–3 篇最直接的代表性文献，source_fragment_ids 仅用于"
                + "覆盖追踪，不要求将其全部论文写进正文。"
            )
        serialized_fragments = (
            [
                {
                    "fragment_id": fragment.fragment_id,
                    "title": fragment.title,
                    "text": fragment.text,
                    **({"group_role": "other"} if fragment.group_role == "other" else {}),
                }
                for fragment in fragments
            ]
            if group_type == "issue"
            else [asdict(fragment) for fragment in fragments]
        )
        payload = {
            "review_topic": query,
            "group_type": group_type,
            "review_fragments": serialized_fragments,
        }
        data = response_data if response_data is not None else llm_client.chat_json(system_prompt, payload, temperature=0.0)
        self._type_response_local.data = data
        expected_fragment_ids = {fragment.fragment_id for fragment in fragments}
        # A source fragment may be intentionally compressed away only when the
        # model makes the omission explicit.  Silent omission is never treated
        # as coverage.
        omitted_reasons = self._parse_omitted_fragment_reasons(
            data.get("omitted_fragment_ids", data.get("uncovered_fragment_ids", []))
        )

        raw_paragraphs = data.get("paragraphs")
        if not isinstance(raw_paragraphs, list) or not raw_paragraphs:
            raise ValueError(f"Type integration returned no structured paragraphs for {group_type}.")
        minimum_paragraphs = 3 if len(fragments) >= 3 else 2 if len(fragments) >= 2 else 1
        if len(raw_paragraphs) < minimum_paragraphs:
            raise ValueError(
                f"Type integration over-compressed {len(fragments)} fragments into "
                f"{len(raw_paragraphs)} paragraph(s); minimum={minimum_paragraphs}."
            )
        quality_issues: list[dict] = []
        maximum_paragraphs = (
            self.issue_type_synthesis_max_paragraphs
            if group_type == "issue"
            else self.type_synthesis_max_paragraphs
        )
        if len(raw_paragraphs) > maximum_paragraphs:
            raise ValueError(
                f"Type integration produced {len(raw_paragraphs)} paragraphs; "
                f"maximum={maximum_paragraphs}."
            )

        fragments_by_id = {fragment.fragment_id: fragment for fragment in fragments}
        source_papers_by_fragment = {
            fragment.fragment_id: set(fragment.cited_papers)
            for fragment in fragments
        }
        paragraphs: list[TypeSynthesisParagraph] = []
        covered_in_paragraphs: set[str] = set()
        seen_citation_clusters: set[tuple[str, ...]] = set()
        sentence_starts: dict[str, int] = {}
        for index, raw in enumerate(raw_paragraphs, start=1):
            if not isinstance(raw, dict):
                raise ValueError(f"Type integration paragraph {index} is not a JSON object.")
            role = str(raw.get("role") or "").strip().lower()
            expected_role = "synthesis" if index == 1 else "appraisal" if index == len(raw_paragraphs) else "exposition"
            if role != expected_role:
                raise ValueError(
                    f"Type integration paragraph {index} must use role={expected_role}, got {role or 'missing'}."
                )
            relation_focus = str(raw.get("relation_focus") or "").strip()
            if not relation_focus:
                raise ValueError(f"Type integration paragraph {index} omitted relation_focus.")
            raw_sentences = raw.get("sentences")
            if not isinstance(raw_sentences, list) or not raw_sentences:
                raise ValueError(f"Type integration paragraph {index} returned no structured sentences.")
            sentences: list[TypeSynthesisSentence] = []
            for sentence_index, raw_sentence in enumerate(raw_sentences, start=1):
                if not isinstance(raw_sentence, dict):
                    raise ValueError(f"Type integration paragraph {index} sentence {sentence_index} is not an object.")
                sentence_text = self._require_text(
                    raw_sentence.get("text"), f"TYPE-{group_type.upper()}-P{index:02d}-S{sentence_index:02d}"
                )
                if self.paper_id_re.search(sentence_text) or re.search(r"[（(][^（）()]*[）)]", sentence_text):
                    raise ValueError(
                        f"Type integration paragraph {index} sentence {sentence_index} must not embed citations in text."
                    )
                if len(sentence_text) > self.type_sentence_max_chars:
                    raise ValueError(
                        f"Type integration paragraph {index} sentence {sentence_index} is too long: "
                        f"{len(sentence_text)} characters; maximum={self.type_sentence_max_chars}."
                    )
                start = re.sub(r"[，。；：、].*$", "", sentence_text).strip()
                if start:
                    sentence_starts[start] = sentence_starts.get(start, 0) + 1
                    if sentence_starts[start] > 1 and len(start) >= 4:
                        raise ValueError(f"Type integration repeats the same sentence opening: {start}.")
                source_fragment_ids = list(
                    dict.fromkeys(str(value) for value in raw_sentence.get("source_fragment_ids", []) if str(value).strip())
                )
                if not source_fragment_ids:
                    raise ValueError(f"Type integration paragraph {index} sentence {sentence_index} omitted source_fragment_ids.")
                unknown_fragments = set(source_fragment_ids) - expected_fragment_ids
                if unknown_fragments:
                    raise ValueError(
                        f"Type integration paragraph {index} sentence {sentence_index} has invalid source fragments: "
                        f"{sorted(unknown_fragments)}."
                    )
                cited_papers = list(
                    dict.fromkeys(str(value) for value in raw_sentence.get("cited_papers", []) if str(value).strip())
                )
                if not cited_papers:
                    cited_papers = list(
                        dict.fromkeys(
                            paper_id
                            for source_id in source_fragment_ids
                            for paper_id in fragments_by_id[source_id].cited_papers
                        )
                    )[: self.type_sentence_max_citations]
                    if cited_papers:
                        quality_issues.append(
                            {
                                "issue_type": "citations_auto_selected_from_declared_sources",
                                "group_type": group_type,
                                "paragraph_index": index,
                                "sentence_index": sentence_index,
                                "source_fragment_ids": source_fragment_ids,
                                "reason": "The LLM omitted cited_papers but explicitly mapped the sentence to source fragments; representative citations were selected from those sources.",
                            }
                        )
                if len(cited_papers) > self.type_sentence_max_citations:
                    cited_papers = cited_papers[: self.type_sentence_max_citations]
                    quality_issues.append(
                        {
                            "issue_type": "citation_cluster_trimmed",
                            "group_type": group_type,
                            "paragraph_index": index,
                            "sentence_index": sentence_index,
                            "reason": "The model returned too many citations for one claim; only three representative papers were retained.",
                        }
                    )
                if not cited_papers:
                    raise ValueError(
                        f"Type integration paragraph {index} sentence {sentence_index} must cite 1–"
                        f"{self.type_sentence_max_citations} representative papers."
                    )
                allowed_papers = set().union(*(source_papers_by_fragment[source_id] for source_id in source_fragment_ids))
                unknown_papers = set(cited_papers) - allowed_papers
                if unknown_papers:
                    cited_papers = [paper_id for paper_id in cited_papers if paper_id in allowed_papers]
                    quality_issues.append(
                        {
                            "issue_type": "citation_outside_declared_sources_removed",
                            "group_type": group_type,
                            "paragraph_index": index,
                            "sentence_index": sentence_index,
                            "removed_papers": sorted(unknown_papers),
                            "reason": "The model cited papers outside its declared source fragments; those citations were removed.",
                        }
                    )
                    if not cited_papers:
                        cited_papers = list(dict.fromkeys(
                            paper_id
                            for source_id in source_fragment_ids
                            for paper_id in fragments_by_id[source_id].cited_papers
                        ))[: self.type_sentence_max_citations]
                citation_cluster = tuple(sorted(cited_papers))
                # A repeated multi-paper cluster reads like a citation dump.
                # Keep a subset of the model-selected evidence before accepting
                # a duplicate; this never adds a paper the model did not select.
                original_cluster = citation_cluster
                while citation_cluster in seen_citation_clusters and len(cited_papers) > 1:
                    cited_papers = cited_papers[:-1]
                    citation_cluster = tuple(sorted(cited_papers))
                if citation_cluster != original_cluster:
                    quality_issues.append(
                        {
                            "issue_type": "citation_cluster_shortened_to_avoid_repetition",
                            "group_type": group_type,
                            "paragraph_index": index,
                            "sentence_index": sentence_index,
                            "original_cited_papers": list(original_cluster),
                            "cited_papers": list(citation_cluster),
                            "reason": "A repeated multi-paper citation cluster was shortened without adding new evidence.",
                        }
                    )
                if citation_cluster in seen_citation_clusters:
                    quality_issues.append(
                        {
                            "issue_type": "repeated_citation_cluster",
                            "group_type": group_type,
                            "paragraph_index": index,
                            "sentence_index": sentence_index,
                            "cited_papers": list(citation_cluster),
                            "reason": "A single-paper citation was necessarily reused across distinct claims; retained for factual traceability and flagged for editorial review.",
                        }
                    )
                seen_citation_clusters.add(citation_cluster)
                sentences.append(TypeSynthesisSentence(sentence_text, source_fragment_ids, cited_papers))
            paragraph_text = "".join(self._render_type_sentence(sentence) for sentence in sentences)
            source_fragment_ids = list(dict.fromkeys(source_id for sentence in sentences for source_id in sentence.source_fragment_ids))
            cited_in_paragraph = sorted({paper_id for sentence in sentences for paper_id in sentence.cited_papers})
            paragraphs.append(
                TypeSynthesisParagraph(
                    relation_focus=relation_focus,
                    text=paragraph_text,
                    source_fragment_ids=source_fragment_ids,
                    cited_papers=cited_in_paragraph,
                    role=role,
                    sentences=sentences,
                )
            )
            covered_in_paragraphs.update(source_fragment_ids)

        unknown_omitted = set(omitted_reasons) - expected_fragment_ids
        if unknown_omitted:
            for fragment_id in unknown_omitted:
                omitted_reasons.pop(fragment_id, None)
            quality_issues.append(
                {
                    "issue_type": "unknown_omission_metadata_ignored",
                    "group_type": group_type,
                    "source_fragment_ids": sorted(unknown_omitted),
                    "reason": "The model returned omission identifiers that are not input fragments; they were ignored rather than treated as coverage.",
                }
            )
        overlap = covered_in_paragraphs & set(omitted_reasons)
        if overlap:
            for fragment_id in overlap:
                omitted_reasons.pop(fragment_id, None)
            quality_issues.append(
                {
                    "issue_type": "redundant_omission_metadata_ignored",
                    "group_type": group_type,
                    "source_fragment_ids": sorted(overlap),
                    "reason": "A source mapped in prose was also listed as omitted; the explicit prose mapping takes precedence.",
                }
            )
        missing = sorted(expected_fragment_ids - covered_in_paragraphs - set(omitted_reasons))
        if missing:
            raise ValueError(
                f"Type integration did not map every source fragment for {group_type}; "
                f"missing={missing}."
            )
        if omitted_reasons:
            quality_issues.append(
                {
                    "issue_type": "intentionally_omitted_duplicate_fragments",
                    "group_type": group_type,
                    "omitted_fragment_reasons": omitted_reasons,
                    "reason": "These fragments were explicitly compressed as redundant rather than silently dropped.",
                }
            )
        text = "\n\n".join(paragraph.text for paragraph in paragraphs)
        self._validate_non_enumerative_prose(text, group_type)
        if group_type == "issue":
            self._validate_issue_citation_style(text)
            self._validate_issue_abstraction(text)
        maximum_chars = (
            self.issue_type_synthesis_max_chars
            if group_type == "issue"
            else self.type_synthesis_max_chars
        )
        if len(text) > maximum_chars:
            raise ValueError(
                f"Type integration text is too long for {group_type}: "
                f"{len(text)} characters; maximum={maximum_chars}."
            )
        cited_in_text = {paper_id for paragraph in paragraphs for paper_id in paragraph.cited_papers}
        covered = [
            fragment.fragment_id
            for fragment in fragments
            if fragment.fragment_id in covered_in_paragraphs
        ]

        return TypeSynthesis(
            type_synthesis_id=f"TYPE-{group_type.upper()}",
            group_type=group_type,
            title=str(data.get("title") or group_type).strip(),
            text=text,
            cited_papers=sorted(cited_in_text),
            covered_fragment_ids=covered,
            uncovered_fragment_ids=[],
            paragraphs=paragraphs,
            quality_issues=quality_issues,
        )

    def _repair_type_integration_llm(
        self,
        llm_client: LLMClient,
        group_type: str,
        fragments: list[ReviewFragment],
        previous_output: dict[str, Any],
        validation_error: str,
    ) -> TypeSynthesis:
        """Repair a returned JSON object without resending all source prose."""
        source_manifest = [
            {"fragment_id": fragment.fragment_id, "cited_papers": fragment.cited_papers}
            for fragment in fragments
        ]
        repair_prompt = (
            "你是严格 JSON 格式修复器。将 previous_output 改写为完整、合法的类型级综述 JSON。"
            "只修复结构、引用映射和验收错误；不要引入 source_manifest 之外的 fragment_id 或 paper_id，"
            "不要在 sentences[].text 中写 P 编号或任何括号引用。每句必须有非空 source_fragment_ids 和"
            "1–3 篇 cited_papers；首段 role=synthesis，中间段 role=exposition，末段 role=appraisal。"
            "所有 source_manifest 中的片段必须在 source_fragment_ids 中覆盖，或写入 omitted_fragment_ids 并说明具体重复原因。"
            "只输出 JSON 对象，且必须含 title、paragraphs、omitted_fragment_ids。"
        )
        data = llm_client.chat_json(
            repair_prompt,
            {
                "group_type": group_type,
                "validation_error": validation_error,
                "source_manifest": source_manifest,
                "previous_output": previous_output,
            },
            temperature=0.0,
        )
        self._type_response_local.data = data
        return self._integrate_type_llm(
            llm_client,
            prompts={},
            query="",
            group_type=group_type,
            fragments=fragments,
            response_data=data,
        )

    def _render_type_sentence(self, sentence: TypeSynthesisSentence) -> str:
        body = sentence.text.rstrip()
        if not re.search(r"[。！？!?]$", body):
            body += "。"
        return body[:-1] + f"（{', '.join(sentence.cited_papers)}）" + body[-1]

    def _parse_omitted_fragment_reasons(self, raw_values: Any) -> dict[str, str]:
        if not isinstance(raw_values, list):
            raise ValueError("Type integration omitted_fragment_ids must be an array.")
        reasons: dict[str, str] = {}
        for raw in raw_values:
            if isinstance(raw, dict):
                fragment_id = str(raw.get("fragment_id") or "").strip()
                reason = str(raw.get("reason") or "").strip()
            else:
                # Older prompts exposed only uncovered_fragment_ids; reject a
                # bare identifier so the model must state why it is safe to omit.
                fragment_id, reason = str(raw or "").strip(), ""
            if not fragment_id:
                continue
            if not reason:
                raise ValueError(f"Type integration omitted {fragment_id} without a compression reason.")
            reasons[fragment_id] = reason
        return reasons

    def _validate_non_enumerative_prose(self, text: str, group_type: str) -> None:
        matched_sequences = [sequence for sequence in self.enumerative_sequences if all(marker in text for marker in sequence)]
        if matched_sequences:
            labels = ["/".join(sequence) for sequence in matched_sequences]
            raise ValueError(
                f"Type integration used list-like enumeration instead of review prose for {group_type}: {labels}."
            )

    def _validate_issue_citation_style(self, text: str) -> None:
        matched_phrases = [phrase for phrase in self.issue_citation_dump_phrases if phrase in text]
        if matched_phrases:
            raise ValueError(
                "Issue synthesis used citation-dump wording instead of claim-adjacent citations: "
                f"{matched_phrases}."
            )
        text_without_parenthetical_citations = re.sub(r"（[^（）]*）|\([^()]*\)", "", text)
        bare_paper_ids = sorted(set(self.paper_id_re.findall(text_without_parenthetical_citations)))
        if bare_paper_ids:
            raise ValueError(
                "Issue synthesis used bare paper_ids outside parentheses: "
                f"{bare_paper_ids}."
            )
        oversized_clusters = []
        for match in re.finditer(r"（([^（）]*)）|\(([^()]*)\)", text):
            cluster_text = match.group(1) if match.group(1) is not None else match.group(2)
            paper_ids = self.paper_id_re.findall(cluster_text or "")
            if len(paper_ids) > 4:
                oversized_clusters.append(paper_ids)
        if oversized_clusters:
            raise ValueError(
                "Issue synthesis used oversized citation clusters; keep at most four representative papers "
                f"per claim: {oversized_clusters}."
            )

    def _normalize_issue_citation_clusters(self, text: str) -> str:
        def normalize(match: re.Match) -> str:
            opening = "（" if match.group(1) is not None else "("
            closing = "）" if opening == "（" else ")"
            cluster_text = match.group(1) if match.group(1) is not None else match.group(2)
            paper_ids = list(dict.fromkeys(self.paper_id_re.findall(cluster_text or "")))
            if len(paper_ids) <= 4:
                return match.group(0)
            return f"{opening}{', '.join(paper_ids[:4])}{closing}"

        return re.sub(r"（([^（）]*)）|\(([^()]*)\)", normalize, text)

    def _validate_issue_abstraction(self, text: str) -> None:
        text_without_citations = re.sub(r"（[^（）]*）|\([^()]*\)", "", text)
        technical_names = sorted(
            set(re.findall(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9-]{3,}(?![A-Za-z0-9_])", text_without_citations))
        )
        if technical_names:
            raise ValueError(
                "Issue type synthesis retained concrete model, system, framework, or benchmark names "
                f"instead of high-level query attention facets: {technical_names}."
            )

    def _final_section_citation_issues(
        self,
        content: str,
        type_syntheses: list[TypeSynthesis],
    ) -> list[dict]:
        """Collect section-level citation warnings without replacing compressed prose."""
        sections = self._markdown_level_two_sections(content)
        issues: list[dict] = []
        for synthesis in type_syntheses:
            expected_titles = self.final_section_titles.get(synthesis.group_type, ())
            section_text = next(
                (sections[title] for title in expected_titles if title in sections),
                None,
            )
            if section_text is None:
                issues.append(
                    {
                        "issue_type": "missing_final_section",
                        "group_type": synthesis.group_type,
                        "reason": f"Final assembly omitted the required section: {list(expected_titles)}.",
                    }
                )
                continue
            section_citations = set(self.paper_id_re.findall(section_text))
            allowed_for_type = set(synthesis.cited_papers)
            if not section_citations:
                issues.append(
                    {
                        "issue_type": "section_without_citation",
                        "group_type": synthesis.group_type,
                        "section_title": expected_titles[0],
                        "reason": "The final section contains no explicit paper_id citation.",
                    }
                )
            elif not (section_citations & allowed_for_type):
                issues.append(
                    {
                        "issue_type": "section_citation_mismatch",
                        "group_type": synthesis.group_type,
                        "section_title": expected_titles[0],
                        "reason": f"The section contains no citation from {synthesis.type_synthesis_id}.",
                    }
                )
        return issues

    def _markdown_level_two_sections(self, content: str) -> dict[str, str]:
        matches = list(re.finditer(r"^##\s+(.+?)\s*$", content, flags=re.MULTILINE))
        sections: dict[str, str] = {}
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            sections[match.group(1).strip()] = content[start:end].strip()
        return sections

    def _require_text(self, value: Any, group_id: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"LLM fragment returned empty text for {group_id}.")
        return text

    def _select_tables(self, tables: list[GroupContributionTable]) -> list[GroupContributionTable]:
        selected = []
        for group_type in ["issue", "problem", "method", "finding"]:
            candidates = [t for t in tables if t.group_type == group_type and len(t.contributions) >= 1]
            candidates.sort(key=lambda t: (t.group_quality_flag != "normal", -len(t.contributions), t.group_id))
            selected.extend(candidates)
        return selected
