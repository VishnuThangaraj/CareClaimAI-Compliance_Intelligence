from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

# Constants

BIOMEDICAL_NER_MODEL = "d4data/biomedical-ner-all"
CLINICAL_NER_MODEL   = "samrawal/bert-base-uncased_clinical-ner"
CHUNK_SIZE           = 512          # max tokens per chunk (BERT limit)
OVERLAP_SIZE         = 50           # token overlap between chunks
CONFIDENCE_THRESHOLD = 0.70         # minimum NER confidence to keep entity


# Domain Patterns (fast regex fallback / augmentation)

BILLING_CODE_PATTERNS: Dict[str, re.Pattern] = {
    "CPT":    re.compile(r'\b(\d{5}[A-Z0-9]?)\b'),
    "HCPCS":  re.compile(r'\b([A-V]\d{4})\b'),
    "ICD10":  re.compile(r'\b([A-Z]\d{2}(?:\.[A-Z0-9]{1,4})?)\b'),
    "DRG":    re.compile(r'\bDRG[\s-]?(\d{3})\b', re.I),
    "NPI":    re.compile(r'\bNPI[\s:]?(\d{10})\b', re.I),
}

CHANGE_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "COVERAGE_EXPANSION":   ["now covered", "added to coverage", "benefit expansion", "newly covered"],
    "COVERAGE_RESTRICTION": ["no longer covered", "non-covered", "excluded", "coverage denied"],
    "REIMBURSEMENT_CHANGE": ["rate change", "fee schedule", "RVU", "reimbursement rate", "payment rate"],
    "PRIOR_AUTH_UPDATE":    ["prior authorization", "pre-authorization", "PA required", "authorization required"],
    "DOCUMENTATION_UPDATE": ["documentation requirement", "medical record", "clinical documentation"],
    "EFFECTIVE_DATE_CHANGE":["effective date", "implementation date", "goes into effect"],
    "CODING_UPDATE":        ["new code", "deleted code", "revised code", "code change", "crosswalk"],
}

FINANCIAL_PATTERN = re.compile(r'\$\s*[\d,]+(?:\.\d{2})?')


# Data Models

@dataclass
class Entity:
    text: str
    label: str          # e.g. "CPT", "DISEASE", "CHANGE_TYPE"
    start: int
    end: int
    confidence: float
    normalized: Optional[str] = None    # canonical form (e.g. ICD-10 code)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyNERResult:
    raw_text: str
    entities: List[Entity]
    billing_codes: List[Dict[str, str]]     # [{type, code, context}]
    change_types: List[str]
    diagnoses: List[str]
    procedures: List[str]
    drugs: List[str]
    financial_amounts: List[str]
    key_sections: List[Dict[str, str]]      # [{section, text}]
    summary_entities: Dict[str, List[str]]  # aggregated by label


# NER Pipeline

class PolicyNERPipeline:

    def __init__(self, use_gpu: bool = False):
        self.device = 0 if (use_gpu and torch.cuda.is_available()) else -1
        self._bio_pipeline = None
        self._clinical_pipeline = None
        self._spacy_nlp = None
        logger.info(f"NER Pipeline init | device={'GPU' if self.device == 0 else 'CPU'}")

    # Model Loading

    def _load_bio_pipeline(self):
        if self._bio_pipeline is None:
            logger.info(f"Loading {BIOMEDICAL_NER_MODEL}...")
            self._bio_pipeline = pipeline(
                "ner",
                model=BIOMEDICAL_NER_MODEL,
                tokenizer=BIOMEDICAL_NER_MODEL,
                aggregation_strategy="simple",
                device=self.device,
            )
            logger.success(f"Loaded {BIOMEDICAL_NER_MODEL}")
        return self._bio_pipeline

    def _load_clinical_pipeline(self):
        if self._clinical_pipeline is None:
            logger.info(f"Loading {CLINICAL_NER_MODEL}...")
            try:
                self._clinical_pipeline = pipeline(
                    "ner",
                    model=CLINICAL_NER_MODEL,
                    aggregation_strategy="simple",
                    device=self.device,
                )
                logger.success(f"Loaded {CLINICAL_NER_MODEL}")
            except Exception as e:
                logger.warning(f"Clinical NER load failed: {e} — using bio-only mode")
        return self._clinical_pipeline

    def _load_spacy(self):
        if self._spacy_nlp is None:
            try:
                import spacy
                self._spacy_nlp = spacy.load("en_core_sci_lg")
                logger.success("Loaded scispaCy en_core_sci_lg")
            except OSError:
                try:
                    import spacy
                    self._spacy_nlp = spacy.load("en_core_web_sm")
                    logger.warning("scispaCy not installed; fell back to en_core_web_sm")
                except Exception:
                    logger.warning("No spaCy model available; NER will use transformers only")
        return self._spacy_nlp

    # Public API───

    def extract(self, text: str) -> PolicyNERResult:

        text = text.strip()
        if not text:
            return self._empty_result(text)

        # Step 1: Transformer NER (chunked)
        transformer_entities = self._transformer_ner(text)

        # Step 2: Regex billing codes
        regex_entities = self._regex_extract_codes(text)

        # Step 3: Change-type detection
        change_types = self._detect_change_types(text)

        # Step 4: Financial amounts
        financial = FINANCIAL_PATTERN.findall(text)

        # Step 5: Aggregate all entities
        all_entities = self._merge_entities(transformer_entities + regex_entities)

        return self._build_result(text, all_entities, change_types, financial)

    # Chunked Transformer NER

    def _transformer_ner(self, text: str) -> List[Entity]:
        bio_pipe = self._load_bio_pipeline()
        entities: List[Entity] = []

        chunks = self._chunk_text(text)
        char_offset = 0

        for chunk_text, offset in chunks:
            try:
                raw = bio_pipe(chunk_text)
                for item in raw:
                    score = float(item.get("score", 0))
                    if score < CONFIDENCE_THRESHOLD:
                        continue
                    entities.append(Entity(
                        text=item["word"].strip(),
                        label=self._normalize_bio_label(item["entity_group"]),
                        start=item["start"] + offset,
                        end=item["end"] + offset,
                        confidence=score,
                    ))
            except Exception as e:
                logger.warning(f"Transformer NER chunk failed: {e}")

        # Optionally augment with clinical NER
        try:
            clin_pipe = self._load_clinical_pipeline()
            if clin_pipe:
                for chunk_text, offset in chunks:
                    raw = clin_pipe(chunk_text)
                    for item in raw:
                        score = float(item.get("score", 0))
                        if score >= CONFIDENCE_THRESHOLD:
                            entities.append(Entity(
                                text=item["word"].strip(),
                                label=self._normalize_clinical_label(item["entity_group"]),
                                start=item["start"] + offset,
                                end=item["end"] + offset,
                                confidence=score,
                            ))
        except Exception:
            pass

        return entities

    def _chunk_text(self, text: str) -> List[Tuple[str, int]]:
        """Split long text into overlapping chunks with character offsets."""
        words = text.split()
        chunks: List[Tuple[str, int]] = []
        current_words: List[str] = []
        current_offset = 0
        char_count = 0
        i = 0

        while i < len(words):
            word = words[i]
            current_words.append(word)
            char_count += len(word) + 1

            if char_count >= CHUNK_SIZE * 4:  # ~4 chars/token heuristic
                chunk_text = " ".join(current_words)
                chunks.append((chunk_text, current_offset))
                # overlap: keep last N words
                overlap = current_words[-OVERLAP_SIZE:] if len(current_words) > OVERLAP_SIZE else current_words[:]
                current_offset += char_count - sum(len(w) + 1 for w in overlap)
                current_words = overlap
                char_count = sum(len(w) + 1 for w in overlap)

            i += 1

        if current_words:
            chunks.append((" ".join(current_words), current_offset))

        return chunks if chunks else [(text, 0)]

    # Regex Extraction───────

    def _regex_extract_codes(self, text: str) -> List[Entity]:
        entities: List[Entity] = []
        for code_type, pattern in BILLING_CODE_PATTERNS.items():
            for m in pattern.finditer(text):
                # Get surrounding context (50 chars)
                ctx_start = max(0, m.start() - 50)
                ctx_end   = min(len(text), m.end() + 50)
                context   = text[ctx_start:ctx_end]

                entities.append(Entity(
                    text=m.group(0),
                    label="BILLING_CODE",
                    start=m.start(),
                    end=m.end(),
                    confidence=1.0,  # regex is deterministic
                    normalized=m.group(1),
                    metadata={"code_type": code_type, "context": context},
                ))
        return entities

    # Change Type Detection──

    def _detect_change_types(self, text: str) -> List[str]:
        text_lower = text.lower()
        detected: List[str] = []
        for change_type, keywords in CHANGE_TYPE_KEYWORDS.items():
            if any(kw.lower() in text_lower for kw in keywords):
                detected.append(change_type)
        return detected

    # Entity Normalization───

    @staticmethod
    def _normalize_bio_label(label: str) -> str:
        mapping = {
            "Disease": "DIAGNOSIS",
            "Chemical": "DRUG",
            "Gene": "GENE",
            "Mutation": "MUTATION",
            "Species": "SPECIES",
            "CellLine": "CELL_LINE",
            "CellType": "CELL_TYPE",
            "DNA": "DNA",
            "RNA": "RNA",
            "Protein": "PROTEIN",
        }
        return mapping.get(label, label.upper())

    @staticmethod
    def _normalize_clinical_label(label: str) -> str:
        mapping = {
            "problem": "DIAGNOSIS",
            "treatment": "PROCEDURE",
            "test": "DIAGNOSTIC_TEST",
            "PROBLEM": "DIAGNOSIS",
            "TREATMENT": "PROCEDURE",
            "TEST": "DIAGNOSTIC_TEST",
        }
        return mapping.get(label, label.upper())

    # Merge & Deduplicate────

    @staticmethod
    def _merge_entities(entities: List[Entity]) -> List[Entity]:
        """Remove duplicate spans; prefer higher-confidence entity."""
        seen: Dict[Tuple[int, int], Entity] = {}
        for ent in entities:
            key = (ent.start, ent.end)
            if key not in seen or ent.confidence > seen[key].confidence:
                seen[key] = ent
        return sorted(seen.values(), key=lambda e: e.start)

    # Result Construction────

    @staticmethod
    def _build_result(
        text: str,
        entities: List[Entity],
        change_types: List[str],
        financial: List[str],
    ) -> PolicyNERResult:
        billing_codes: List[Dict[str, str]] = []
        diagnoses, procedures, drugs = [], [], []
        key_sections: List[Dict[str, str]] = []

        for ent in entities:
            if ent.label == "BILLING_CODE":
                billing_codes.append({
                    "code": ent.normalized or ent.text,
                    "type": ent.metadata.get("code_type", "UNKNOWN"),
                    "context": ent.metadata.get("context", ""),
                    "confidence": str(round(ent.confidence, 3)),
                })
            elif ent.label == "DIAGNOSIS":
                diagnoses.append(ent.text)
            elif ent.label == "PROCEDURE":
                procedures.append(ent.text)
            elif ent.label == "DRUG":
                drugs.append(ent.text)

        # Deduplicate lists
        billing_codes = list({d["code"]: d for d in billing_codes}.values())
        diagnoses = list(dict.fromkeys(diagnoses))
        procedures = list(dict.fromkeys(procedures))
        drugs = list(dict.fromkeys(drugs))

        # Group entities by label
        summary: Dict[str, List[str]] = {}
        for ent in entities:
            summary.setdefault(ent.label, []).append(ent.text)

        return PolicyNERResult(
            raw_text=text,
            entities=entities,
            billing_codes=billing_codes,
            change_types=change_types,
            diagnoses=diagnoses,
            procedures=procedures,
            drugs=drugs,
            financial_amounts=financial,
            key_sections=key_sections,
            summary_entities={k: list(dict.fromkeys(v)) for k, v in summary.items()},
        )

    @staticmethod
    def _empty_result(text: str) -> PolicyNERResult:
        return PolicyNERResult(
            raw_text=text,
            entities=[],
            billing_codes=[],
            change_types=[],
            diagnoses=[],
            procedures=[],
            drugs=[],
            financial_amounts=[],
            key_sections=[],
            summary_entities={},
        )

_pipeline_instance: Optional[PolicyNERPipeline] = None


def get_ner_pipeline(use_gpu: bool = False) -> PolicyNERPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = PolicyNERPipeline(use_gpu=use_gpu)
    return _pipeline_instance


def extract_policy_entities(text: str) -> PolicyNERResult:
    return get_ner_pipeline().extract(text)
