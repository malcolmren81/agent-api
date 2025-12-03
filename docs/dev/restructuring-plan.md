# Palet8 Agents-API Restructuring Plan

## Objective
Restructure Palet8's agent system to align with HelloAgents' "tool-centric" architecture. Extract embedded logic from monolithic agents into modular services and tools, externalize configurations, and simplify agents to be thin orchestrators.

**Delivery Approach:** Incremental PRs (5 phases)

---

## Current State Analysis

### PaliAgent (`palet8_agents/agents/pali_agent.py`) - ~614 lines
**Embedded Data Classes (Lines 72-141):**
- `RequirementsStatus` (completeness tracking with weighted scoring)

**Hard-coded Configs:**
- `completeness_threshold=0.5`, `max_qa_rounds=None`
- Completeness weights: subject=0.5, style=0.2, colors=0.15, mood=0.15
- `min_input_length=3`, `max_input_length=10000`
- LLM temperatures: 0.2 (extraction), 0.7 (response)

**Embedded System Prompt (Lines 23-69)**

**1 Service Dependency:** TextLLM

---

### PlannerAgent (`palet8_agents/agents/planner_agent.py`) - ~2000 lines
**Embedded Data Classes/Enums (Lines 57-294):**
- `PlannerPhase` (enum)
- `ContextCompleteness`, `SafetyClassification`, `PromptDimensions`
- `GenerationParameters`, `PipelineConfig`, `AssemblyRequest`, `EvaluationFeedback`

**Hard-coded Configs (Lines 389-465):**
- `REQUIRED_FIELDS`, `IMPORTANT_FIELDS`, `OPTIONAL_FIELDS`
- `COMPLETENESS_WEIGHTS` (7 fields with weights)
- `SAFETY_RISK_KEYWORDS` (4 severity levels, 20+ keywords)
- `DUAL_PIPELINE_TRIGGERS` (4 trigger categories)
- `DUAL_PIPELINES` (3 pipeline configs)
- Thresholds: `min_context_completeness=0.5`, `min_prompt_quality=0.45`, etc.

**Embedded System Prompt (Lines 301-359)**

**7 Service Dependencies:** TextLLM, Reasoning, Context, ModelInfo, PromptTemplate, PromptComposer, WebSearch

---

### EvaluatorAgent (`palet8_agents/agents/evaluator_agent.py`) - ~1240 lines
**Embedded Data Classes/Enums (Lines 49-364):**
- `EvaluationPhase`, `EvaluationDecision` (enums)
- `PromptQualityDimension`, `ResultQualityDimension` (enums)
- `PromptQualityResult`, `RetrySuggestion`, `ResultQualityResult`, `EvaluationPlan`

**Hard-coded Weight/Threshold Tables (Lines 88-204):**
- `PROMPT_QUALITY_WEIGHTS` (3 modes × 5 dimensions)
- `PROMPT_QUALITY_THRESHOLDS` (3 modes × 6 thresholds)
- `RESULT_QUALITY_WEIGHTS` (3 modes × 7 dimensions)
- `RESULT_QUALITY_THRESHOLDS` (3 modes × 8 thresholds)

**Embedded System Prompt (Lines 211-260)**

**2 Service Dependencies:** TextLLM, Reasoning

---

### SafetyAgent (`palet8_agents/agents/safety_agent.py`) - ~670 lines
**Embedded Data Classes/Enums (Lines 76-139):**
- `SafetyCategory`, `SafetySeverity` (enums)
- `SafetyFlag`, `SafetyResult`

**Partial Externalization:** Already loads from `config/safety_config.yaml`

**Hard-coded (Lines 585-606):**
- Score-to-severity mapping
- Severity penalty table

**Embedded System Prompt (Lines 30-73)**

**1 Service Dependency:** TextLLM

---

## Implementation Plan

### Phase 1: Create Models Package (Extract Data Classes)
**New Directory:** `palet8_agents/models/`

| File | Contents (move from) |
|------|---------------------|
| `__init__.py` | Export all models |
| `requirements.py` | `RequirementsStatus` (pali) |
| `context.py` | `ContextCompleteness` (planner) |
| `safety.py` | `SafetyClassification` (planner), `SafetyCategory`, `SafetySeverity`, `SafetyFlag`, `SafetyResult` (safety) |
| `prompt.py` | `PromptDimensions` (planner), `PromptQualityDimension`, `PromptQualityResult` (evaluator) |
| `generation.py` | `GenerationParameters`, `PipelineConfig`, `AssemblyRequest` (planner) |
| `evaluation.py` | `EvaluationPhase`, `EvaluationDecision`, `ResultQualityDimension`, `ResultQualityResult`, `RetrySuggestion`, `EvaluationPlan`, `EvaluationFeedback` (evaluator) |
| `enums.py` | `PlannerPhase` (planner) |

#### Detailed Model Structures:

**`models/requirements.py`:**
```python
@dataclass
class RequirementsStatus:
    subject: Optional[str] = None
    style: Optional[str] = None
    colors: List[str] = field(default_factory=list)
    mood: Optional[str] = None
    composition: Optional[str] = None
    include_elements: List[str] = field(default_factory=list)
    avoid_elements: List[str] = field(default_factory=list)
    reference_image: Optional[str] = None
    additional_notes: Optional[str] = None

    @property
    def completeness_score(self) -> float: ...
    @property
    def is_complete(self) -> bool: ...
    @property
    def missing_fields(self) -> List[str]: ...
    def to_dict(self) -> Dict[str, Any]: ...
```

**`models/context.py`:**
```python
@dataclass
class ContextCompleteness:
    score: float  # 0.0 to 1.0
    is_sufficient: bool
    missing_fields: List[str]
    clarifying_questions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**`models/safety.py`:**
```python
class SafetyCategory(Enum):
    NSFW = "nsfw"
    VIOLENCE = "violence"
    HATE = "hate"
    IP_TRADEMARK = "ip_trademark"
    ILLEGAL = "illegal"

class SafetySeverity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SafetyClassification:
    is_safe: bool
    requires_review: bool
    risk_level: str  # "low", "medium", "high"
    categories: List[str]
    flags: Dict[str, bool] = field(default_factory=dict)
    reason: str = ""

@dataclass
class SafetyFlag:
    category: SafetyCategory
    severity: SafetySeverity
    score: float
    description: str
    source: str  # "input", "prompt", "image"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SafetyResult:
    job_id: str
    is_safe: bool
    overall_score: float
    flags: List[SafetyFlag] = field(default_factory=list)
    blocked_categories: List[str] = field(default_factory=list)
    user_message: Optional[str] = None
    alternatives: List[str] = field(default_factory=list)
```

**`models/prompt.py`:**
```python
class PromptQualityDimension(Enum):
    COVERAGE = "coverage"
    CLARITY = "clarity"
    PRODUCT_CONSTRAINTS = "product_constraints"
    STYLE_ALIGNMENT = "style_alignment"
    CONTROL_SURFACE = "control_surface"

@dataclass
class PromptDimensions:
    subject: Optional[str] = None
    aesthetic: Optional[str] = None
    color: Optional[str] = None
    composition: Optional[str] = None
    background: Optional[str] = None
    lighting: Optional[str] = None
    texture: Optional[str] = None
    detail_level: Optional[str] = None
    mood: Optional[str] = None
    expression: Optional[str] = None
    pose: Optional[str] = None
    art_movement: Optional[str] = None
    reference_style: Optional[str] = None
    technical: Dict[str, str] = field(default_factory=dict)

@dataclass
class PromptQualityResult:
    overall: float
    dimensions: Dict[str, float] = field(default_factory=dict)
    mode: str = "STANDARD"
    threshold: float = 0.70
    decision: str = "PASS"  # PASS, FIX_REQUIRED, POLICY_FAIL
    feedback: List[str] = field(default_factory=list)
    failed_dimensions: List[str] = field(default_factory=list)
```

**`models/generation.py`:**
```python
@dataclass
class GenerationParameters:
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    num_images: int = 1
    provider_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineConfig:
    pipeline_type: str = "single"  # "single" or "dual"
    pipeline_name: Optional[str] = None
    stage_1_model: str = ""
    stage_1_purpose: str = ""
    stage_2_model: Optional[str] = None
    stage_2_purpose: Optional[str] = None
    decision_rationale: str = ""

@dataclass
class AssemblyRequest:
    prompt: str = ""
    negative_prompt: str = ""
    mode: str = "STANDARD"
    dimensions: Dict[str, Any] = field(default_factory=dict)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    model_id: str = ""
    model_rationale: str = ""
    model_alternatives: List[str] = field(default_factory=list)
    parameters: GenerationParameters = field(default_factory=GenerationParameters)
    reference_image_url: Optional[str] = None
    reference_strength: float = 0.75
    prompt_quality_score: float = 0.0
    quality_acceptable: bool = False
    safety: SafetyClassification = field(default_factory=...)
    estimated_cost: float = 0.0
    estimated_time_ms: int = 0
    context_used: Optional[Dict[str, Any]] = None
    job_id: str = ""
    user_id: str = ""
    product_type: str = ""
    print_method: Optional[str] = None
    revision_count: int = 0
```

**`models/evaluation.py`:**
```python
class EvaluationPhase(Enum):
    CREATE_PLAN = "create_plan"
    EXECUTE = "execute"

class EvaluationDecision(Enum):
    PASS = "PASS"
    FIX_REQUIRED = "FIX_REQUIRED"
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    POLICY_FAIL = "POLICY_FAIL"

class ResultQualityDimension(Enum):
    PROMPT_FIDELITY = "prompt_fidelity"
    PRODUCT_READINESS = "product_readiness"
    TECHNICAL_QUALITY = "technical_quality"
    BACKGROUND_COMPOSITION = "background_composition"
    AESTHETIC = "aesthetic"
    TEXT_LEGIBILITY = "text_legibility"
    SET_CONSISTENCY = "set_consistency"

@dataclass
class RetrySuggestion:
    dimension: str
    suggested_changes: List[str] = field(default_factory=list)

@dataclass
class ResultQualityResult:
    overall: float
    dimensions: Dict[str, float] = field(default_factory=dict)
    mode: str = "STANDARD"
    threshold: float = 0.80
    decision: str = "APPROVE"
    feedback: List[str] = field(default_factory=list)
    failed_dimensions: List[str] = field(default_factory=list)
    retry_suggestions: List[RetrySuggestion] = field(default_factory=list)

@dataclass
class EvaluationPlan:
    job_id: str
    prompt: str
    negative_prompt: str = ""
    mode: str = "STANDARD"
    product_type: Optional[str] = None
    print_method: Optional[str] = None
    dimensions_requested: Dict[str, Any] = field(default_factory=dict)
    prompt_quality: Optional[PromptQualityResult] = None
    result_weights: Dict[str, float] = field(default_factory=dict)
    result_thresholds: Dict[str, float] = field(default_factory=dict)

@dataclass
class EvaluationFeedback:
    passed: bool
    overall_score: float
    issues: List[str]
    retry_suggestions: List[str]
    dimension_scores: Dict[str, float] = field(default_factory=dict)
```

---

### Phase 2: Externalize Configurations
**New Directory:** `config/` (extend existing)

| File | Contents |
|------|----------|
| `pali_config.yaml` | Completeness weights, thresholds, input limits, LLM temperatures |
| `context_analysis.yaml` | `REQUIRED_FIELDS`, `IMPORTANT_FIELDS`, `OPTIONAL_FIELDS`, `COMPLETENESS_WEIGHTS`, thresholds |
| `prompt_evaluation.yaml` | `PROMPT_QUALITY_WEIGHTS`, `PROMPT_QUALITY_THRESHOLDS` (all 3 modes) |
| `result_evaluation.yaml` | `RESULT_QUALITY_WEIGHTS`, `RESULT_QUALITY_THRESHOLDS` (all 3 modes) |
| `pipeline.yaml` | `DUAL_PIPELINE_TRIGGERS`, `DUAL_PIPELINES` |
| `safety_config.yaml` | (exists) Add `SAFETY_RISK_KEYWORDS`, severity penalties |

#### Detailed Config Structures:

**`config/pali_config.yaml`:**
```yaml
# Pali Agent Configuration
completeness:
  threshold: 0.5
  weights:
    subject: 0.50
    style: 0.20
    colors: 0.15
    mood: 0.15

input_validation:
  min_length: 3
  max_length: 10000

llm:
  extraction_temperature: 0.2
  response_temperature: 0.7
  max_qa_rounds: null  # unlimited

required_fields:
  - subject

recommended_fields:
  - style
  - colors
  - mood
```

**`config/context_analysis.yaml`:**
```yaml
# Context Analysis Configuration (Planner)
fields:
  required:
    - subject
  important:
    - style
    - aesthetic
    - colors
    - product_type
  optional:
    - mood
    - composition
    - background
    - lighting

completeness_weights:
  subject: 0.40
  style: 0.15
  aesthetic: 0.15
  colors: 0.10
  product_type: 0.10
  mood: 0.05
  composition: 0.05

thresholds:
  min_completeness: 0.5
  min_prompt_quality: 0.45
  max_prompt_revisions: 3
  max_fix_iterations: 3

rag:
  min_context_items: 2
  web_search_max_results: 3
```

**`config/prompt_evaluation.yaml`:**
```yaml
# Prompt Quality Evaluation Configuration
weights:
  RELAX:
    coverage: 0.30
    clarity: 0.25
    product_constraints: 0.20
    style_alignment: 0.10
    control_surface: 0.15
  STANDARD:
    coverage: 0.25
    clarity: 0.25
    product_constraints: 0.20
    style_alignment: 0.15
    control_surface: 0.15
  COMPLEX:
    coverage: 0.22
    clarity: 0.17
    product_constraints: 0.22
    style_alignment: 0.22
    control_surface: 0.17

thresholds:
  RELAX:
    overall: 0.50
    coverage: 0.50
    clarity: 0.40
    product_constraints: 0.40
    style_alignment: 0.30
    control_surface: 0.30
  STANDARD:
    overall: 0.70
    coverage: 0.70
    clarity: 0.60
    product_constraints: 0.60
    style_alignment: 0.50
    control_surface: 0.50
  COMPLEX:
    overall: 0.85
    coverage: 0.85
    clarity: 0.75
    product_constraints: 0.80
    style_alignment: 0.70
    control_surface: 0.70

required_fields_by_mode:
  RELAX:
    - subject
  STANDARD:
    - subject
    - aesthetic
    - background
  COMPLEX:
    - subject
    - aesthetic
    - background
    - composition
    - lighting

clarity_checks:
  contradiction_pairs:
    - ["dark", "bright white background"]
    - ["minimalist", "highly detailed intricate"]
    - ["simple", "complex elaborate"]
  vague_terms:
    - nice
    - cool
    - good looking
    - awesome
    - great
  penalties:
    contradiction: -0.3
    vague_term: -0.1
    negative_contradiction: -0.2
```

**`config/result_evaluation.yaml`:**
```yaml
# Result Quality Evaluation Configuration
weights:
  RELAX:
    prompt_fidelity: 0.25
    product_readiness: 0.30
    technical_quality: 0.20
    background_composition: 0.10
    aesthetic: 0.10
    text_legibility: 0.05
    set_consistency: 0.00
  STANDARD:
    prompt_fidelity: 0.25
    product_readiness: 0.20
    technical_quality: 0.20
    background_composition: 0.15
    aesthetic: 0.15
    text_legibility: 0.05
    set_consistency: 0.00
  COMPLEX:
    prompt_fidelity: 0.22
    product_readiness: 0.18
    technical_quality: 0.18
    background_composition: 0.17
    aesthetic: 0.17
    text_legibility: 0.05
    set_consistency: 0.03

thresholds:
  RELAX:
    overall: 0.70
    prompt_fidelity: 0.50
    product_readiness: 0.60
    technical_quality: 0.50
    background_composition: 0.40
    aesthetic: 0.40
    text_legibility: 0.50
    set_consistency: 0.0
  STANDARD:
    overall: 0.80
    prompt_fidelity: 0.70
    product_readiness: 0.70
    technical_quality: 0.60
    background_composition: 0.60
    aesthetic: 0.60
    text_legibility: 0.60
    set_consistency: 0.0
  COMPLEX:
    overall: 0.85
    prompt_fidelity: 0.80
    product_readiness: 0.75
    technical_quality: 0.75
    background_composition: 0.70
    aesthetic: 0.70
    text_legibility: 0.70
    set_consistency: 0.60

quality_checks:
  min_resolution: 1024
  min_coverage_percent: 0.8
  sharpness_threshold: 0.5
  noise_threshold: 0.3

text_detection_keywords:
  - text
  - logo
  - typography
  - lettering
  - words
```

**`config/pipeline.yaml`:**
```yaml
# Pipeline Configuration
dual_pipeline_triggers:
  text_in_image:
    - text
    - typography
    - lettering
    - font
    - words
    - title
    - headline
    - quote
  character_refinement:
    - character edit
    - face fix
    - expression change
    - pose adjust
  multi_element:
    - multiple subjects
    - complex composition
    - layered design
  production_quality:
    - print-ready
    - production
    - high accuracy
    - 4k
    - poster

dual_pipelines:
  creative_art:
    stage_1_model: midjourney-v7
    stage_1_purpose: "Initial creative generation"
    stage_2_model: nano-banana-2-pro
    stage_2_purpose: "Refinement and detail enhancement"
  photorealistic:
    stage_1_model: imagen-4-ultra
    stage_1_purpose: "Photorealistic base generation"
    stage_2_model: nano-banana-2-pro
    stage_2_purpose: "Detail and lighting refinement"
  layout_poster:
    stage_1_model: flux-2-flex
    stage_1_purpose: "Layout and composition"
    stage_2_model: qwen-image-edit
    stage_2_purpose: "Text and element placement"

cost_estimation:
  default_per_image: 0.04
  default_latency_ms: 15000
  dual_pipeline_latency_multiplier: 2.5
```

**`config/safety_config.yaml` (additions):**
```yaml
# Add to existing safety_config.yaml
risk_keywords:
  critical:
    - nude
    - naked
    - xxx
    - porn
    - pornograph
    - nsfw
    - sexual
    - erotic
    - hentai
    - child abuse
    - terror attack
    - mass shooting
    - torture of children
    - nazi propaganda
    - genocide
    - ethnic cleansing
    - white supremacy
    - child exploitation
    - human trafficking
    - csam
  high:
    - explicit
    - violence
    - gore
    - hate
  medium:
    - sexy
    - provocative
    - weapon
    - blood
    - scary
  low:
    - adult
    - mature
    - dark
    - aggressive

severity_penalties:
  none: 0.0
  low: 0.05
  medium: 0.15
  high: 0.30
  critical: 0.50

score_to_severity:
  critical: 0.9
  high: 0.7
  medium: 0.5
  low: 0.3

safety_threshold: 0.5
```

**New Directory:** `prompts/`

| File | Contents |
|------|----------|
| `pali_system.txt` | Pali agent system prompt (~50 lines) |
| `planner_system.txt` | Planner agent system prompt (~60 lines) |
| `evaluator_system.txt` | Evaluator agent system prompt (~50 lines) |
| `safety_system.txt` | Safety agent system prompt (~45 lines) |
| `dimension_filling.txt` | Dimension filling prompt template |
| `requirements_extraction.txt` | Requirements extraction prompt template |

#### Prompt File Contents:

**`prompts/pali_system.txt`:**
```
You are Pali, a friendly design assistant for Palet8's print-on-demand platform.
You are the only agent that talks directly to users.

YOUR ROLE:
- Gather design requirements through natural conversation
- Guide users through UI selectors (product, template, aspect ratio, style)
- Be warm, encouraging, and creative
- Keep exchanges short and natural

UI SELECTORS TO REFERENCE:
- Product category: apparel, drinkware, wall art, accessories, stickers
- Product template: specific product within category
- Aspect ratio: square, landscape, portrait, wide, tall
- Aesthetic style: realistic, illustration, cartoon, minimal, vintage, streetwear
- Characters: system library or user-uploaded
- Reference image: upload or URL
- Text in image: content and font style

INFORMATION TO GATHER:
- Product category & template
- Aspect ratio
- Aesthetic style
- Characters (if needed)
- Reference images (if any)
- Text to include (if any)
- Core design concept/idea

WHEN COMPLETE:
Summarize selections, output structured brief, mark ready for planning.

TONE:
Act as a creative partner. Make the design process easy and fun.
```

**`prompts/requirements_extraction.txt`:**
```
Extract design requirements from the conversation as JSON.

Return ONLY valid JSON with these fields:
{
  "subject": "main subject or concept",
  "style": "visual style",
  "colors": ["list", "of", "colors"],
  "mood": "emotional tone",
  "composition": "composition notes",
  "include_elements": ["elements", "to", "include"],
  "avoid_elements": ["elements", "to", "avoid"]
}

If a field is not mentioned, use null or empty array.
```

---

### Phase 3: Create New Services
**Directory:** `palet8_agents/services/`

| Service | Responsibility | Extracted From |
|---------|---------------|----------------|
| `requirements_analysis_service.py` | Analyze conversation for requirements, calculate completeness | `PaliAgent.analyze_requirements()`, `RequirementsStatus` logic |
| `context_analysis_service.py` | Evaluate context completeness, generate clarifying questions | `PlannerAgent._evaluate_context_completeness()`, `_generate_question_for_field()` |
| `dimension_selection_service.py` | Select dimensions based on mode/product/style | `PlannerAgent._select_dimensions()`, `_fill_missing_dimensions()` |
| `model_selection_service.py` | Choose optimal image model/pipeline | `PlannerAgent._select_model()`, `_decide_pipeline()` |
| `prompt_evaluation_service.py` | Assess prompt quality, propose revisions | `PlannerAgent._evaluate_and_revise_prompt()`, `EvaluatorAgent._evaluate_prompt_quality()` |
| `result_evaluation_service.py` | Evaluate generated images | `EvaluatorAgent._evaluate_result_quality()` and all `_score_*` methods |
| `safety_classification_service.py` | Content safety checks | `SafetyAgent._check_text_content()`, `_check_ip_trademark()`, `_quick_keyword_check()` |
| `memory_service.py` | Long-term storage/retrieval using PostgreSQL + pgvector | Leverage existing: `embedding_service.py`, `scripts/vector_db_setup.sql` |

#### Detailed Service Implementations:

**`services/requirements_analysis_service.py`:**
```python
class RequirementsAnalysisService:
    """Analyze conversation to extract and score design requirements."""

    def __init__(self, text_service: TextLLMService, config_path: Path = None):
        self._text_service = text_service
        self._config = self._load_config(config_path or "config/pali_config.yaml")

    async def analyze_conversation(
        self,
        conversation: Conversation,
        system_prompt: Optional[str] = None,
    ) -> RequirementsStatus:
        """
        Extract requirements from conversation using LLM.

        Args:
            conversation: Full conversation history
            system_prompt: Optional override for extraction prompt

        Returns:
            RequirementsStatus with extracted fields and completeness score
        """
        # Use LLM to extract JSON from conversation
        # Parse into RequirementsStatus
        # Calculate completeness using config weights
        pass

    def calculate_completeness(self, requirements: RequirementsStatus) -> float:
        """Calculate weighted completeness score."""
        weights = self._config["completeness"]["weights"]
        score = 0.0
        if requirements.subject:
            score += weights["subject"]
        if requirements.style:
            score += weights["style"]
        if requirements.colors:
            score += weights["colors"]
        if requirements.mood:
            score += weights["mood"]
        return score

    def get_missing_fields(self, requirements: RequirementsStatus) -> List[str]:
        """Return list of missing required/recommended fields."""
        pass

    def is_complete(self, requirements: RequirementsStatus) -> bool:
        """Check if minimum requirements are met (subject required)."""
        return requirements.subject is not None
```

**`services/context_analysis_service.py`:**
```python
class ContextAnalysisService:
    """Evaluate context completeness for planner decision-making."""

    def __init__(self, config_path: Path = None):
        self._config = self._load_config(config_path or "config/context_analysis.yaml")

    def evaluate_completeness(
        self,
        requirements: Dict[str, Any],
    ) -> ContextCompleteness:
        """
        Evaluate if requirements provide sufficient context for planning.

        Args:
            requirements: Dict of gathered requirements

        Returns:
            ContextCompleteness with score, missing fields, clarifying questions
        """
        weights = self._config["completeness_weights"]
        threshold = self._config["thresholds"]["min_completeness"]

        score = 0.0
        missing = []
        questions = []

        for field, weight in weights.items():
            if requirements.get(field):
                score += weight
            else:
                missing.append(field)
                questions.append(self._generate_question(field))

        return ContextCompleteness(
            score=score,
            is_sufficient=score >= threshold,
            missing_fields=missing,
            clarifying_questions=questions,
        )

    def _generate_question(self, field: str) -> str:
        """Generate clarifying question for a missing field."""
        question_map = {
            "subject": "What would you like as the main subject of your design?",
            "style": "What visual style are you looking for?",
            "aesthetic": "What aesthetic feeling should the design convey?",
            "colors": "Do you have any color preferences?",
            "product_type": "What product will this design be used on?",
            "mood": "What mood or emotion should the design evoke?",
            "composition": "Any preferences for how elements should be arranged?",
            "background": "What kind of background would you like?",
        }
        return question_map.get(field, f"Could you tell me more about the {field}?")
```

**`services/dimension_selection_service.py`:**
```python
class DimensionSelectionService:
    """Select prompt dimensions based on mode, product, and style."""

    def __init__(
        self,
        text_service: TextLLMService,
        prompt_template_service: PromptTemplateService,
        config_path: Path = None,
    ):
        self._text_service = text_service
        self._prompt_template_service = prompt_template_service
        self._config = self._load_config(config_path or "config/context_analysis.yaml")

    async def select_dimensions(
        self,
        mode: str,  # RELAX, STANDARD, COMPLEX
        requirements: Dict[str, Any],
        product_type: Optional[str] = None,
        print_method: Optional[str] = None,
    ) -> PromptDimensions:
        """
        Select and populate dimensions for prompt composition.

        Args:
            mode: Generation mode (affects required dimensions)
            requirements: User requirements dict
            product_type: Target product (affects technical specs)
            print_method: Print method (screen print, DTG, etc.)

        Returns:
            PromptDimensions with selected values
        """
        # Map requirements to dimensions
        dimensions = self._map_requirements_to_dimensions(requirements)

        # Fill missing required dimensions using LLM
        missing = self._get_missing_for_mode(dimensions, mode)
        if missing:
            dimensions = await self._fill_missing_dimensions(dimensions, requirements, missing)

        # Add technical specs for COMPLEX mode
        if mode == "COMPLEX":
            dimensions.technical = self._build_technical_specs(product_type, print_method)

        return dimensions

    def _map_requirements_to_dimensions(self, requirements: Dict) -> PromptDimensions:
        """Direct mapping from requirements to dimension fields."""
        return PromptDimensions(
            subject=requirements.get("subject"),
            aesthetic=requirements.get("aesthetic") or requirements.get("style"),
            color=", ".join(requirements.get("colors", [])) or None,
            composition=requirements.get("composition"),
            background=requirements.get("background"),
            mood=requirements.get("mood"),
        )

    async def _fill_missing_dimensions(
        self,
        dimensions: PromptDimensions,
        requirements: Dict,
        missing: List[str],
    ) -> PromptDimensions:
        """Use LLM to generate values for missing dimensions."""
        prompt = self._load_prompt("prompts/dimension_filling.txt")
        # Call LLM with context and parse response
        pass

    def _build_technical_specs(
        self,
        product_type: Optional[str],
        print_method: Optional[str],
    ) -> Dict[str, str]:
        """Build technical specifications for COMPLEX mode."""
        specs = {}
        if print_method == "screen_print":
            specs["color_separation"] = "spot colors, max 6"
            specs["halftone"] = "required for gradients"
        # Add DPI, bleed, safe zone based on product
        return specs
```

**`services/model_selection_service.py`:**
```python
class ModelSelectionService:
    """Select optimal image generation model and pipeline."""

    def __init__(
        self,
        model_info_service: ModelInfoService,
        config_path: Path = None,
    ):
        self._model_info_service = model_info_service
        self._config = self._load_config(config_path or "config/pipeline.yaml")

    async def select_model(
        self,
        mode: str,
        requirements: Dict[str, Any],
        needs_speed: bool = False,
        needs_quality: bool = False,
    ) -> Tuple[str, str, List[str]]:
        """
        Select optimal model based on requirements.

        Args:
            mode: Generation mode
            requirements: User requirements
            needs_speed: Prioritize fast generation
            needs_quality: Prioritize quality (premium)

        Returns:
            Tuple of (model_id, rationale, alternatives)
        """
        # Get available models and compatibility scores
        models = await self._model_info_service.get_available_models()

        # Score each model
        scored = []
        for model in models:
            score = model.compatibility_score
            if needs_speed and "fast" in model.capabilities:
                score += 0.1
            if needs_quality:
                score += 0.2 * model.quality_score
            if mode == "COMPLEX":
                score += 0.1 * model.quality_score
            scored.append((model, score))

        # Sort and select
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = scored[0][0]
        alternatives = [m.id for m, _ in scored[1:4]]

        return selected.id, self._build_rationale(selected, mode), alternatives

    async def select_pipeline(
        self,
        requirements: Dict[str, Any],
        prompt: str,
    ) -> PipelineConfig:
        """
        Decide single vs dual pipeline based on requirements.

        Args:
            requirements: User requirements
            prompt: Generated prompt text

        Returns:
            PipelineConfig with pipeline type and models
        """
        triggers = self._config["dual_pipeline_triggers"]
        combined_text = f"{prompt} {str(requirements)}".lower()

        # Check for dual pipeline triggers
        triggered_category = None
        for category, keywords in triggers.items():
            if any(kw in combined_text for kw in keywords):
                triggered_category = category
                break

        if not triggered_category:
            return PipelineConfig(pipeline_type="single")

        # Select appropriate dual pipeline
        pipelines = self._config["dual_pipelines"]
        if "text" in triggered_category or "layout" in triggered_category:
            preset = pipelines["layout_poster"]
        elif "photo" in str(requirements.get("style", "")).lower():
            preset = pipelines["photorealistic"]
        else:
            preset = pipelines["creative_art"]

        return PipelineConfig(
            pipeline_type="dual",
            pipeline_name=triggered_category,
            stage_1_model=preset["stage_1_model"],
            stage_1_purpose=preset["stage_1_purpose"],
            stage_2_model=preset["stage_2_model"],
            stage_2_purpose=preset["stage_2_purpose"],
        )
```

**`services/prompt_evaluation_service.py`:**
```python
class PromptEvaluationService:
    """Assess prompt quality and propose revisions."""

    def __init__(
        self,
        reasoning_service: ReasoningService,
        config_path: Path = None,
    ):
        self._reasoning_service = reasoning_service
        self._config = self._load_config(config_path or "config/prompt_evaluation.yaml")

    async def assess_quality(
        self,
        prompt: str,
        negative_prompt: str,
        mode: str,
        product_type: Optional[str] = None,
        dimensions: Optional[Dict] = None,
    ) -> PromptQualityResult:
        """
        Assess prompt quality across multiple dimensions.

        Args:
            prompt: Positive prompt text
            negative_prompt: Negative prompt text
            mode: RELAX, STANDARD, or COMPLEX
            product_type: Target product for constraint checking
            dimensions: Requested dimensions for coverage check

        Returns:
            PromptQualityResult with scores, decision, and feedback
        """
        weights = self._config["weights"][mode]
        thresholds = self._config["thresholds"][mode]

        # Score each dimension
        scores = {
            "coverage": self._score_coverage(prompt, mode, dimensions),
            "clarity": self._score_clarity(prompt, negative_prompt),
            "product_constraints": self._score_product_constraints(prompt, product_type),
            "style_alignment": self._score_style_alignment(prompt, mode),
            "control_surface": self._score_control_surface(negative_prompt),
        }

        # Calculate weighted overall
        overall = sum(scores[d] * weights[d] for d in scores)

        # Determine failed dimensions
        failed = [d for d, s in scores.items() if s < thresholds.get(d, 0.5)]

        # Make decision
        if overall < thresholds["overall"] or failed:
            decision = "FIX_REQUIRED"
        else:
            decision = "PASS"

        return PromptQualityResult(
            overall=overall,
            dimensions=scores,
            mode=mode,
            threshold=thresholds["overall"],
            decision=decision,
            feedback=self._generate_feedback(scores, thresholds),
            failed_dimensions=failed,
        )

    def _score_coverage(self, prompt: str, mode: str, dimensions: Dict) -> float:
        """Check if required dimensions are covered in prompt."""
        required = self._config["required_fields_by_mode"][mode]
        present = sum(1 for f in required if dimensions and dimensions.get(f))
        return present / len(required) if required else 1.0

    def _score_clarity(self, prompt: str, negative_prompt: str) -> float:
        """Check for contradictions and vague terms."""
        score = 1.0
        checks = self._config["clarity_checks"]

        # Check contradiction pairs
        for pair in checks["contradiction_pairs"]:
            if pair[0] in prompt.lower() and pair[1] in prompt.lower():
                score += checks["penalties"]["contradiction"]

        # Check vague terms
        for term in checks["vague_terms"]:
            if term in prompt.lower():
                score += checks["penalties"]["vague_term"]

        # Check negative contradicts positive
        prompt_words = set(prompt.lower().split())
        neg_words = set(negative_prompt.lower().split())
        if prompt_words & neg_words:
            score += checks["penalties"]["negative_contradiction"]

        return max(0.0, min(1.0, score))

    async def propose_revision(
        self,
        prompt: str,
        quality_result: PromptQualityResult,
    ) -> str:
        """Use reasoning service to propose improved prompt."""
        return await self._reasoning_service.propose_prompt_revision(
            prompt=prompt,
            feedback=quality_result.feedback,
            failed_dimensions=quality_result.failed_dimensions,
        )
```

**`services/result_evaluation_service.py`:**
```python
class ResultEvaluationService:
    """Evaluate generated image quality."""

    def __init__(
        self,
        reasoning_service: ReasoningService,
        config_path: Path = None,
    ):
        self._reasoning_service = reasoning_service
        self._config = self._load_config(config_path or "config/result_evaluation.yaml")

    async def evaluate_image(
        self,
        image_data: Dict[str, Any],
        plan: EvaluationPlan,
    ) -> ResultQualityResult:
        """
        Evaluate generated image against prompt and quality standards.

        Args:
            image_data: Dict with image URL, metadata, analysis results
            plan: EvaluationPlan with prompt, mode, product context

        Returns:
            ResultQualityResult with scores and decision
        """
        mode = plan.mode
        weights = self._config["weights"][mode]
        thresholds = self._config["thresholds"][mode]

        # Score each dimension
        scores = {}
        feedback = []

        scores["prompt_fidelity"] = await self._score_prompt_fidelity(
            image_data, plan.prompt, plan.product_type
        )
        scores["product_readiness"] = self._score_product_readiness(image_data)
        scores["technical_quality"] = self._score_technical_quality(image_data)
        scores["background_composition"] = self._score_background_composition(
            image_data, plan.dimensions_requested
        )
        scores["aesthetic"] = self._score_aesthetic(image_data, plan)

        # Conditional scoring
        if self._has_text_content(plan.prompt):
            scores["text_legibility"] = self._score_text_legibility(image_data)
        if self._is_multi_image(image_data):
            scores["set_consistency"] = self._score_set_consistency(image_data)

        # Calculate weighted overall
        overall = sum(scores[d] * weights.get(d, 0.1) for d in scores)

        # Determine failed dimensions
        failed = [d for d, s in scores.items() if s < thresholds.get(d, 0.5)]

        # Make decision
        if overall < thresholds["overall"] or failed:
            decision = "REJECT"
            suggestions = self._generate_retry_suggestions(scores, thresholds)
        else:
            decision = "APPROVE"
            suggestions = []

        return ResultQualityResult(
            overall=overall,
            dimensions=scores,
            mode=mode,
            threshold=thresholds["overall"],
            decision=decision,
            feedback=feedback,
            failed_dimensions=failed,
            retry_suggestions=suggestions,
        )

    async def _score_prompt_fidelity(
        self,
        image_data: Dict,
        prompt: str,
        product_type: Optional[str],
    ) -> float:
        """Use reasoning service to assess prompt adherence."""
        description = image_data.get("description", "")
        if not description:
            return 0.6  # Default if no description

        alignment = await self._reasoning_service.assess_design_alignment(
            prompt=prompt,
            description=description,
            product_type=product_type,
        )
        return alignment.prompt_adherence

    def _score_product_readiness(self, image_data: Dict) -> float:
        """Check resolution, coverage, cropping issues."""
        score = 0.7
        checks = self._config["quality_checks"]

        if image_data.get("resolution", 0) < checks["min_resolution"]:
            score -= 0.3
        if image_data.get("coverage", 1.0) < checks["min_coverage_percent"]:
            score -= 0.2
        if image_data.get("has_cropping_issues"):
            score -= 0.3

        return max(0.0, score)

    def _score_technical_quality(self, image_data: Dict) -> float:
        """Check for defects, sharpness, noise."""
        score = 0.8
        checks = self._config["quality_checks"]

        defects = image_data.get("detected_defects", [])
        score -= 0.15 * len(defects)

        if image_data.get("sharpness_score", 1.0) < checks["sharpness_threshold"]:
            score -= 0.2
        if image_data.get("noise_level", 0.0) > checks["noise_threshold"]:
            score -= 0.2

        return max(0.0, score)
```

**`services/safety_classification_service.py`:**
```python
class SafetyClassificationService:
    """Content safety classification and IP checks."""

    def __init__(
        self,
        text_service: TextLLMService,
        config_path: Path = None,
    ):
        self._text_service = text_service
        self._config = self._load_config(config_path or "config/safety_config.yaml")
        self._ip_blocklist = self._build_ip_blocklist()

    async def classify_content(
        self,
        text: str,
        source: str = "input",  # input, prompt, image
    ) -> Optional[SafetyFlag]:
        """
        Classify content for safety violations.

        Args:
            text: Content to classify
            source: Where the content came from

        Returns:
            SafetyFlag if violation detected, None if safe
        """
        if not text or len(text.strip()) < 3:
            return None

        # Fast keyword check first
        keyword_flag = self._quick_keyword_check(text, source)
        if keyword_flag and keyword_flag.severity == SafetySeverity.CRITICAL:
            return keyword_flag

        # IP/trademark check
        ip_flag = self._check_ip_trademark(text, source)
        if ip_flag:
            return ip_flag

        # LLM-based classification for edge cases
        if keyword_flag:
            return keyword_flag

        return await self._llm_classify(text, source)

    def _quick_keyword_check(self, text: str, source: str) -> Optional[SafetyFlag]:
        """Fast keyword-based safety check."""
        text_lower = text.lower()
        keywords = self._config["risk_keywords"]

        # Check critical keywords (always block)
        for kw in keywords["critical"]:
            if kw in text_lower:
                return SafetyFlag(
                    category=SafetyCategory.NSFW,
                    severity=SafetySeverity.CRITICAL,
                    score=1.0,
                    description=f"Critical keyword detected: {kw}",
                    source=source,
                )

        # Check other severity levels
        for level in ["high", "medium", "low"]:
            for kw in keywords.get(level, []):
                if kw in text_lower:
                    return SafetyFlag(
                        category=self._categorize_keyword(kw),
                        severity=SafetySeverity[level.upper()],
                        score=self._config["score_to_severity"][level],
                        description=f"{level.title()} risk keyword: {kw}",
                        source=source,
                    )

        return None

    def _check_ip_trademark(self, text: str, source: str) -> Optional[SafetyFlag]:
        """Check for IP/trademark violations."""
        text_lower = text.lower()

        for term in self._ip_blocklist:
            if term in text_lower:
                return SafetyFlag(
                    category=SafetyCategory.IP_TRADEMARK,
                    severity=SafetySeverity.MEDIUM,
                    score=0.6,
                    description=f"Potential IP/trademark: {term}",
                    source=source,
                    metadata={"visibility_control": "user_private"},
                )

        return None

    def get_severity_penalty(self, severity: SafetySeverity) -> float:
        """Get score penalty for a severity level."""
        return self._config["severity_penalties"].get(severity.value, 0.0)
```

**`services/memory_service.py`:**
```python
class MemoryService:
    """Long-term memory storage using PostgreSQL + pgvector."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        db_url: Optional[str] = None,
    ):
        self._embedding_service = embedding_service
        self._db_url = db_url or os.environ.get("DATABASE_URL")
        self._pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._db_url)
        return self._pool

    async def store_design_summary(
        self,
        job_id: str,
        user_id: str,
        summary: str,
        prompt: str,
        product_type: Optional[str] = None,
        style: Optional[str] = None,
        image_url: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Store design summary with embedding for later retrieval.

        Returns:
            ID of stored summary
        """
        # Generate embedding
        embedding = await self._embedding_service.embed_for_storage(summary)

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                INSERT INTO design_summaries
                (job_id, user_id, summary, final_prompt, product_type, style,
                 image_url, embedding, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
                """,
                job_id, user_id, summary, prompt, product_type, style,
                image_url, embedding, json.dumps(metadata or {}),
            )
            return str(result["id"])

    async def search_similar_prompts(
        self,
        query: str,
        user_id: Optional[str] = None,
        product_type: Optional[str] = None,
        min_score: float = 0.7,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar prompts using vector similarity.

        Args:
            query: Search query text
            user_id: Optional filter by user
            product_type: Optional filter by product
            min_score: Minimum evaluation score
            limit: Max results

        Returns:
            List of matching prompts with similarity scores
        """
        # Generate query embedding
        query_embedding = await self._embedding_service.embed_for_search(query)

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Build query with optional filters
            sql = """
                SELECT id, prompt, negative_prompt, style, product_type,
                       evaluation_score, metadata,
                       1 - (embedding <=> $1::vector) as similarity
                FROM prompt_embeddings
                WHERE embedding IS NOT NULL
                  AND evaluation_score >= $2
            """
            params = [query_embedding, min_score]

            if user_id:
                sql += " AND user_id = $3"
                params.append(user_id)
            if product_type:
                sql += f" AND product_type = ${len(params) + 1}"
                params.append(product_type)

            sql += " ORDER BY embedding <=> $1::vector LIMIT $" + str(len(params) + 1)
            params.append(limit)

            rows = await conn.fetch(sql, *params)
            return [dict(row) for row in rows]

    async def get_user_history(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get recent design history for a user."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, summary, title, product_type, style,
                       image_url, created_at
                FROM design_summaries
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                user_id, limit,
            )
            return [dict(row) for row in rows]

    async def get_art_references(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search art library for reference images."""
        query_embedding = await self._embedding_service.embed_for_search(query)

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            sql = """
                SELECT id, name, description, category, tags,
                       image_url, thumbnail_url,
                       1 - (embedding <=> $1::vector) as similarity
                FROM art_library
                WHERE embedding IS NOT NULL
            """
            params = [query_embedding]

            if category:
                sql += " AND category = $2"
                params.append(category)

            sql += " ORDER BY embedding <=> $1::vector LIMIT $" + str(len(params) + 1)
            params.append(limit)

            rows = await conn.fetch(sql, *params)
            return [dict(row) for row in rows]

    async def close(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
```

---

### Phase 4: Create New Tools (Wrap Services)
**Directory:** `palet8_agents/tools/`

| Tool | Actions | Wraps Service |
|------|---------|---------------|
| `requirements_tool.py` | `analyze_requirements`, `get_completeness` | RequirementsAnalysisService |
| `context_tool.py` | `evaluate_completeness`, `get_user_history`, `get_art_references` | ContextAnalysisService, MemoryService |
| `dimension_tool.py` | `select_dimensions` | DimensionSelectionService |
| `model_selector_tool.py` | `select_model`, `select_pipeline` | ModelSelectionService |
| `prompt_quality_tool.py` | `assess_prompt_quality`, `revise_prompt` | PromptEvaluationService |
| `image_evaluation_tool.py` | `evaluate_image`, `evaluate_set` | ResultEvaluationService |
| `safety_tool.py` | `classify_content`, `check_ip` | SafetyClassificationService |
| `memory_tool.py` | `store`, `retrieve`, `search` | MemoryService |

#### Tool Registry Updates:

**`tools/registry.py` additions:**
```python
class ToolRegistry:
    """Central registry for agent tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool by name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Get tool by name."""
        return self._tools.get(name)

    async def call(self, name: str, action: str, **kwargs) -> ToolResult:
        """Call a tool action with parameters."""
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, data=None, error=f"Tool not found: {name}")
        return await tool(action=action, **kwargs)

    def get_all_schemas(self) -> List[Dict]:
        """Get OpenAI-format schemas for all tools."""
        return [tool.get_openai_schema() for tool in self._tools.values()]


def create_default_registry(
    text_service: TextLLMService,
    reasoning_service: ReasoningService,
    memory_service: MemoryService,
    # ... other services
) -> ToolRegistry:
    """Create registry with all default tools."""
    registry = ToolRegistry()

    # Create services
    requirements_service = RequirementsAnalysisService(text_service)
    context_service = ContextAnalysisService()
    dimension_service = DimensionSelectionService(text_service, prompt_template_service)
    model_service = ModelSelectionService(model_info_service)
    prompt_eval_service = PromptEvaluationService(reasoning_service)
    result_eval_service = ResultEvaluationService(reasoning_service)
    safety_service = SafetyClassificationService(text_service)

    # Register tools
    registry.register(RequirementsTool(requirements_service))
    registry.register(ContextTool(context_service, memory_service))
    registry.register(DimensionTool(dimension_service))
    registry.register(ModelSelectorTool(model_service))
    registry.register(PromptQualityTool(prompt_eval_service))
    registry.register(ImageEvaluationTool(result_eval_service))
    registry.register(SafetyTool(safety_service))
    registry.register(MemoryTool(memory_service))

    return registry
```

---

### Phase 5: Simplify Agents
Reduce agents to orchestrators that call tools via registry.

**PaliAgent (target: ~150 lines):**
```python
async def run(self, context, user_input=None):
    # 1. Validate input
    validation = self._validate_input(user_input)
    if not validation.valid:
        return self._create_result(success=False, error=validation.issues)

    # 2. Analyze requirements via tool
    requirements = await self.call_tool("requirements", "analyze_requirements", conversation)

    # 3. Check completeness
    if requirements.is_complete:
        # Store to memory and delegate to planner
        await self.call_tool("memory", "store", context.conversation)
        context.requirements = requirements.to_dict()
        return self._create_result(data=requirements, next_agent="planner")

    # 4. Generate response asking for more info
    response = await self._generate_response(requirements.missing_fields)
    return self._create_result(data=response, requires_user_input=True)
```

**PlannerAgent (target: ~300 lines):**
```python
async def run(self, context, user_input=None):
    # 1. Check context completeness
    completeness = await self.call_tool("context", "evaluate_completeness", context.requirements)
    if not completeness.is_sufficient:
        return self._create_result(requires_user_input=True, data=completeness.clarifying_questions)

    # 2. Safety check
    safety = await self.call_tool("safety", "classify_content", context.requirements)
    if not safety.is_safe:
        return self._create_result(success=False, error="safety_violation")

    # 3. Select dimensions
    dimensions = await self.call_tool("dimension", "select_dimensions", mode, requirements)

    # 4. Compose prompt (existing service)
    prompt = await self._prompt_composer.compose(dimensions, context)

    # 5. Evaluate prompt quality
    quality = await self.call_tool("prompt_quality", "assess_prompt_quality", prompt)
    if not quality.acceptable:
        prompt = await self.call_tool("prompt_quality", "revise_prompt", prompt, quality.feedback)

    # 6. Select model/pipeline
    model = await self.call_tool("model_selector", "select_model", mode, requirements)

    # 7. Build and return AssemblyRequest
    return self._create_result(data=AssemblyRequest(...))
```

**EvaluatorAgent (target: ~200 lines):**
```python
async def run(self, context, phase):
    if phase == "create_plan":
        result = await self.call_tool("prompt_quality", "assess_prompt_quality", context.plan.prompt)
        return self._create_result(data=result, next_agent="planner" if result.decision == "FIX_REQUIRED" else None)
    else:  # execute
        result = await self.call_tool("image_evaluation", "evaluate_image", context.image_data, context.plan)
        return self._create_result(data=result, next_agent="pali" if result.decision == "APPROVE" else "planner")
```

**SafetyAgent (target: ~150 lines):**
```python
async def on_event(self, event_type, event_data, job_id):
    flag = await self.call_tool("safety", "classify_content", event_data, event_type)
    if flag:
        self._job_flags[job_id].append(flag)
        self._update_job_score(job_id, flag.severity)
```

---

## File Changes Summary

### New Files to Create
```
palet8_agents/
├── models/
│   ├── __init__.py
│   ├── requirements.py
│   ├── context.py
│   ├── safety.py
│   ├── prompt.py
│   ├── generation.py
│   ├── evaluation.py
│   └── enums.py
├── services/
│   ├── requirements_analysis_service.py
│   ├── context_analysis_service.py
│   ├── dimension_selection_service.py
│   ├── model_selection_service.py
│   ├── prompt_evaluation_service.py
│   ├── result_evaluation_service.py
│   ├── safety_classification_service.py
│   └── memory_service.py
├── tools/
│   ├── requirements_tool.py
│   ├── context_tool.py (extend existing)
│   ├── dimension_tool.py
│   ├── model_selector_tool.py
│   ├── prompt_quality_tool.py
│   ├── image_evaluation_tool.py
│   ├── safety_tool.py (extend existing)
│   └── memory_tool.py
config/
├── pali_config.yaml
├── context_analysis.yaml
├── prompt_evaluation.yaml
├── result_evaluation.yaml
└── pipeline.yaml
prompts/
├── pali_system.txt
├── planner_system.txt
├── evaluator_system.txt
├── safety_system.txt
├── dimension_filling.txt
└── requirements_extraction.txt
```

### Files to Modify
```
palet8_agents/agents/pali_agent.py       (reduce from ~614 to ~150 lines)
palet8_agents/agents/planner_agent.py    (reduce from ~2000 to ~300 lines)
palet8_agents/agents/evaluator_agent.py  (reduce from ~1240 to ~200 lines)
palet8_agents/agents/safety_agent.py     (reduce from ~670 to ~150 lines)
palet8_agents/tools/registry.py          (add new tools)
palet8_agents/tools/__init__.py          (export new tools)
config/safety_config.yaml                (add keywords, penalties)
```

---

## Execution Order (Incremental PRs)

### PR 1: Models Package + Configs
**Scope:** Extract data classes, externalize configurations (no breaking changes)
- Create `palet8_agents/models/` with all data classes/enums
- Create `config/*.yaml` files with weights, thresholds
- Create `prompts/*.txt` files with system prompts
- Update imports in existing agents (backward compatible)
- **Testing:** Existing tests should pass unchanged

### PR 2: New Services (Part 1 - Core)
**Scope:** Create foundational services
- `requirements_analysis_service.py`
- `context_analysis_service.py`
- `safety_classification_service.py`
- `memory_service.py` (leveraging existing pgvector)
- **Testing:** Unit tests for each service

### PR 3: New Services (Part 2 - Evaluation & Selection)
**Scope:** Create evaluation and selection services
- `dimension_selection_service.py`
- `model_selection_service.py`
- `prompt_evaluation_service.py`
- `result_evaluation_service.py`
- **Testing:** Unit tests for each service

### PR 4: Tools + Registry
**Scope:** Create tools wrapping services
- Create all 8 tools
- Update `registry.py` to register new tools
- Update `tools/__init__.py` exports
- **Testing:** Tool integration tests

### PR 5: Agent Refactoring
**Scope:** Simplify agents to use tools
- Refactor `PaliAgent` (~614 → ~150 lines)
- Refactor `PlannerAgent` (~2000 → ~300 lines)
- Refactor `EvaluatorAgent` (~1240 → ~200 lines)
- Refactor `SafetyAgent` (~670 → ~150 lines)
- **Testing:** Full integration tests, verify existing API contracts

Dont change Agent System Prompt

---

## Additional Implementation Requirements

### Memory Strategy
- **Short-term**: Continue storing conversation history in `AgentContext` for session state
- **Long-term**: Persist key details through the memory tool when needed
- **Unified approach**: Both session memory and long-term memory work together

### Error Handling
All simplified agents must include robust error handling when calling tools:
```python
async def run(self, context, user_input=None):
    # Call tool with error checking
    result = await self.call_tool("requirements", "analyze_requirements", conversation)

    if not result.success:
        logger.error(f"[{self.name}] Tool call failed: {result.error}")
        return self._create_result(
            success=False,
            error=result.error,
            error_code=result.error_code or "TOOL_ERROR"
        )

    # Type conversion with validation
    try:
        requirements = RequirementsStatus(**result.data)
    except (TypeError, ValueError) as e:
        logger.error(f"[{self.name}] Type conversion failed: {e}")
        return self._create_result(success=False, error=f"Invalid data format: {e}")
```

### Debug Logging
Add checkpoint logs at each agent state transition:
```python
import logging

logger = logging.getLogger(__name__)

async def run(self, context, user_input=None):
    logger.debug(f"[{self.name}] START - job_id={context.job_id}")

    # Step 1
    logger.debug(f"[{self.name}] CHECKPOINT: Analyzing requirements")
    result = await self.call_tool(...)
    logger.debug(f"[{self.name}] CHECKPOINT: Requirements analysis complete - success={result.success}")

    # Step 2
    logger.debug(f"[{self.name}] CHECKPOINT: Checking completeness")
    ...

    logger.debug(f"[{self.name}] END - success={final_result.success}")
    return final_result
```

### Testing Strategy
- **Incremental tests**: Add unit tests for each new service/tool as created
- **Service tests**: Test services in isolation with mocked dependencies
- **Tool tests**: Test tools with mocked services
- **Integration tests**: Test full agent workflows after refactoring
- **Backward compatibility**: Ensure existing API contracts remain unchanged

---

## Benefits

- **Modularity**: Services and tools testable independently
- **Configurability**: Weights, thresholds, prompts editable without code changes
- **Extensibility**: New capabilities added as services/tools
- **Maintainability**: Agents reduced to ~20% of current size
- **HelloAgents Alignment**: Tool-centric architecture matches reference framework
