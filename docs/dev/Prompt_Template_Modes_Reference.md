# Prompt Template Modes Reference

**Purpose**: Define prompt structures and one-shot examples for Relax/Standard/Complex modes.
**Status**: DRAFT - Pending stakeholder review before implementation.

---

## Mode Selection Decision Logic

| Factor | RELAX | STANDARD | COMPLEX |
|--------|-------|----------|---------|
| User keyword length | 3-10 words | 10-25 words | 25-50 words |
| Creativity slider | > 0.7 (high) | 0.3-0.7 (mid) | < 0.3 (precise) |
| Reference image | Optional, loose | Required if provided | Mandatory analysis |
| Color specification | General mood | Mood-based palette | Exact hex codes |
| Print requirements | None | General category | Full spec with method |
| Composition | Model decides | Rule-of-thirds, centered | Explicit grid % |

---

## Mode 1: RELAX

### Structure
```
[Style] [Subject] for [Product]
```

### Token Range
20-50 tokens

### Characteristics
- Abstract concepts allowed, no structure required
- Model interprets freely
- No negative prompt required
- Temperature: 0.8-1.0
- Top-P: 0.95, Top-K: 50

### One-Shot Example

**Input Requirements:**
```json
{
  "subject": "cute cat",
  "style": "kawaii",
  "product_type": "T-shirt"
}
```

**Output Prompt:**
```
Cute cat illustration for T-shirt, kawaii style
```

**Negative Prompt:**
```
(none)
```

---

## Mode 2: STANDARD

### Structure
```
[Subject Description], [Style Specification], [Technical Requirements], [Product Context]
```

### Token Range
50-150 tokens

### Characteristics
- Structured as [Subject] + [Action/State] + [Context]
- Mandatory main subject
- Specify style + era + medium
- Center-weighted or rule-of-thirds composition
- Basic negative prompt
- Temperature: 0.5-0.7
- Top-P: 0.85, Top-K: 40

### One-Shot Example

**Input Requirements:**
```json
{
  "subject": "playful cartoon cat with big eyes sitting in a coffee cup",
  "style": "kawaii Japanese illustration",
  "colors": ["soft pastel"],
  "background": "clean white background",
  "composition": "centered composition",
  "product_type": "T-shirt front chest print"
}
```

**Output Prompt:**
```
A playful cartoon cat with big eyes sitting in a coffee cup,
kawaii Japanese illustration style with soft pastel colors,
clean white background, centered composition,
for front chest T-shirt print, high resolution
```

**Negative Prompt:**
```
blurry, low quality, distorted, watermark
```

---

## Mode 3: COMPLEX

### Structure
```
[Detailed Subject]
[Comprehensive Style Chain]

LIGHTING: [light source, shadow, ambient details]

TEXTURE: [surface finish, material simulation]

COLOR SEPARATION: [max colors, hex codes, print compatibility]

TYPOGRAPHY: [text handling or "no text"]

COMPOSITION: [background, positioning with percentages]

BLEED/SAFE: [bleed dimensions, safe zone percentages]

PRODUCT: [product type, print method, size constraints]

NEGATIVE: [comprehensive exclusion list]
```

### Token Range
150-400 tokens

### Characteristics
- Explicit exclusions required
- AI improvises <20% of visual decisions
- Every ambiguity resolved in prompt
- Exact values for all visual elements
- Temperature: 0.2-0.4
- Top-P: 0.7, Top-K: 20

### One-Shot Example

**Input Requirements:**
```json
{
  "subject": "playful cartoon cat with oversized sparkling eyes sitting inside an oversized pastel pink coffee cup with steam wisps",
  "style": "kawaii Japanese illustration inspired by Sanrio aesthetics, soft cel-shading with subtle gradients",
  "colors": ["#FFB6C1", "#87CEEB", "#FFFACD", "#2F2F2F"],
  "color_separation": 4,
  "background": "pure white #FFFFFF",
  "composition": "centered at 50/50 with subject filling 65% of frame",
  "product_type": "cotton T-shirt front chest",
  "print_method": "screen printing",
  "aspect_ratio": "1:1",
  "dimensions": "3000x3000px"
}
```

**Output Prompt:**
```
A single playful cartoon cat character with oversized sparkling eyes
sitting inside an oversized pastel pink coffee cup with steam wisps,
kawaii Japanese illustration style inspired by Sanrio aesthetics,
soft cel-shading with subtle gradients.

LIGHTING: Flat ambient lighting with no harsh shadows, soft diffused glow
around steam wisps, minimal drop shadow at 10% opacity beneath cup.

TEXTURE: Smooth vector-like finish with slight paper grain overlay at 5%,
screen-print ink texture simulation on solid color areas.

COLOR SEPARATION: Max 4 spot colors for screen printing compatibility:
#FFB6C1 (Pantone 182 C), #87CEEB (Pantone 292 C),
#FFFACD (Pantone 7499 C), #2F2F2F (outlines). No gradients,
halftone dots acceptable for color transitions.

TYPOGRAPHY: No text in design.

COMPOSITION: Pure white #FFFFFF background for easy isolation,
centered at 50/50 with subject filling 65% of frame.

BLEED/SAFE: 3mm bleed on all edges, safe zone inner 85%,
no critical elements within outer 15%. 1:1 aspect ratio at 3000x3000px.

PRODUCT: Optimized for screen printing on cotton T-shirt front chest,
max 4-color separation, 12x16 inch print area.
```

**Negative Prompt:**
```
blurry, pixelated, watermark, text, extra limbs, asymmetrical eyes,
color banding, jpeg artifacts, busy background, realistic style, gradients,
photographic textures, complex shadows, logos, brand names, copyrighted characters
```

---

## Dimension Handling by Mode

| Dimension | RELAX | STANDARD | COMPLEX |
|-----------|-------|----------|---------|
| **User Keywords** | 3-10 words, abstract OK | 10-25 words, structured | 25-50 words, full spec |
| **Reference Image** | Optional, loose inspiration | 40-60% influence | 70-90% adherence required |
| **Composition** | Model decides freely | Center/rule-of-thirds | Explicit grid % |
| **Style** | Single keyword | Style + era + medium | Full style chain |
| **Background** | Any acceptable | "Clean" or "solid color" | Exact hex code |
| **Color Palette** | Model interprets | Mood-based | Exact hex codes, count limit |
| **Lighting** | Model decides | General mood | Explicit direction + angles |
| **Typography** | No constraint | Font category if needed | Full spec or "no text" |

---

## Generation Parameters by Mode

| Parameter | RELAX | STANDARD | COMPLEX |
|-----------|-------|----------|---------|
| Temperature | 0.8-1.0 | 0.5-0.7 | 0.2-0.4 |
| Top-P | 0.95 | 0.85 | 0.70 |
| Top-K | 50 | 40 | 20 |
| Reference Strength | 0.3 | 0.5 | 0.8 |

---

## Conflict Resolution Rules

When dimensions conflict, apply these priorities:

1. **Print Method > Style**: Print constraints override artistic style
2. **Aspect Ratio > Composition**: Aspect ratio is hard constraint
3. **Color Separation > Color Palette**: Max colors limits palette
4. **Brand Consistency > Creative Freedom** (Complex mode only)
5. **Resolution > Speed**: Always prioritize resolution for print

---

## Questions for Review

1. Are the one-shot examples appropriate for our product categories?
2. Should we add more examples for specific product types (wall art, phone case, etc.)?
3. Are the mode selection thresholds correct for our use cases?
4. Should creativity slider have more granular influence on mode selection?
5. Are there additional dimensions we need to support?

---

**Next Steps After Approval:**
1. Implement PromptWriterService with these templates
2. Update PlannerAgent to use mode selection logic
3. Update EvaluatorAgent to evaluate against mode-appropriate criteria
