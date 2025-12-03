# Prompt Template Framework for Print/Embroidery Product Graphics

## Configuration Summary
- **Agent Context**: Basic persona exists → Add task-specific instructions only
- **Input Type**: Text keywords + Reference images (dual input)
- **Product Scope**: All categories (Apparel, Accessories, Home goods)
- **Model Strategy**: Model-agnostic template

---

## Master Template Table

### Dimension Priority Weights (P1=Critical, P2=Important, P3=Optional)

| Priority | Dimensions |
|----------|------------|
| **P1** | User Keywords, Image Style, Product Type, Background, Aspect Ratio, IP/Legal Safety |
| **P2** | Reference Image, Image Composition, Color Palette, Clarity, Bleed/Safe Zones, Color Separation, Detail Level Slider, Lighting/Shadows |
| **P3** | Text/Typography, Texture/Material, Characters in Image, User History, Temperature, Top-K/Top-P, Seeds & Iterations |
| **P4** (Portrait/Photo) | Skin Texture, Camera & Lens Aesthetic, Film Stock & Emulation, Environment & Weather, Fashion & Clothing, Set Design & Background |
| **P5** (Style Direction) | Realism ↔ Artistic Slider |

---

### Main Prompt Template Table

| Dimension | Relax Mode | Standard Mode | Complex Mode |
|-----------|------------|---------------|--------------|
| **User Keywords** | 3-10 words, abstract concepts allowed, no structure required | 10-25 words, structured as [Subject] + [Action/State] + [Context], mandatory main subject | 25-50 words, structured as [Subject] + [Style Modifiers] + [Technical Specs] + [Negative Constraints], explicit exclusions required |
| **Reference Image** | Optional, used for loose inspiration only, no strict adherence | Required if provided, extract dominant style/color/composition, 40-60% influence weight | Mandatory analysis: extract exact color codes, composition grid, style fingerprint, 70-90% adherence required |
| **Image Composition** | Model decides freely, accept any layout | Center-weighted or rule-of-thirds, single focal point, clean margins (10% safe zone) | Explicit grid specification (e.g., "centered at 50/50, subject fills 60% frame"), golden ratio or specified proportions |
| **Image Style** | Single style keyword (e.g., "minimalist", "vintage") | Style + era + medium (e.g., "minimalist vector art, 2020s aesthetic") | Full style chain: [Base Style] + [Sub-style] + [Medium] + [Texture] + [Lighting] (e.g., "Japanese ukiyo-e woodblock, Edo period, handmade paper texture, diffused natural lighting") |
| **Product Type** | Implicit from context, no explicit constraint | Specify category: "for T-shirt print" / "phone case design" / "mug wrap" | Full specification: product + placement + print method (e.g., "front chest print for cotton T-shirt, DTG printing, max 12x16 inch print area") |
| **Clarity/Resolution** | Default model output, no upscaling requirement | "High resolution", "print quality", "sharp details" | Explicit DPI/PPI requirement, "vector-sharp edges", "no compression artifacts", "4K minimum base resolution" |
| **Aspect Ratio** | Model default or square | Product-appropriate preset: 1:1 (centered), 3:4 (portrait), 16:9 (banner) | Exact pixel dimensions or ratio with tolerance (e.g., "1080x1350px ±5%", "9:16 vertical, no cropping") |
| **Background** | Any background acceptable | "Clean background" or "solid color background" or "isolated subject" | "Pure white #FFFFFF background" / "transparent PNG ready" / "single flat color [hex code]", specify edge treatment |
| **Color Palette** | Model interprets freely | Specify mood (e.g., "warm earth tones", "monochrome", "vibrant pop colors") | Explicit hex codes or Pantone references, max color count for print method (e.g., "4-color limit: #FF5733, #33FF57, #3357FF, #FFFFFF") |
| **Characters in Image** | Allow any interpretation | Specify count + general type (e.g., "single cartoon character", "no human faces") | Full character spec: pose, expression, clothing, proportion, art style consistency with reference |
| **User History** | Ignore | Light reference: "consistent with user's previous style preferences" | Heavy personalization: analyze past 5+ generations, extract style DNA, maintain brand consistency score |
| **Temperature** | 0.8-1.0 (high creativity) | 0.5-0.7 (balanced) | 0.2-0.4 (deterministic, reproducible) |
| **Top-K / Top-P** | Top-P: 0.95, Top-K: 50 (diverse sampling) | Top-P: 0.85, Top-K: 40 (controlled diversity) | Top-P: 0.7, Top-K: 20 (focused, predictable output) |
| **Lighting/Shadows** | Model decides freely, any lighting acceptable | Specify general lighting mood: "flat lighting", "soft shadows", "dramatic lighting" | Explicit light source direction, shadow angle, ambient occlusion level (e.g., "top-left 45° key light, soft diffused fill, minimal drop shadow at 15% opacity") |
| **Text/Typography** | No text constraints, model may add text freely | If text needed: specify font category (serif/sans/display), placement zone, max word count | Full typography spec: font style reference, exact placement coordinates, text hierarchy, kerning hints, "no unintended text artifacts" in negatives |
| **Texture/Material** | Any texture acceptable | Specify texture category: "smooth", "fabric-like", "paper texture", "glossy" | Explicit texture reference: material simulation (e.g., "cotton fabric weave visible", "screen-print ink texture with slight cracking"), texture overlay percentage |
| **Color Separation** | Not considered, full-color output | Specify color mode: "CMYK-safe colors", "limited palette for screen printing" | Explicit separation: "max 4 spot colors", "each color on separate layer conceptually", Pantone-ready output, "avoid gradients for screen print" |
| **Bleed/Safe Zones** | No margin consideration | Standard safe zone: "15% margin from edges", "no critical elements near border" | Exact bleed spec: "3mm bleed on all sides", "safe zone inner 85%", "extend background to bleed edge", product-specific trim marks guidance |
| **IP / Legal Safety** | Avoid obvious copyrighted characters; use generic archetypes freely | Explicitly avoid named franchises, logos, celebrity likenesses; use descriptive archetypes (e.g., "friendly robot" not "WALL-E style") | Strict IP audit: no brand names, no character references, no trademarked styles; use original design language only; add negative: "no logos, no brand names, no copyrighted characters" |
| **Seeds & Iterations** | Random seeds, 1-3 outputs for quick exploration | 3-5 fixed seeds for reproducible variety; iterate 10-20 times for selection | Specific seed locks for batch consistency; 50-100+ iterations for systematic A/B testing; document winning seeds for campaign reuse |
| **Detail Level Slider** | Brief descriptors only; let AI improvise 70%+ of visual decisions | Moderate detail: describe main qualities (mood, lighting type, general colors); AI fills 40-50% of gaps | High detail lock-down: exact values for all visual elements; AI improvises <20%; every ambiguity resolved in prompt |
| **Skin Texture** (portraits) | Model default, general "realistic skin" | Specify texture level: "visible pores", "freckles", "unretouched skin", "peach fuzz" | Explicit micro-detail: "pore-level detail", "subsurface scattering", "micro-blemishes", "natural oil sheen", "goosebumps", specific skin conditions if relevant |
| **Camera & Lens Aesthetic** | Model default or generic "photo" | Specify general aesthetic: "35mm film look", "shallow depth of field", "analog camera look", "soft vintage" | Full camera spec: specific camera model (e.g., "Leica M6", "Hasselblad 500 C/M", "Pentax 67"), lens type ("85mm portrait", "macro 1:1", "anamorphic"), aperture ("f/1.4 bokeh", "f/8 sharp") |
| **Film Stock & Emulation** | Not specified, model interprets freely | General film mood: "Kodak warmth", "grainy film texture", "slight color shift" | Specific stock reference: "Kodak Portra 400", "CineStill 800T", "Fujifilm Velvia 50", "Ilford HP5 Plus 400", film-specific color science |
| **Environment & Weather** | Any environment acceptable | Specify general conditions: "clear sky", "overcast", "indoor studio" | Exact atmospheric spec: "golden hour with dramatic clouds", "misty bamboo forest", "heavy rain with puddle reflections", "aurora borealis", "dust storm", humidity level |
| **Fashion & Clothing Style** | Model interprets or none specified | Specify general style: "streetwear", "minimalist outfit", "business formal" | Full fashion spec: era + style + details (e.g., "1920s flapper with beaded fringe", "cyberpunk attire with neon accents", "Victorian dress with lace collar") |
| **Set Design & Background** | Simple background or model default | Specify background type: "seamless white backdrop", "urban alley", "studio cyclorama" | Full set specification: "art deco interior with geometric patterns", "Japanese tatami room with shoji screens", "brutalist concrete wall with weathered texture", prop placement |
| **Realism ↔ Artistic Slider** | Model interprets freely based on style keywords | Specify direction: "photorealistic", "hyper-realistic" OR "stylized illustration", "artistic interpretation" | Explicit spectrum position with anchors (see Realism Spectrum table below) |

---

### Realism ↔ Artistic Spectrum

| Direction | Level | Keywords & Anchors |
|-----------|-------|-------------------|
| **← Artistic** | Abstract | abstract art, geometric shapes, non-representational, expressionist |
| | Stylized | stylized illustration, cartoon, anime, cel-shaded, flat design, vector art |
| | Semi-Stylized | semi-realistic, stylized realism, Pixar style, illustrated portrait |
| **Center** | Balanced | editorial illustration, concept art, digital painting, matte painting |
| **Realistic →** | Semi-Real | soft realism, painterly realism, classical oil painting, Renaissance style |
| | Photorealistic | photorealistic, hyper-realistic, indistinguishable from photo, DSLR quality |
| | Documentary | raw photo, unedited photograph, candid snapshot, documentary style |

**Usage by Product Type:**
| Product | Recommended Direction | Rationale |
|---------|----------------------|-----------|
| T-shirt graphics | Stylized → Semi-Stylized | Prints better, more iconic |
| Phone cases | Stylized → Balanced | Artistic appeal, recognizable |
| Fine art prints | Balanced → Semi-Real | Gallery quality, artistic value |
| Product mockups | Photorealistic | Marketing authenticity |
| Editorial/Magazine | Balanced → Photorealistic | Context-dependent |
| Children's products | Stylized | Friendly, approachable |

---

### Prompt Length Guidelines by Mode

| Mode | Total Prompt Length | Recommended Structure |
|------|--------------------|-----------------------|
| **Relax** | 20-50 tokens | `[Style] [Subject] for [Product]` |
| **Standard** | 50-150 tokens | `[Subject Description], [Style Specification], [Technical Requirements], [Product Context]` |
| **Complex** | 150-400 tokens | `[Detailed Subject], [Comprehensive Style Chain], [Lighting/Texture], [Exact Technical Specs], [Color Separation], [Bleed/Typography], [Product + Print Method], [Negative Prompts]` |

**Readability Principle**: Prompts should remain **human-readable sentences**, not dense keyword lists. Structure as natural language flowing from Subject → Description → Style → Technical specs. Avoid:
- ❌ `cat, kawaii, pink, 1:1, 300dpi, white bg, no text`
- ✅ `A kawaii cat illustration in soft pink tones, 1:1 aspect ratio at 300dpi, on a clean white background, without text elements.`

---

### Conflict Resolution Rules

| Conflict Type | Resolution |
|---------------|------------|
| Style vs. Print Method | Print method constraints take priority (e.g., embroidery limits color count) |
| User Keywords vs. Reference Image | Standard: 60% keywords / 40% image. Complex: explicit weight in prompt |
| Aspect Ratio vs. Composition | Aspect ratio is hard constraint; adjust composition within ratio |
| Creative Freedom vs. Brand Consistency | Complex mode: brand consistency wins. Relax mode: creative freedom wins |
| Resolution vs. Speed | Always prioritize resolution for print products |
| Color Palette vs. Color Separation | Separation limits override palette (e.g., 4-color limit trumps 6-color palette) |
| Texture vs. Print Method | Screen print: avoid complex textures. DTG: textures acceptable |
| Typography vs. Model Capability | If model struggles with text, use "no text" and add typography post-processing |
| Lighting vs. Flat Design | For screen print/embroidery: flat lighting preferred over dramatic shadows |
| Bleed vs. Composition | Extend non-critical elements to bleed; keep focal point in safe zone |

---

### Negative Prompt Standards by Mode

| Mode | Negative Prompt Requirement |
|------|----------------------------|
| **Relax** | None required, model handles quality implicitly |
| **Standard** | Basic quality negatives: "blurry, low quality, distorted, watermark" |
| **Complex** | Comprehensive: "blurry, pixelated, low resolution, watermark, text artifacts, color banding, jpeg artifacts, cropped, out of frame, deformed, disfigured, bad anatomy, extra limbs, duplicate elements" + context-specific negatives |

---

### Product-Specific Presets

| Product | Background | Print Method | Special Notes |
|---------|------------|--------------|---------------|
| T-shirt (front) | Transparent/White | DTG, Screen Print | Keep 15% margin for print bleed; max 12x16 inch |
| T-shirt (all-over) | Seamless tileable | Sublimation | Ensure pattern continuity at seams |
| Phone Case | Solid or transparent | UV Print, Sublimation | Account for camera cutout zone; edge wrap |
| Mug (wrap) | White preferred | Sublimation | Design for cylinder wrap distortion; handle gap |
| Poster | Any | Offset, Digital | High-res mandatory (300 DPI+); bleed 3mm |
| Tote Bag | Transparent | Screen Print, DTG | Simple designs (≤4 colors) print better |
| Embroidery | N/A (thread on fabric) | Machine Embroidery | Max 6-8 thread colors; avoid fine details <2mm |

---

## MECE Validation

| Criterion | Status | Notes |
|-----------|--------|-------|
| Mutually Exclusive | ✓ | 29 dimensions cover distinct aspects; clear separation between camera/film/lighting/composition |
| Collectively Exhaustive | ✓ | Covers: Input → Visual → Technical → Legal → Photo/Portrait → Environment → Output |
| Priority Weighted | ✓ | P1 (6) / P2 (8) / P3 (7) / P4-Portrait (7) / P5-Style (1) hierarchy with context-specific layers |
| Conflict-Free | ✓ | 10 conflict resolution rules handle dimension interactions explicitly |
| Length-Complexity Balanced | ✓ | Relax: 20-50, Standard: 50-150, Complex: 150-400+ tokens |
| Print-Ready | ✓ | Manufacturing dimensions (Color Separation, Bleed, Texture) fully addressed |
| Legal-Safe | ✓ | IP/Legal Safety dimension prevents copyright/trademark violations |
| Reproducible | ✓ | Seeds & Iterations dimension enables A/B testing and campaign consistency |
| Photo-Ready | ✓ | P4 layer covers: Skin, Camera, Film Stock, Environment, Fashion, Set Design |
| Style-Directed | ✓ | P5 Realism ↔ Artistic slider with product-specific recommendations |
| Reference Libraries | ✓ | Quick-lookup tables for all major keyword categories |

---

## Example Prompts by Mode

### Relax Mode Example
```
Cute cat illustration for T-shirt, kawaii style
```

### Standard Mode Example
```
A playful cartoon cat with big eyes sitting in a coffee cup,
kawaii Japanese illustration style with soft pastel colors,
clean white background, centered composition,
for front chest T-shirt print, high resolution
```

### Complex Mode Example
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

TYPOGRAPHY: No text in design. Negative: "no unintended text artifacts,
no watermarks, no random letters."

COMPOSITION: Pure white #FFFFFF background for easy isolation,
centered at 50/50 with subject filling 65% of frame.

BLEED/SAFE: 3mm bleed on all edges, safe zone inner 85%,
no critical elements within outer 15%. 1:1 aspect ratio at 3000x3000px.

PRODUCT: Optimized for screen printing on cotton T-shirt front chest,
max 4-color separation, 12x16 inch print area.

NEGATIVE: blurry, pixelated, watermark, text, extra limbs, asymmetrical eyes,
color banding, jpeg artifacts, busy background, realistic style, gradients,
photographic textures, complex shadows.
```

---

## Implementation Notes

1. **No Additional Persona Needed**: Since basic persona exists, template focuses on task parameters only
2. **Model-Agnostic Design**: All specifications use natural language; no model-specific syntax
3. **Graceful Degradation**: If model doesn't support a parameter, prompt still works with reduced precision
4. **Extensibility**: New product types can be added to Product-Specific Presets table

---

## Quick Reference Libraries

### Skin Texture Keywords
| Level | Keywords |
|-------|----------|
| Subtle | detailed skin, visible pores, peach fuzz, natural redness |
| Medium | freckles, beauty marks, moles, unretouched, skin grain, micro-shadowing in pores |
| Intense | pore-level detail, subsurface scattering, micro-blemishes, goosebumps, natural oil sheen |
| Imperfections | scars, acne, hyperpigmentation, sun spots, age spots, under-eye bags, dry patches |

### Camera Models Reference
| Type | Models | Look |
|------|--------|------|
| 35mm Rangefinder | Leica M6, Contax T2 | Classic film, Zeiss contrast |
| 35mm SLR | Canon AE-1, Nikon F3, Canon 5D Mark II | Documentary, snapshot, early DSLR |
| Medium Format | Hasselblad 500 C/M, Pentax 67, Mamiya RZ67, Rolleiflex 2.8F | Sharp, shallow DOF, portrait clarity |
| Instant | Polaroid SX-70, Fuji Instax | Soft, vintage, instant film |
| Cinema | ARRI Alexa | Cinema color science |
| Modern Digital | Fujifilm X100V | Filmic color, 23mm |

### Film Stock Reference
| Mood | Stocks |
|------|--------|
| Warm Portrait | Kodak Portra 400, Kodak Portra 160 |
| Vivid Color | Kodak Ektar 100, Fujifilm Velvia 50 |
| Everyday Warm | Kodak Gold 200, Agfa Vista 200 |
| Cinematic | CineStill 800T (tungsten), CineStill 50D (daylight), Kodak Vision3 |
| B&W Classic | Kodak Tri-X 400, Ilford HP5 Plus 400 |
| B&W High ISO | Ilford Delta 3200, Kodak T-Max 400 |
| Slide Film | Fujifilm Provia 100F, Kodak Ektachrome E100 |
| Creative | LomoChrome Purple, Lomography 800 |

### Lighting Keywords
| Mood | Keywords |
|------|----------|
| Natural | golden hour, blue hour, overcast light, window light, daylight-balanced |
| Studio | soft diffused light, studio strobe, high-key, low-key, butterfly lighting |
| Dramatic | harsh midday sun, backlit, rim lighting, chiaroscuro, Rembrandt lighting |
| Atmospheric | volumetric light, god rays, candlelit, moonlit, neon-lit |
| Vintage | soft vintage lighting, tungsten glow, direct flash photography |

### Composition Keywords
| Type | Keywords |
|------|----------|
| Framing | rule of thirds, centered, symmetry, asymmetry, negative space, minimal framing |
| Angle | dutch angle, bird's-eye view, worm's-eye view, over-the-shoulder, POV |
| Distance | macro 1:1, close-up, extreme close-up, medium shot, full-body, ultra-wide |
| Lens Effect | shallow depth of field, deep focus, telephoto compression, bokeh, tilt-shift |

### Color Palette Keywords
| Style | Keywords |
|-------|----------|
| Monochrome | black and white, monochrome, duotone, sepia tone |
| Muted | muted tones, desaturated, low contrast, pastel palette |
| Vibrant | vibrant saturation, jewel tones, neon palette, high contrast |
| Cinematic | teal and orange, cinematic color grade, split toning |
| Themed | warm tones, cool tones, earth tones, complementary colors |

### Environment & Weather Keywords
| Category | Keywords |
|----------|----------|
| Sky | clear sky, dramatic clouds, overcast, stormy sky, aurora borealis, starry night |
| Precipitation | light drizzle, heavy rain, snow flurries, fresh snowfall, blizzard whiteout |
| Atmosphere | foggy, misty, humid haze, smoggy urban air, dust storm, sandstorm |
| Ground | puddles and reflections, frosted ground, morning dew |

### Background & Set Design Keywords
| Style | Keywords |
|-------|----------|
| Studio | seamless white backdrop, pastel colorama, cyclorama studio |
| Urban | neon alleyway, aged brick alley, industrial warehouse, brutalist concrete |
| Interior | art deco interior, mid-century modern, Japanese tatami room, Victorian library |
| Themed | retro 80s arcade, 1950s diner booth, baroque palace hall, gothic cathedral |
| Natural | misty bamboo forest, desert dunes, rocky seaside cliffs, snowy pine forest |

### Fashion Era Keywords
| Era/Style | Keywords |
|-----------|----------|
| Contemporary | streetwear, minimalist, athleisure, smart casual, business formal |
| Decades | retro 70s, vintage 90s, 2000s Y2K, 1920s flapper |
| Subculture | punk style, grunge aesthetic, gothic fashion, bohemian |
| Fantasy | cyberpunk attire, steampunk outfit, fantasy armor, sci-fi bodysuit |
| Haute | haute couture, avant-garde fashion |
