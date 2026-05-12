---
name: Gas Dynamics Analytics
colors:
  surface: '#f9f9f9'
  surface-dim: '#dadada'
  surface-bright: '#f9f9f9'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f3f3f3'
  surface-container: '#eeeeee'
  surface-container-high: '#e8e8e8'
  surface-container-highest: '#e2e2e2'
  on-surface: '#1a1c1c'
  on-surface-variant: '#454652'
  inverse-surface: '#2f3131'
  inverse-on-surface: '#f1f1f1'
  outline: '#767683'
  outline-variant: '#c6c5d4'
  surface-tint: '#4c56af'
  primary: '#000666'
  on-primary: '#ffffff'
  primary-container: '#1a237e'
  on-primary-container: '#8690ee'
  inverse-primary: '#bdc2ff'
  secondary: '#1b6d24'
  on-secondary: '#ffffff'
  secondary-container: '#a0f399'
  on-secondary-container: '#217128'
  tertiary: '#001944'
  on-tertiary: '#ffffff'
  tertiary-container: '#002c6e'
  on-tertiary-container: '#6b95f3'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#e0e0ff'
  primary-fixed-dim: '#bdc2ff'
  on-primary-fixed: '#000767'
  on-primary-fixed-variant: '#343d96'
  secondary-fixed: '#a3f69c'
  secondary-fixed-dim: '#88d982'
  on-secondary-fixed: '#002204'
  on-secondary-fixed-variant: '#005312'
  tertiary-fixed: '#d9e2ff'
  tertiary-fixed-dim: '#b0c6ff'
  on-tertiary-fixed: '#001945'
  on-tertiary-fixed-variant: '#00429c'
  background: '#f9f9f9'
  on-background: '#1a1c1c'
  surface-variant: '#e2e2e2'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  title-lg:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-numeric:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
  kpi-value:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '700'
    lineHeight: 34px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 8px
  margin-horizontal: 16px
  gutter: 12px
  card-padding: 16px
  section-gap: 24px
---

## Brand & Style

The design system is engineered for precision, reliability, and technical authority. It serves a dual audience of operational managers and data scientists who require immediate, actionable insights into fuel station throughput and queueing bottlenecks.

The visual style follows a **Corporate Modern** approach, leveraging the structured logic of Material Design 3 (React Native Paper). It emphasizes clarity over decoration, using a rigorous information hierarchy to make complex mathematical models—such as M/M/1 or M/M/c queues—instantly digestible. The aesthetic is "analytical-first," characterized by high-density data displays, crisp functional iconography, and a sober professional tone that instills confidence in the underlying calculations.

## Colors

The palette for this design system is rooted in high-contrast corporate tones to ensure legibility in various lighting conditions, including outdoor environments.

- **Primary (Deep Corporate Blue):** Used for structural elements, headers, and primary actions. It represents stability and institutional knowledge.
- **Secondary (Energy Green):** Reserved exclusively for positive metrics, "Optimal" state indicators, and successful simulation completions.
- **Neutral (Cool Grays):** The background utilizes a clean #F5F5F5 to reduce eye strain, while various gray scales differentiate between secondary text and disabled states.
- **Surface:** Pure white is used for cards and interactive inputs to provide a clear "paper" metaphor against the neutral background.

## Typography

This design system utilizes **Inter** as the primary typeface due to its exceptional legibility and neutral, systematic appearance. For mathematical output and tabular data, **JetBrains Mono** is introduced to ensure that numbers and formulas align perfectly, aiding in vertical scanning of data sets.

- **Headlines:** Use tight letter-spacing and bold weights to ground the page.
- **KPI Values:** Specifically scaled for rapid scanning; these should always be the most prominent typographic element on a results screen.
- **Formulas:** Rendered in a slightly reduced size with increased line height to accommodate superscripts and subscripts common in Queue Theory notation.

## Layout & Spacing

The layout utilizes a **fluid grid** system based on an 8px square baseline. 

- **Margins:** A consistent 16px lateral margin is maintained on mobile to ensure content doesn't feel cramped.
- **Stacking:** Vertical spacing follows a 4px/8px/16px/24px scale. Use 24px to separate distinct analytical modules (e.g., Input Parameters vs. Results).
- **KPI Grid:** Results are displayed in a 2-column grid within cards, providing a balanced view of "Wait Time" vs "System Utilization."

## Elevation & Depth

In alignment with Material Design 3, this design system uses **Tonal Layers** supplemented by subtle ambient shadows to define depth.

- **Level 0 (Background):** #F5F5F5. Non-interactive surface.
- **Level 1 (Cards/Inputs):** White surface with a 1px soft border (#E0E0E0) and a low-opacity shadow (Y: 2, Blur: 4, Opacity: 0.05).
- **Level 2 (Active States/FAB):** Elevated with a more pronounced shadow to indicate interactivity.
- **Depth Cues:** Use "Energy Green" or "Deep Blue" subtle glows (2-4% opacity) behind primary KPI cards to denote "Live" calculation states.

## Shapes

The design system employs a **Rounded** (8px) corner strategy. This provides a professional yet modern feel that is less aggressive than sharp corners but more formal than fully pill-shaped components.

- **Small Components:** Checkboxes and small chips use 4px (Soft) radius.
- **Medium Components:** Cards, Segmented Buttons, and Input Fields use 8px (Rounded).
- **Large Components:** Floating Action Buttons (FAB) and Bottom Sheets use 16px (Rounded-LG) or full-pill shapes where appropriate for Material 3 compliance.

## Components

### Buttons & Inputs
- **Segmented Buttons:** Used for toggling between queue models (e.g., M/M/1, M/G/1). The active segment uses the Primary color with white text.
- **Text Inputs:** Outlined Material style. Labels should always be visible (floating) to ensure the user doesn't lose track of variables like "Arrival Rate ($\lambda$)."

### Cards & KPIs
- **KPI Cards:** Feature a top-aligned label in `body-md` (Gray) and a centered, large value in `kpi-value`. If the metric exceeds a threshold, the value color shifts to `error_color_hex`.
- **Analytical Cards:** Grouped inputs or results housed in white containers with `rounded-lg` corners.

### Navigation & Actions
- **FAB:** A prominent circular button in the bottom right, using the Primary color. It is reserved for "Run Simulation" or "Add Station."
- **Data Tables:** Used for "Queue State History." Rows should have a subtle zebra-stripe at 2% opacity for horizontal tracking.

### Feedback
- **Progress Bars:** Use a linear "Energy Green" bar to show simulation progress. 
- **Snackbars:** Minimalist, used for confirming "Model Saved" or "Data Exported."