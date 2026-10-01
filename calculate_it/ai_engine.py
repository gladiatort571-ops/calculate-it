"""AI mode engine for calculate-it.

Uses Google's Gemini API to process queries that cannot be handled by offline engines.
Requires GEMINI_API_KEY environment variable (or a default key can be configured).
"""

import os
import re
from typing import Optional


# Keywords that automatically trigger AI mode when present in the input.
# Matching is case-insensitive and whole-word, so "solve" triggers AI mode
# but "xsolvent" does not.
AI_TRIGGER_KEYWORDS = [
    "solve", "find", "balance", "calculate", "compute", "determine",
    "what", "whats", "how", "why", "explain", "derive", "prove",
    "simplify", "evaluate", "integrate", "differentiate", "predict",
    "estimate", "show", "help",
]

# Marker prepended to every AI-driven answer so the user can tell it apart
# from an offline-engine answer.
AI_FLAG = "[ai]"


def _strip_markdown(text: str) -> str:
    """Convert Markdown/LaTeX-flavored text to plain text."""
    if not text:
        return text
    # Remove LaTeX math delimiters and display-math blocks
    text = re.sub(r"\$\$(.+?)\$\$", r"\1", text, flags=re.DOTALL)
    text = text.replace("$", "")
    # Remove inline code backticks and code fences
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = text.replace("`", "")
    # Remove bold/italic markers
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", text)
    # Remove heading markers
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Horizontal rules
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Convert unordered list bullets to '-'
    text = re.sub(r"^\s*[*•]\s+", "- ", text, flags=re.MULTILINE)
    # Mathfrac -> (a / b)
    text = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1 / \2)", text)
    # Common LaTeX commands
    text = text.replace("\\times", "*").replace("\\cdot", "*")
    text = text.replace("\\pi", "pi").replace("\\approx", "~=")
    text = text.replace("\\neq", "!=").replace("\\leq", "<=").replace("\\geq", ">=")
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
    text = text.replace("{", "(").replace("}", ")")
    # Drop leftover backslashes
    text = text.replace("\\", "")
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def should_use_ai(expression: str) -> bool:
    """Return True when the expression contains an AI trigger keyword."""
    if not expression:
        return False
    for keyword in AI_TRIGGER_KEYWORDS:
        if re.search(r"\b" + re.escape(keyword) + r"\b", expression, re.IGNORECASE):
            return True
    return False


class AIEngine:
    """Engine for AI-powered calculations using Google's Gemini API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.1-flash-lite"):
        """Initialize the AI engine.

        Args:
            api_key: Gemini API key. If None, uses GEMINI_API_KEY env var.
            model: The Gemini model to use.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self._client = None
        self._config = None

    def _get_client(self):
        """Lazy-load the Gemini client."""
        if self._client is None:
            try:
                from google import genai
                from google.genai import types
                if not self.api_key:
                    raise RuntimeError(
                        "No Gemini API key found. Set the GEMINI_API_KEY environment variable."
                    )
                self._client = genai.Client(api_key=self.api_key)
                # Disable automatic function calling (we use no tools) to
                # silence the SDK's AFC deprecation warning.
                self._config = types.GenerateContentConfig(
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                )
            except ImportError:
                raise RuntimeError(
                    "The 'google-genai' package is required for AI mode. "
                    "Install it with: pip install google-genai"
                )
        return self._client

    # Quick deterministic hints so obvious problems never misclassify.
    _CHEM_HINTS = (
        "enthalpy", "entropy", "reaction", "combustion", "molarity", "mole", "mol",
        "balance", "balancing", "redox", "oxidat", "chemist", "delta_h", "deltah",
        "stoichiometr", "organic", "acid", "base", "ph ", "element", "compound",
        "methane", "ethanol", "benzene", "oxygen", "hydrogen", "nitrogen", "sodium",
        "chlorine", "iron", "copper", "calcium", "potassium", "magnesium", "zinc",
    )
    _PHYS_HINTS = (
        "velocity", "momentum", "acceleration", "kinetic", "potential energy",
        "force", "voltage", "resistance", "ohm", "current", "wavelength",
        "frequency", "capacitance", "impedance", "magnetic", "electric field",
        "newton", "joule", "watt", "pascal", "density", "friction", "gravity",
        "ke", "pe", " m/s", "kg", "energy",
    )
    _CHEM_FORMULA = re.compile(r"\b[a-z]{0,3}\d*[A-Z]?[a-z]?\d+\b\s*(\+|-->|->)|\b(ch4|h2o|o2|co2|n2|h2|na|cl2|hcl|naoh|co|nh3|c6h\w*|so2|h2so4)\b", re.IGNORECASE)

    def classify(self, expression: str) -> str:
        """Classify a problem as 'math', 'physics', or 'chemistry'.

        Uses deterministic hints first for obvious cases, then the AI.
        Falls back to 'math' on any error.
        """
        text = expression.lower()
        chem_hit = any(h in text for h in self._CHEM_HINTS) or bool(self._CHEM_FORMULA.search(expression))
        phys_hit = any(h in text for h in self._PHYS_HINTS)
        if chem_hit and not phys_hit:
            return "chemistry"
        if phys_hit and not chem_hit:
            return "physics"
        try:
            client = self._get_client()
            prompt = (
                "Classify the following problem into exactly one category: math, physics, or chemistry.\n"
                "Rules of thumb:\n"
                "- chemistry: chemical formulas (ch4, o2, h2o, na, co2), reactions, combustion, enthalpy, entropy, molarity, moles, balancing equations, redox, gases (R, T, P, V)\n"
                "- physics: mechanics/electromagnetism quantities with units (m/s, kg, J, N, V, A, ke, pe, momentum, velocity, acceleration), constants like c, G\n"
                "- math: pure algebra, equations with variables, calculus, trigonometry, arithmetic\n"
                "When in doubt between math and a science with chemical formulas, prefer chemistry for formulas and physics for units.\n"
                "Reply with only that single lowercase word, nothing else.\n\n"
                f"Problem: {expression}"
            )
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self._config,
            )
            text = (response.text or "").strip().lower()
            for mode in ("physics", "chemistry", "math"):
                if mode in text:
                    return mode
            return "math"
        except Exception:
            return "math"

    def solve(self, expression: str, mode: str = "math") -> str:
        """Solve a query using Gemini AI.

        Args:
            expression: The user's query.
            mode: The mode context ('math', 'physics', or 'chemistry').

        Returns:
            String representation of the result.
        """
        try:
            client = self._get_client()
            prompt = self._build_prompt(expression, mode)
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self._config,
            )
            text = _strip_markdown(response.text)
            if text.lstrip().startswith(AI_FLAG):
                return text
            return f"{AI_FLAG}\n{text}"
        except Exception as e:
            return f"Error: Could not get AI response: {e}"

    def _build_prompt(self, expression: str, mode: str) -> str:
        """Build a context-aware prompt for the AI."""
        context = {
            "math": "You are a mathematics expert. Solve the following math problem.",
            "physics": "You are a physics expert. Solve the following physics problem, including units.",
            "chemistry": "You are a chemistry expert. Solve the following chemistry problem.",
        }
        wants_explanation = any(
            kw in expression.lower() for kw in ("explain", "explanation", "steps", "step by step", "derive", "show work", "why")
        )
        if wants_explanation:
            style = "Show your work step by step and explain your reasoning."
        else:
            style = "Be as brief as possible: give only the final result. Do not show steps, working, or extra words unless the user asks to explain."
        return f"{context.get(mode, context['math'])} {style} Answer in plain text only: no markdown, no LaTeX, no code fences.\n\nProblem: {expression}"

    def _answer(self, query: str, mode: str) -> str:
        """Answer a query using my knowledge.

        Args:
            query: The user's query.
            mode: The mode context.

        Returns:
            Answer string.
        """
        query_lower = query.lower().strip()

        if mode == "chemistry":
            return self._answer_chemistry(query_lower)
        elif mode == "physics":
            return self._answer_physics(query_lower)
        else:
            return self._answer_math(query_lower)

    def _answer_chemistry(self, query: str) -> str:
        """Answer chemistry queries."""
        # Check for element combination queries
        if ' and ' in query and ('reaction' in query or 'compound' in query):
            return self._find_reactions_for_elements(query)

        # Check for specific element queries
        elements = ['hydrogen', 'oxygen', 'carbon', 'nitrogen', 'sodium', 'chlorine',
                   'iron', 'copper', 'silver', 'gold', 'aluminum', 'calcium', 'potassium',
                   'magnesium', 'zinc', 'sulfur', 'phosphorus', 'fluorine', 'bromine', 'iodine']

        for elem in elements:
            if elem in query:
                return self._get_element_reactions(elem)

        # Default chemistry response
        return self._get_general_chemistry_info(query)

    def _answer_physics(self, query: str) -> str:
        """Answer physics queries."""
        return f"Physics query: {query}\n\nI can help with physics calculations. Please provide specific values and the quantity you want to solve for."

    def _answer_math(self, query: str) -> str:
        """Answer math queries."""
        return f"Math query: {query}\n\nI can help with mathematical calculations. Please provide a specific expression or equation."

    def _find_reactions_for_elements(self, query: str) -> str:
        """Find reactions involving specific elements."""
        # Parse elements from query
        elements = []
        element_names = {
            'hydrogen': 'H', 'oxygen': 'O', 'carbon': 'C', 'nitrogen': 'N',
            'sodium': 'Na', 'chlorine': 'Cl', 'iron': 'Fe', 'copper': 'Cu',
            'silver': 'Ag', 'gold': 'Au', 'aluminum': 'Al', 'calcium': 'Ca',
            'potassium': 'K', 'magnesium': 'Mg', 'zinc': 'Zn', 'sulfur': 'S',
            'phosphorus': 'P', 'fluorine': 'F', 'bromine': 'Br', 'iodine': 'I'
        }

        for name, symbol in element_names.items():
            if name in query:
                elements.append((name, symbol))

        if not elements:
            return "Could not identify elements in your query."

        # Generate reactions based on elements
        reactions = self._generate_reactions(elements)
        return reactions

    def _generate_reactions(self, elements: list) -> str:
        """Generate known reactions for given elements."""
        # Common reactions database (built-in knowledge)
        reactions_db = {
            ('H', 'O'): [
                "2H2 + O2 → 2H2O (combustion of hydrogen)",
                "2H2O → 2H2 + O2 (electrolysis of water)",
                "H2 + ½O2 → H2O (formation of water)",
                "CH4 + 2O2 → CO2 + 2H2O (combustion of methane)",
                "C6H12O6 + 6O2 → 6CO2 + 6H2O (cellular respiration)",
                "2H2O2 → 2H2O + O2 (decomposition of hydrogen peroxide)",
            ],
            ('Na', 'Cl'): [
                "2Na + Cl2 → 2NaCl (formation of table salt)",
                "NaCl → Na + ½Cl2 (electrolysis of molten salt)",
                "NaOH + HCl → NaCl + H2O (neutralization)",
            ],
            ('Fe', 'O'): [
                "4Fe + 3O2 → 2Fe2O3 (rusting of iron)",
                "2Fe + O2 → 2FeO (formation of iron(II) oxide)",
                "3Fe + 2O2 → Fe3O4 (formation of magnetite)",
            ],
            ('C', 'O'): [
                "C + O2 → CO2 (complete combustion)",
                "2C + O2 → 2CO (incomplete combustion)",
                "CO2 + C → 2CO (Boudouard reaction)",
                "CaCO3 → CaO + CO2 (thermal decomposition)",
            ],
            ('N', 'H'): [
                "N2 + 3H2 → 2NH3 (Haber process)",
                "NH3 + H2O → NH4+ + OH- (ammonia in water)",
                "2NH3 + H2SO4 → (NH4)2SO4 (ammonium sulfate formation)",
            ],
            ('S', 'O'): [
                "S + O2 → SO2 (combustion of sulfur)",
                "2SO2 + O2 → 2SO3 (contact process)",
                "SO3 + H2O → H2SO4 (sulfuric acid formation)",
            ],
            ('P', 'O'): [
                "4P + 5O2 → 2P2O5 (combustion of phosphorus)",
                "P2O5 + 3H2O → 2H3PO4 (phosphoric acid formation)",
            ],
            ('Ca', 'O'): [
                "2Ca + O2 → 2CaO (formation of calcium oxide)",
                "CaO + H2O → Ca(OH)2 (slaking of lime)",
                "CaCO3 → CaO + CO2 (thermal decomposition)",
            ],
            ('K', 'O'): [
                "4K + O2 → 2K2O (formation of potassium oxide)",
                "2K + O2 → K2O2 (formation of potassium peroxide)",
                "K2O + H2O → 2KOH (potassium hydroxide formation)",
            ],
            ('Mg', 'O'): [
                "2Mg + O2 → 2MgO (combustion of magnesium)",
                "MgO + H2O → Mg(OH)2 (magnesium hydroxide formation)",
            ],
            ('Zn', 'O'): [
                "2Zn + O2 → 2ZnO (formation of zinc oxide)",
                "ZnO + 2HCl → ZnCl2 + H2O (acid reaction)",
            ],
            ('Al', 'O'): [
                "4Al + 3O2 → 2Al2O3 (formation of aluminum oxide)",
                "2Al2O3 → 4Al + 3O2 (Hall-Heroult process)",
            ],
            ('Cu', 'O'): [
                "2Cu + O2 → 2CuO (formation of copper(II) oxide)",
                "CuO + H2 → Cu + H2O (reduction of copper oxide)",
            ],
            ('Ag', 'O'): [
                "4Ag + O2 → 2Ag2O (formation of silver oxide)",
                "2Ag2O → 4Ag + O2 (decomposition)",
            ],
            ('H', 'Cl'): [
                "H2 + Cl2 → 2HCl (formation of hydrogen chloride)",
                "HCl + NaOH → NaCl + H2O (neutralization)",
            ],
            ('H', 'N'): [
                "N2 + 3H2 → 2NH3 (Haber process)",
                "NH3 + HCl → NH4Cl (ammonium chloride formation)",
            ],
            ('H', 'S'): [
                "H2 + S → H2S (formation of hydrogen sulfide)",
                "H2S + 2O2 → H2SO4 (oxidation to sulfuric acid)",
            ],
            ('H', 'F'): [
                "H2 + F2 → 2HF (formation of hydrogen fluoride)",
                "HF + NaOH → NaF + H2O (neutralization)",
            ],
            ('H', 'Br'): [
                "H2 + Br2 → 2HBr (formation of hydrogen bromide)",
                "HBr + NaOH → NaBr + H2O (neutralization)",
            ],
            ('H', 'I'): [
                "H2 + I2 → 2HI (formation of hydrogen iodide)",
                "HI + NaOH → NaI + H2O (neutralization)",
            ],
            ('C', 'H'): [
                "C + 2H2 → CH4 (formation of methane)",
                "CH4 + Cl2 → CH3Cl + HCl (halogenation)",
                "C2H4 + H2 → C2H6 (hydrogenation)",
            ],
            ('C', 'N'): [
                "C + N2 → (CN)2 (formation of cyanogen)",
                "HCN + NaOH → NaCN + H2O (neutralization)",
            ],
            ('C', 'S'): [
                "C + 2S → CS2 (formation of carbon disulfide)",
                "CS2 + 3O2 → CO2 + 2SO2 (combustion)",
            ],
            ('N', 'O'): [
                "N2 + O2 → 2NO (formation of nitric oxide)",
                "2NO + O2 → 2NO2 (formation of nitrogen dioxide)",
                "4NH3 + 5O2 → 4NO + 6H2O (Ostwald process)",
            ],
            ('N', 'Cl'): [
                "NCl3 + 3H2O → NH3 + 3HOCl (hydrolysis)",
            ],
            ('S', 'H'): [
                "S + H2 → H2S (formation of hydrogen sulfide)",
                "H2S + 2O2 → SO2 + H2O (combustion)",
            ],
            ('S', 'Cl'): [
                "S + Cl2 → S2Cl2 (formation of disulfur dichloride)",
                "S2Cl2 + 2H2O → 2HCl + H2S + SO2 (hydrolysis)",
            ],
            ('P', 'H'): [
                "2P + 3H2 → 2PH3 (formation of phosphine)",
                "PH3 + 2O2 → H3PO4 (combustion)",
            ],
            ('P', 'Cl'): [
                "2P + 3Cl2 → 2PCl3 (formation of phosphorus trichloride)",
                "PCl3 + Cl2 → PCl5 (formation of phosphorus pentachloride)",
            ],
            ('P', 'F'): [
                "2P + 3F2 → 2PF3 (formation of phosphorus trifluoride)",
            ],
            ('Si', 'O'): [
                "Si + O2 → SiO2 (formation of silicon dioxide)",
                "SiO2 + 2C → Si + 2CO (carbothermic reduction)",
            ],
            ('Si', 'H'): [
                "Si + 2H2 → SiH4 (formation of silane)",
                "SiH4 + 2O2 → SiO2 + 2H2O (combustion)",
            ],
            ('Si', 'Cl'): [
                "Si + 2Cl2 → SiCl4 (formation of silicon tetrachloride)",
                "SiCl4 + 2H2O → SiO2 + 4HCl (hydrolysis)",
            ],
            ('B', 'O'): [
                "4B + 3O2 → 2B2O3 (formation of boron oxide)",
                "B2O3 + 3H2O → 2H3BO3 (formation of boric acid)",
            ],
            ('B', 'H'): [
                "2B + 3H2 → 2BH3 (formation of borane)",
                "BH3 + 3O2 → B2O3 + 3H2O (combustion)",
            ],
            ('B', 'N'): [
                "B + N2 → 2BN (formation of boron nitride)",
            ],
        }

        # Find matching reactions
        elem_symbols = tuple(sorted([e[1] for e in elements]))
        reactions = []

        # Check for exact match
        if elem_symbols in reactions_db:
            reactions = reactions_db[elem_symbols]
        else:
            # Check for reverse match
            for key, value in reactions_db.items():
                if set(key) == set(elem_symbols):
                    reactions = value
                    break

        if not reactions:
            return f"No specific reactions found for {' and '.join([e[0] for e in elements])}. Try a different combination."

        # Format the output
        result = f"Known reactions involving {' and '.join([e[0] for e in elements])}:\n\n"
        for i, reaction in enumerate(reactions, 1):
            result += f"{i}. {reaction}\n"

        return result

    def _get_element_reactions(self, element: str) -> str:
        """Get reactions for a specific element."""
        element_reactions = {
            'hydrogen': [
                "2H2 + O2 → 2H2O (combustion)",
                "N2 + 3H2 → 2NH3 (Haber process)",
                "H2 + Cl2 → 2HCl (hydrogen chloride formation)",
                "C + 2H2 → CH4 (methane formation)",
                "2H2O → 2H2 + O2 (electrolysis)",
            ],
            'oxygen': [
                "2H2 + O2 → 2H2O (combustion of hydrogen)",
                "C + O2 → CO2 (complete combustion)",
                "CH4 + 2O2 → CO2 + 2H2O (methane combustion)",
                "2SO2 + O2 → 2SO3 (contact process)",
                "4Fe + 3O2 → 2Fe2O3 (rusting)",
            ],
            'carbon': [
                "C + O2 → CO2 (complete combustion)",
                "2C + O2 → 2CO (incomplete combustion)",
                "CO2 + C → 2CO (Boudouard reaction)",
                "CaCO3 → CaO + CO2 (thermal decomposition)",
                "CH4 + 2O2 → CO2 + 2H2O (methane combustion)",
            ],
            'nitrogen': [
                "N2 + 3H2 → 2NH3 (Haber process)",
                "N2 + O2 → 2NO (nitric oxide formation)",
                "4NH3 + 5O2 → 4NO + 6H2O (Ostwald process)",
                "2NO + O2 → 2NO2 (nitrogen dioxide formation)",
            ],
            'sodium': [
                "2Na + Cl2 → 2NaCl (table salt formation)",
                "2Na + 2H2O → 2NaOH + H2 (reaction with water)",
                "NaCl → Na + ½Cl2 (electrolysis)",
            ],
            'chlorine': [
                "H2 + Cl2 → 2HCl (hydrogen chloride formation)",
                "2Na + Cl2 → 2NaCl (sodium chloride formation)",
                "Cl2 + 2NaOH → NaCl + NaClO + H2O (disproportionation)",
            ],
            'iron': [
                "4Fe + 3O2 → 2Fe2O3 (rusting)",
                "Fe + 2HCl → FeCl2 + H2 (acid reaction)",
                "Fe + CuSO4 → FeSO4 + Cu (displacement)",
            ],
            'copper': [
                "2Cu + O2 → 2CuO (copper oxide formation)",
                "CuO + H2 → Cu + H2O (reduction)",
                "Cu + 2AgNO3 → Cu(NO3)2 + 2Ag (displacement)",
            ],
            'silver': [
                "4Ag + O2 → 2Ag2O (silver oxide formation)",
                "Ag + 2HNO3 → AgNO3 + NO2 + H2O (nitric acid reaction)",
            ],
            'gold': [
                "Au + 4HCl + HNO3 → AuCl4- + NO + 2H2O (aqua regia)",
            ],
            'aluminum': [
                "4Al + 3O2 → 2Al2O3 (aluminum oxide formation)",
                "2Al + 6HCl → 2AlCl3 + 3H2 (acid reaction)",
            ],
            'calcium': [
                "2Ca + O2 → 2CaO (calcium oxide formation)",
                "CaO + H2O → Ca(OH)2 (slaking of lime)",
                "CaCO3 → CaO + CO2 (thermal decomposition)",
            ],
            'potassium': [
                "4K + O2 → 2K2O (potassium oxide formation)",
                "2K + 2H2O → 2KOH + H2 (reaction with water)",
            ],
            'magnesium': [
                "2Mg + O2 → 2MgO (combustion)",
                "Mg + 2HCl → MgCl2 + H2 (acid reaction)",
            ],
            'zinc': [
                "2Zn + O2 → 2ZnO (zinc oxide formation)",
                "Zn + 2HCl → ZnCl2 + H2 (acid reaction)",
                "Zn + CuSO4 → ZnSO4 + Cu (displacement)",
            ],
            'sulfur': [
                "S + O2 → SO2 (combustion)",
                "2SO2 + O2 → 2SO3 (contact process)",
                "SO3 + H2O → H2SO4 (sulfuric acid formation)",
            ],
            'phosphorus': [
                "4P + 5O2 → 2P2O5 (combustion)",
                "P2O5 + 3H2O → 2H3PO4 (phosphoric acid formation)",
            ],
            'fluorine': [
                "H2 + F2 → 2HF (hydrogen fluoride formation)",
                "2F2 + 2H2O → 4HF + O2 (reaction with water)",
            ],
            'bromine': [
                "H2 + Br2 → 2HBr (hydrogen bromide formation)",
                "2Na + Br2 → 2NaBr (sodium bromide formation)",
            ],
            'iodine': [
                "H2 + I2 → 2HI (hydrogen iodide formation)",
                "2Na + I2 → 2NaI (sodium iodide formation)",
            ],
        }

        reactions = element_reactions.get(element, [])
        if not reactions:
            return f"No specific reactions found for {element}."

        result = f"Common reactions involving {element}:\n\n"
        for i, reaction in enumerate(reactions, 1):
            result += f"{i}. {reaction}\n"

        return result

    def _get_general_chemistry_info(self, query: str) -> str:
        """Get general chemistry information."""
        return f"Chemistry query: {query}\n\nI can help with chemical reactions, element properties, and compound information. Please be specific about what you want to know."

    def is_available(self) -> bool:
        """Check if AI mode is available (always True for built-in AI)."""
        return True
