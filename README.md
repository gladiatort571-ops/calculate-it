# calculate-it

A terminal calculator for math, physics, and chemistry. CLI command: `c`.

## Installation

```bash
pip install .
# with AI support:
pip install ".[ai]"
```

For development (tests):

```bash
pip install ".[dev]"
pytest
```

## Quick start

```bash
c "2x+3=7"                    # math -> x = 2
c "m=1.5, v=8.94, ke"          # physics -> ke = 59.9427 J
c "ch4"                        # chemistry -> compound properties
c "enthalpy of ch4 + o2"       # AI -> [ai] -890.8 kJ/mol
```

The mode (math / physics / chemistry) is **auto-detected**, so you normally
don't need any flags. Offline engines answer first; if they can't, the AI
engine answers instead and marks the output with `[ai]`.

## AI configuration

AI features need a Gemini API key:

```bash
export GEMINI_API_KEY="your-key-here"
```

Without it, everything except AI-triggered requests still works.

## CLI reference

```
c [-p | -c] [-g | -o | -r] [-a] ["expression"]
```

| Flag | Meaning |
| ---- | ------- |
| `-p`, `--physics`   | Force physics mode |
| `-c`, `--chemistry` | Force chemistry mode |
| `-g`, `--general`   | (with `-c`) general chemistry lookup |
| `-o`, `--organic`   | (with `-o`) organic chemistry lookup (organic properties first) |
| `-r`, `--redox`     | (with `-r`) redox/equilibrium view (Eo values, half-reactions) |
| `-a`, `--ai`        | Force the AI to answer |
| `-v`, `--version`   | Print version |

### AI trigger keywords

If the expression contains words like `solve`, `find`, `balance`,
`calculate`, `determine`, `explain`, `derive`, ... the request goes
straight to the AI engine and the answer is prefixed with `[ai]`.
Answers are brief (result only) by default; include `explain`, `steps`,
`why`, etc. to get a step-by-step answer.

---

## Math mode (default)

Priority variables: `x, y, z, t`. Equations with several variables are
solved for each separately unless you pick one with a trailing comma:

```
c "xy - 1 = 0"     -> x = 1/y; y != 0 | y = 1/x; x != 0
c "xy - 1 = 0, x"  -> x = 1/y; y != 0
```

Imaginary unit is `i`.

### Minimal use of parentheses

| Input | Parsed as |
| ----- | --------- |
| `ax+b` | `a * x + b` |
| `a x+1` | `a * (x + 1)` |
| `sinax+b` | `sin(a * x + b)` |
| `sinax + b` | `sin(a * x) + b` |
| `sinax + cosbx` | `sin(a * x) + cos(b * x)` |

### Direct equation solver (implicit & explicit)

```
c "2x=3"        -> x = 3/2 | x = 1.5
c "4x+1 = 3"    -> x = 1/2 | x = 0.5
c "a + b = 2"   -> a = 2 - b | b = 2 - a
c "sinax = 1"   -> x = pi/2 + pi*k (rad) | x = 90 + 180*k (deg)
```

### Constants (9 digits)

`pi = 3.14159265`, `eu = 2.71828183`, `tau = 6.28318531`

### Power expressions

```
c "2x3 = 128"        -> 2 * x**3 = 128 -> x = 4
c "axn + b = 1"      -> a * x**n + b = 1
c "2e3 - 1e2x = 0"   -> 2*10**3 - 10**2*x = 0 -> x = 20
c "cos2t"            -> cos(2*t)
c "cos2 t+a"         -> cos(t + a)**2
c "cos2 2t + a"      -> cos(2*t)**2 + a
```

---

## Physics mode (`-p`, or auto-detected)

SI units by default. Values are passed as `name=value` pairs
separated by commas; the last item is what to solve for. If you omit
the target, everything solvable is computed.

```bash
c -p "m=1.5, v=8.94, ke"     # -> ke = 59.9427 J
c -p "m=1.5, v=8.94"         # -> p = ... | ke = ... | ...
c -p "I=2, R=10, V"          # -> V = 20 V
```

Symbols such as `P` (pressure vs power) are disambiguated by context;
ambiguous cases are handed to the AI.

Supported formulas include: `ke = 1/2 m v^2`, `pe = mgh`, `p = mv`,
`F = ma` (and Lorentz/Coulomb/gravitation/Hooke forms), `E = mc^2`,
`E = hf`, `V = IR`, `P = VI`, `v = d/t`, `a = v/t`, `W = Fd`,
`l = mvr`, `lambda = v/f`, ideal gas law forms, and more.

Constants include `me`, `mp`, `mn`, `amu`, `ro`, `Eo`, `h`, `hb`,
`qe`, `mu`, `e`, `lp`, `tp`, `c`, `R`, `g` (e.g. `c -p "c"` prints
`c = 299792458 m/s`).

---

## Chemistry mode (`-c`, or auto-detected)

CGS units (cm, g). Molarity `M`, moles `n`, pressure `P`, density `p`,
masses `m`, volumes `v/V`, concentration `c`, `na = 6.023e23`,
`R = 8.34e-3 kJ/mol.K`.

```bash
c -c "n=2, V=0.5, M"      # -> M = 4 mol/L
c -c "ch4"                # general compound info
```

### General chemistry (`-c -g`)

Bare formula/element lookup plus reaction hints:

```bash
c -c -g "h2o"       # compound properties
c -c -g "h + o"     # known reactions for those elements
```

### Organic chemistry (`-c -o`)

Organic properties first; reaction examples like:

```bash
c -c -o "ch4 + o2"  # CH4 + 2O2 -> 2H2O + CO2 | exothermic
```

### Redox & equilibrium (`-c -r`)

```bash
c -c -r "fe"        # element info incl. Eo = -0.44 V, oxidation states
```

Shows standard reduction potentials, pH/equilibrium details, and
k-factor info where available.

---

## How the routing works

1. If `-p` or `-c` is passed, that mode is used.
2. Otherwise, quick keyword hints (chemical formulas, units, etc.)
   pick a mode; if still unclear, the AI classifies it as
   `math` / `physics` / `chemistry`.
3. AI-trigger keywords (`solve`, `find`, `balance`, ...) route the
   whole request to the AI engine (`[ai]` prefix on the answer).
4. Otherwise the offline engine answers; if it can't, the AI answers
   as a fallback (again prefixed with `[ai]`).

## License

GPL-3.0 — see [LICENSE](LICENSE).
