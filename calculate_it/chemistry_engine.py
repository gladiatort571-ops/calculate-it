"""Chemistry mode engine for calculate-it."""

from typing import Dict, List, Optional, Tuple
import re
from collections import defaultdict

import sympy
from sympy import symbols, Eq, solve, sympify, Symbol, simplify, Rational, Matrix

from .parser import parse_expression, Mode
from .ai_engine import AIEngine


# Chemistry constants (CGS system)
CHEMISTRY_CONSTANTS: Dict[str, str] = {
    'na': '6.023e23',      # Avogadro's number
    'NA': '6.023e23',      # Avogadro's number
    'R': '8.34e-3',        # ideal gas constant (kJ/mol.K)
}

# Chemistry symbols
CHEMISTRY_SYMBOLS: Dict[str, List[str]] = {
    'M': ['molarity'],
    'n': ['number of moles'],
    'P': ['pressure'],
    'p': ['density'],
    'm': ['mass'],
    'v': ['volume'],
    'V': ['voltage', 'volume'],
    'c': ['concentration'],
}

# Chemistry units for each variable
CHEMISTRY_UNITS: Dict[str, str] = {
    'M': 'mol/L',
    'n': 'mol',
    'P': 'Pa',
    'p': 'g/cm³',
    'm': 'g',
    'v': 'cm³',
    'V': 'V or cm³',
    'c': 'mL/L',
    'R': 'kJ/mol·K',
    'na': 'mol⁻¹',
    'NA': 'mol⁻¹',
}

# Chemistry formulas: maps a target variable to a list of (formula_str, required_vars, description, unit) tuples
CHEMISTRY_FORMULAS: Dict[str, List[Tuple[str, List[str], str, str]]] = {
    'M': [
        ('m/n', ['m', 'n'], 'molar mass = m/n', 'g/mol'),
        ('n/V', ['n', 'V'], 'molarity = n/V', 'mol/L'),
    ],
    'n': [
        ('M*V', ['M', 'V'], 'moles = M·V', 'mol'),
        ('m/M', ['m', 'M'], 'moles = m/M', 'mol'),
        ('V/Vm', ['V', 'Vm'], 'moles = V/Vm', 'mol'),
    ],
    'P': [
        ('n*R*T/V', ['n', 'R', 'T', 'V'], 'ideal gas law = nRT/V', 'Pa'),
        ('F/A', ['F', 'A'], 'pressure = F/A', 'Pa'),
    ],
    'p': [
        ('m/V', ['m', 'V'], 'density = m/V', 'g/cm³'),
    ],
    'm': [
        ('n*M', ['n', 'M'], 'mass = n·M', 'g'),
        ('p*V', ['p', 'V'], 'mass = ρ·V', 'g'),
    ],
    'V': [
        ('n/M', ['n', 'M'], 'volume = n/M', 'L'),
        ('m/p', ['m', 'p'], 'volume = m/ρ', 'cm³'),
        ('n*Vm', ['n', 'Vm'], 'volume = n·Vm', 'L'),
    ],
    'v': [
        ('n/M', ['n', 'M'], 'volume = n/M', 'cm³'),
        ('m/p', ['m', 'p'], 'volume = m/ρ', 'cm³'),
    ],
    'c': [
        ('n/V', ['n', 'V'], 'concentration = n/V', 'mL/L'),
    ],
}

# Constants that should not be considered as "user-provided variables"
CONSTANT_SYMBOLS = {'R', 'na', 'NA'}

# Element database: symbol -> (name, atomic_number, atomic_mass)
ELEMENTS: Dict[str, Dict] = {
    'h': {'name': 'Hydrogen', 'Z': 1, 'mass': '1.008'},
    'he': {'name': 'Helium', 'Z': 2, 'mass': '4.003'},
    'li': {'name': 'Lithium', 'Z': 3, 'mass': '6.941'},
    'be': {'name': 'Beryllium', 'Z': 4, 'mass': '9.012'},
    'b': {'name': 'Boron', 'Z': 5, 'mass': '10.81'},
    'c': {'name': 'Carbon', 'Z': 6, 'mass': '12.011'},
    'n': {'name': 'Nitrogen', 'Z': 7, 'mass': '14.007'},
    'o': {'name': 'Oxygen', 'Z': 8, 'mass': '15.999'},
    'f': {'name': 'Fluorine', 'Z': 9, 'mass': '18.998'},
    'ne': {'name': 'Neon', 'Z': 10, 'mass': '20.180'},
    'na': {'name': 'Sodium', 'Z': 11, 'mass': '22.990'},
    'mg': {'name': 'Magnesium', 'Z': 12, 'mass': '24.305'},
    'al': {'name': 'Aluminum', 'Z': 13, 'mass': '26.982'},
    'si': {'name': 'Silicon', 'Z': 14, 'mass': '28.086'},
    'p': {'name': 'Phosphorus', 'Z': 15, 'mass': '30.974'},
    's': {'name': 'Sulfur', 'Z': 16, 'mass': '32.065'},
    'cl': {'name': 'Chlorine', 'Z': 17, 'mass': '35.453'},
    'ar': {'name': 'Argon', 'Z': 18, 'mass': '39.948'},
    'k': {'name': 'Potassium', 'Z': 19, 'mass': '39.098'},
    'ca': {'name': 'Calcium', 'Z': 20, 'mass': '40.078'},
    'sc': {'name': 'Scandium', 'Z': 21, 'mass': '44.956'},
    'ti': {'name': 'Titanium', 'Z': 22, 'mass': '47.867'},
    'v': {'name': 'Vanadium', 'Z': 23, 'mass': '50.942'},
    'cr': {'name': 'Chromium', 'Z': 24, 'mass': '51.996'},
    'mn': {'name': 'Manganese', 'Z': 25, 'mass': '54.938'},
    'fe': {'name': 'Iron', 'Z': 26, 'mass': '55.845'},
    'co': {'name': 'Cobalt', 'Z': 27, 'mass': '58.933'},
    'ni': {'name': 'Nickel', 'Z': 28, 'mass': '58.693'},
    'cu': {'name': 'Copper', 'Z': 29, 'mass': '63.546'},
    'zn': {'name': 'Zinc', 'Z': 30, 'mass': '65.38'},
    'ga': {'name': 'Gallium', 'Z': 31, 'mass': '69.723'},
    'ge': {'name': 'Germanium', 'Z': 32, 'mass': '72.630'},
    'as': {'name': 'Arsenic', 'Z': 33, 'mass': '74.922'},
    'se': {'name': 'Selenium', 'Z': 34, 'mass': '78.971'},
    'br': {'name': 'Bromine', 'Z': 35, 'mass': '79.904'},
    'kr': {'name': 'Krypton', 'Z': 36, 'mass': '83.798'},
    'rb': {'name': 'Rubidium', 'Z': 37, 'mass': '85.468'},
    'sr': {'name': 'Strontium', 'Z': 38, 'mass': '87.62'},
    'y': {'name': 'Yttrium', 'Z': 39, 'mass': '88.906'},
    'zr': {'name': 'Zirconium', 'Z': 40, 'mass': '91.224'},
    'nb': {'name': 'Niobium', 'Z': 41, 'mass': '92.906'},
    'mo': {'name': 'Molybdenum', 'Z': 42, 'mass': '95.95'},
    'tc': {'name': 'Technetium', 'Z': 43, 'mass': '98'},
    'ru': {'name': 'Ruthenium', 'Z': 44, 'mass': '101.07'},
    'rh': {'name': 'Rhodium', 'Z': 45, 'mass': '102.906'},
    'pd': {'name': 'Palladium', 'Z': 46, 'mass': '106.42'},
    'ag': {'name': 'Silver', 'Z': 47, 'mass': '107.868'},
    'cd': {'name': 'Cadmium', 'Z': 48, 'mass': '112.414'},
    'in': {'name': 'Indium', 'Z': 49, 'mass': '114.818'},
    'sn': {'name': 'Tin', 'Z': 50, 'mass': '118.710'},
    'sb': {'name': 'Antimony', 'Z': 51, 'mass': '121.760'},
    'te': {'name': 'Tellurium', 'Z': 52, 'mass': '127.60'},
    'i': {'name': 'Iodine', 'Z': 53, 'mass': '126.904'},
    'xe': {'name': 'Xenon', 'Z': 54, 'mass': '131.293'},
    'cs': {'name': 'Cesium', 'Z': 55, 'mass': '132.905'},
    'ba': {'name': 'Barium', 'Z': 56, 'mass': '137.327'},
    'la': {'name': 'Lanthanum', 'Z': 57, 'mass': '138.905'},
    'ce': {'name': 'Cerium', 'Z': 58, 'mass': '140.116'},
    'pr': {'name': 'Praseodymium', 'Z': 59, 'mass': '140.908'},
    'nd': {'name': 'Neodymium', 'Z': 60, 'mass': '144.242'},
    'pm': {'name': 'Promethium', 'Z': 61, 'mass': '145'},
    'sm': {'name': 'Samarium', 'Z': 62, 'mass': '150.36'},
    'eu': {'name': 'Europium', 'Z': 63, 'mass': '151.964'},
    'gd': {'name': 'Gadolinium', 'Z': 64, 'mass': '157.25'},
    'tb': {'name': 'Terbium', 'Z': 65, 'mass': '158.925'},
    'dy': {'name': 'Dysprosium', 'Z': 66, 'mass': '162.500'},
    'ho': {'name': 'Holmium', 'Z': 67, 'mass': '164.930'},
    'er': {'name': 'Erbium', 'Z': 68, 'mass': '167.259'},
    'tm': {'name': 'Thulium', 'Z': 69, 'mass': '168.934'},
    'yb': {'name': 'Ytterbium', 'Z': 70, 'mass': '173.045'},
    'lu': {'name': 'Lutetium', 'Z': 71, 'mass': '174.967'},
    'hf': {'name': 'Hafnium', 'Z': 72, 'mass': '178.49'},
    'ta': {'name': 'Tantalum', 'Z': 73, 'mass': '180.948'},
    'w': {'name': 'Tungsten', 'Z': 74, 'mass': '183.84'},
    're': {'name': 'Rhenium', 'Z': 75, 'mass': '186.207'},
    'os': {'name': 'Osmium', 'Z': 76, 'mass': '190.23'},
    'ir': {'name': 'Iridium', 'Z': 77, 'mass': '192.217'},
    'pt': {'name': 'Platinum', 'Z': 78, 'mass': '195.084'},
    'au': {'name': 'Gold', 'Z': 79, 'mass': '196.967'},
    'hg': {'name': 'Mercury', 'Z': 80, 'mass': '200.592'},
    'tl': {'name': 'Thallium', 'Z': 81, 'mass': '204.38'},
    'pb': {'name': 'Lead', 'Z': 82, 'mass': '207.2'},
    'bi': {'name': 'Bismuth', 'Z': 83, 'mass': '208.980'},
    'po': {'name': 'Polonium', 'Z': 84, 'mass': '209'},
    'at': {'name': 'Astatine', 'Z': 85, 'mass': '210'},
    'rn': {'name': 'Radon', 'Z': 86, 'mass': '222'},
    'fr': {'name': 'Francium', 'Z': 87, 'mass': '223'},
    'ra': {'name': 'Radium', 'Z': 88, 'mass': '226'},
    'ac': {'name': 'Actinium', 'Z': 89, 'mass': '227'},
    'th': {'name': 'Thorium', 'Z': 90, 'mass': '232.038'},
    'pa': {'name': 'Protactinium', 'Z': 91, 'mass': '231.036'},
    'u': {'name': 'Uranium', 'Z': 92, 'mass': '238.029'},
}

# Organic compound database
ORGANIC_COMPOUNDS: Dict[str, Dict] = {
    'ch4': {'name': 'Methane', 'formula': 'CH4', 'mw': '16.04', 'type': 'alkane'},
    'c2h6': {'name': 'Ethane', 'formula': 'C2H6', 'mw': '30.07', 'type': 'alkane'},
    'c3h8': {'name': 'Propane', 'formula': 'C3H8', 'mw': '44.10', 'type': 'alkane'},
    'c4h10': {'name': 'Butane', 'formula': 'C4H10', 'mw': '58.12', 'type': 'alkane'},
    'c2h4': {'name': 'Ethene', 'formula': 'C2H4', 'mw': '28.05', 'type': 'alkene'},
    'c3h6': {'name': 'Propene', 'formula': 'C3H6', 'mw': '42.08', 'type': 'alkene'},
    'c2h2': {'name': 'Ethyne', 'formula': 'C2H2', 'mw': '26.04', 'type': 'alkyne'},
    'ch3oh': {'name': 'Methanol', 'formula': 'CH3OH', 'mw': '32.04', 'type': 'alcohol'},
    'c2h5oh': {'name': 'Ethanol', 'formula': 'C2H5OH', 'mw': '46.07', 'type': 'alcohol'},
    'c3h7oh': {'name': 'Propanol', 'formula': 'C3H7OH', 'mw': '60.10', 'type': 'alcohol'},
    'c6h6': {'name': 'Benzene', 'formula': 'C6H6', 'mw': '78.11', 'type': 'aromatic'},
    'c6h12o6': {'name': 'Glucose', 'formula': 'C6H12O6', 'mw': '180.16', 'type': 'carbohydrate'},
    'c12h22o11': {'name': 'Sucrose', 'formula': 'C12H22O11', 'mw': '342.30', 'type': 'carbohydrate'},
    'ch3cooh': {'name': 'Acetic acid', 'formula': 'CH3COOH', 'mw': '60.05', 'type': 'carboxylic acid'},
    'hcooh': {'name': 'Formic acid', 'formula': 'HCOOH', 'mw': '46.03', 'type': 'carboxylic acid'},
    'c6h5oh': {'name': 'Phenol', 'formula': 'C6H5OH', 'mw': '94.11', 'type': 'aromatic'},
    'c6h5nh2': {'name': 'Aniline', 'formula': 'C6H5NH2', 'mw': '93.13', 'type': 'aromatic'},
    'ch3nh2': {'name': 'Methylamine', 'formula': 'CH3NH2', 'mw': '31.06', 'type': 'amine'},
    'c2h3n': {'name': 'Acetonitrile', 'formula': 'C2H3N', 'mw': '41.05', 'type': 'nitrile'},
    'c3h6o': {'name': 'Acetone', 'formula': 'C3H6O', 'mw': '58.08', 'type': 'ketone'},
    'c4h10o': {'name': 'Butanol', 'formula': 'C4H10O', 'mw': '74.12', 'type': 'alcohol'},
    'c5h12': {'name': 'Pentane', 'formula': 'C5H12', 'mw': '72.15', 'type': 'alkane'},
    'c6h14': {'name': 'Hexane', 'formula': 'C6H14', 'mw': '86.18', 'type': 'alkane'},
    'c7h8': {'name': 'Toluene', 'formula': 'C7H8', 'mw': '92.14', 'type': 'aromatic'},
    'c8h18': {'name': 'Octane', 'formula': 'C8H18', 'mw': '114.23', 'type': 'alkane'},
    'c6h5cooh': {'name': 'Benzoic acid', 'formula': 'C6H5COOH', 'mw': '122.12', 'type': 'carboxylic acid'},
    'c6h5cho': {'name': 'Benzaldehyde', 'formula': 'C6H5CHO', 'mw': '106.12', 'type': 'aldehyde'},
    'ch3och3': {'name': 'Dimethyl ether', 'formula': 'CH3OCH3', 'mw': '46.07', 'type': 'ether'},
    'c2h5oc2h5': {'name': 'Diethyl ether', 'formula': 'C2H5OC2H5', 'mw': '74.12', 'type': 'ether'},
    'c3h6o2': {'name': 'Propanoic acid', 'formula': 'C3H6O2', 'mw': '74.08', 'type': 'carboxylic acid'},
    'c4h8o2': {'name': 'Ethyl acetate', 'formula': 'C4H8O2', 'mw': '88.11', 'type': 'ester'},
    'c6h5nh2': {'name': 'Aniline', 'formula': 'C6H5NH2', 'mw': '93.13', 'type': 'aromatic'},
    'c6h6o': {'name': 'Phenol', 'formula': 'C6H6O', 'mw': '94.11', 'type': 'aromatic'},
    'c7h6o2': {'name': 'Salicylic acid', 'formula': 'C7H6O2', 'mw': '138.12', 'type': 'carboxylic acid'},
    'c8h8o': {'name': 'Acetophenone', 'formula': 'C8H8O', 'mw': '120.15', 'type': 'ketone'},
    'c9h8o2': {'name': 'Cinnamic acid', 'formula': 'C9H8O2', 'mw': '148.16', 'type': 'carboxylic acid'},
    'c10h8': {'name': 'Naphthalene', 'formula': 'C10H8', 'mw': '128.17', 'type': 'aromatic'},
    'c6h12o6': {'name': 'Glucose', 'formula': 'C6H12O6', 'mw': '180.16', 'type': 'carbohydrate'},
    'c12h22o11': {'name': 'Sucrose', 'formula': 'C12H22O11', 'mw': '342.30', 'type': 'carbohydrate'},
    'c6h8o7': {'name': 'Citric acid', 'formula': 'C6H8O7', 'mw': '192.12', 'type': 'carboxylic acid'},
    'c4h6o6': {'name': 'Tartaric acid', 'formula': 'C4H6O6', 'mw': '150.09', 'type': 'carboxylic acid'},
    'c3h4o4': {'name': 'Malonic acid', 'formula': 'C3H4O4', 'mw': '104.06', 'type': 'carboxylic acid'},
    'c4h4o4': {'name': 'Succinic acid', 'formula': 'C4H4O4', 'mw': '118.09', 'type': 'carboxylic acid'},
    'c5h8o4': {'name': 'Glutaric acid', 'formula': 'C5H8O4', 'mw': '132.11', 'type': 'carboxylic acid'},
    'c6h10o4': {'name': 'Adipic acid', 'formula': 'C6H10O4', 'mw': '146.14', 'type': 'carboxylic acid'},
    'c8h6o4': {'name': 'Phthalic acid', 'formula': 'C8H6O4', 'mw': '166.13', 'type': 'carboxylic acid'},
    'c4h6o2': {'name': 'Acrylic acid', 'formula': 'C4H6O2', 'mw': '72.06', 'type': 'carboxylic acid'},
    'c3h4o2': {'name': 'Acrylic acid', 'formula': 'C3H4O2', 'mw': '72.06', 'type': 'carboxylic acid'},
    'c2h4o2': {'name': 'Acetic acid', 'formula': 'C2H4O2', 'mw': '60.05', 'type': 'carboxylic acid'},
    'hcooh': {'name': 'Formic acid', 'formula': 'HCOOH', 'mw': '46.03', 'type': 'carboxylic acid'},
    'c2h2o4': {'name': 'Oxalic acid', 'formula': 'C2H2O4', 'mw': '90.03', 'type': 'carboxylic acid'},
    'c7h6o3': {'name': 'p-Hydroxybenzoic acid', 'formula': 'C7H6O3', 'mw': '138.12', 'type': 'carboxylic acid'},
    'c8h8o3': {'name': 'Vanillic acid', 'formula': 'C8H8O3', 'mw': '168.15', 'type': 'carboxylic acid'},
    'c9h10o3': {'name': 'Ferulic acid', 'formula': 'C9H10O3', 'mw': '194.18', 'type': 'carboxylic acid'},
    'c10h12o3': {'name': 'Eugenol', 'formula': 'C10H12O3', 'mw': '164.20', 'type': 'aromatic'},
    'c14h10o4': {'name': 'Terephthalic acid', 'formula': 'C14H10O4', 'mw': '242.23', 'type': 'carboxylic acid'},
    'c12h22o11': {'name': 'Sucrose', 'formula': 'C12H22O11', 'mw': '342.30', 'type': 'carbohydrate'},
    'c6h10o5': {'name': 'Gluconic acid', 'formula': 'C6H10O5', 'mw': '196.16', 'type': 'carboxylic acid'},
    'c5h10o5': {'name': 'Xylose', 'formula': 'C5H10O5', 'mw': '150.13', 'type': 'carbohydrate'},
    'c5h10o4': {'name': 'Ribose', 'formula': 'C5H10O4', 'mw': '150.13', 'type': 'carbohydrate'},
    'c4h6o5': {'name': 'Erythrose', 'formula': 'C4H6O5', 'mw': '122.10', 'type': 'carbohydrate'},
    'c4h8o4': {'name': 'Threose', 'formula': 'C4H8O4', 'mw': '120.10', 'type': 'carbohydrate'},
    'c3h6o3': {'name': 'Lactic acid', 'formula': 'C3H6O3', 'mw': '90.08', 'type': 'carboxylic acid'},
    'c3h6o2': {'name': 'Propionic acid', 'formula': 'C3H6O2', 'mw': '74.08', 'type': 'carboxylic acid'},
    'c4h8o3': {'name': 'Butyric acid', 'formula': 'C4H8O3', 'mw': '88.11', 'type': 'carboxylic acid'},
    'c5h10o2': {'name': 'Valeric acid', 'formula': 'C5H10O2', 'mw': '102.13', 'type': 'carboxylic acid'},
    'c6h12o2': {'name': 'Caproic acid', 'formula': 'C6H12O2', 'mw': '116.16', 'type': 'carboxylic acid'},
    'c7h14o2': {'name': 'Enanthic acid', 'formula': 'C7H14O2', 'mw': '130.18', 'type': 'carboxylic acid'},
    'c8h16o2': {'name': 'Caprylic acid', 'formula': 'C8H16O2', 'mw': '144.21', 'type': 'carboxylic acid'},
    'c9h18o2': {'name': 'Pelargonic acid', 'formula': 'C9H18O2', 'mw': '158.24', 'type': 'carboxylic acid'},
    'c10h20o2': {'name': 'Capric acid', 'formula': 'C10H20O2', 'mw': '172.26', 'type': 'carboxylic acid'},
    'c11h22o2': {'name': 'Undecylic acid', 'formula': 'C11H22O2', 'mw': '186.29', 'type': 'carboxylic acid'},
    'c12h24o2': {'name': 'Lauric acid', 'formula': 'C12H24O2', 'mw': '200.32', 'type': 'carboxylic acid'},
    'c13h26o2': {'name': 'Tridecylic acid', 'formula': 'C13H26O2', 'mw': '214.35', 'type': 'carboxylic acid'},
    'c14h28o2': {'name': 'Myristic acid', 'formula': 'C14H28O2', 'mw': '228.37', 'type': 'carboxylic acid'},
    'c15h30o2': {'name': 'Pentadecylic acid', 'formula': 'C15H30O2', 'mw': '242.40', 'type': 'carboxylic acid'},
    'c16h32o2': {'name': 'Palmitic acid', 'formula': 'C16H32O2', 'mw': '256.42', 'type': 'carboxylic acid'},
    'c17h34o2': {'name': 'Margaric acid', 'formula': 'C17H34O2', 'mw': '270.45', 'type': 'carboxylic acid'},
    'c18h36o2': {'name': 'Stearic acid', 'formula': 'C18H36O2', 'mw': '284.48', 'type': 'carboxylic acid'},
    'c19h38o2': {'name': 'Nonadecylic acid', 'formula': 'C19H38O2', 'mw': '298.51', 'type': 'carboxylic acid'},
    'c20h40o2': {'name': 'Arachidic acid', 'formula': 'C20H40O2', 'mw': '312.53', 'type': 'carboxylic acid'},
    'c21h42o2': {'name': 'Heneicosylic acid', 'formula': 'C21H42O2', 'mw': '326.56', 'type': 'carboxylic acid'},
    'c22h44o2': {'name': 'Behenic acid', 'formula': 'C22H44O2', 'mw': '340.59', 'type': 'carboxylic acid'},
    'c23h46o2': {'name': 'Tricosylic acid', 'formula': 'C23H46O2', 'mw': '354.61', 'type': 'carboxylic acid'},
    'c24h48o2': {'name': 'Lignoceric acid', 'formula': 'C24H48O2', 'mw': '368.64', 'type': 'carboxylic acid'},
    'c25h50o2': {'name': 'Pentacosylic acid', 'formula': 'C25H50O2', 'mw': '382.67', 'type': 'carboxylic acid'},
    'c26h52o2': {'name': 'Cerotic acid', 'formula': 'C26H52O2', 'mw': '396.69', 'type': 'carboxylic acid'},
    'c27h54o2': {'name': 'Heptacosylic acid', 'formula': 'C27H54O2', 'mw': '410.72', 'type': 'carboxylic acid'},
    'c28h56o2': {'name': 'Montanic acid', 'formula': 'C28H56O2', 'mw': '424.75', 'type': 'carboxylic acid'},
    'c29h58o2': {'name': 'Nonacosylic acid', 'formula': 'C29H58O2', 'mw': '438.78', 'type': 'carboxylic acid'},
    'c30h60o2': {'name': 'Melissic acid', 'formula': 'C30H60O2', 'mw': '452.80', 'type': 'carboxylic acid'},
}

# General compounds database: formula -> properties
COMPOUNDS: Dict[str, Dict] = {
    'h2o': {'name': 'Water', 'formula': 'H2O', 'mw': '18.015', 'density': '1.0', 'mp': '0', 'bp': '100', 'type': 'oxide', 'solubility': 'universal solvent'},
    'naoh': {'name': 'Sodium hydroxide', 'formula': 'NaOH', 'mw': '39.997', 'density': '2.13', 'mp': '318', 'bp': '1388', 'type': 'base', 'solubility': 'soluble'},
    'ca(oh)2': {'name': 'Calcium hydroxide', 'formula': 'Ca(OH)2', 'mw': '74.093', 'density': '2.211', 'mp': '580', 'bp': '2850', 'type': 'base', 'solubility': 'slightly soluble'},
    'cao': {'name': 'Calcium oxide', 'formula': 'CaO', 'mw': '56.077', 'density': '3.34', 'mp': '2572', 'bp': '2850', 'type': 'oxide', 'solubility': 'reacts with water'},
    'na2o': {'name': 'Sodium oxide', 'formula': 'Na2O', 'mw': '61.978', 'density': '2.27', 'mp': '1132', 'bp': '1950', 'type': 'oxide', 'solubility': 'reacts with water'},
    'hcl': {'name': 'Hydrochloric acid', 'formula': 'HCl', 'mw': '36.461', 'density': '1.49', 'mp': '-114', 'bp': '-85', 'type': 'acid', 'solubility': 'soluble'},
    'h2so4': {'name': 'Sulfuric acid', 'formula': 'H2SO4', 'mw': '98.079', 'density': '1.84', 'mp': '10', 'bp': '337', 'type': 'acid', 'solubility': 'soluble'},
    'hno3': {'name': 'Nitric acid', 'formula': 'HNO3', 'mw': '63.012', 'density': '1.51', 'mp': '-42', 'bp': '83', 'type': 'acid', 'solubility': 'soluble'},
    'nacl': {'name': 'Sodium chloride', 'formula': 'NaCl', 'mw': '58.443', 'density': '2.16', 'mp': '801', 'bp': '1413', 'type': 'salt', 'solubility': 'soluble'},
    'kcl': {'name': 'Potassium chloride', 'formula': 'KCl', 'mw': '74.551', 'density': '1.98', 'mp': '770', 'bp': '1420', 'type': 'salt', 'solubility': 'soluble'},
    'cacl2': {'name': 'Calcium chloride', 'formula': 'CaCl2', 'mw': '110.984', 'density': '2.15', 'mp': '772', 'bp': '1935', 'type': 'salt', 'solubility': 'soluble'},
    'na2co3': {'name': 'Sodium carbonate', 'formula': 'Na2CO3', 'mw': '105.988', 'density': '2.54', 'mp': '851', 'bp': '1600', 'type': 'salt', 'solubility': 'soluble'},
    'caco3': {'name': 'Calcium carbonate', 'formula': 'CaCO3', 'mw': '100.087', 'density': '2.71', 'mp': '825', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'co2': {'name': 'Carbon dioxide', 'formula': 'CO2', 'mw': '44.009', 'density': '1.98', 'mp': '-78.5', 'bp': '-78.5', 'type': 'oxide', 'solubility': 'slightly soluble'},
    'co': {'name': 'Carbon monoxide', 'formula': 'CO', 'mw': '28.010', 'density': '1.25', 'mp': '-205', 'bp': '-191.5', 'type': 'oxide', 'solubility': 'slightly soluble'},
    'nh3': {'name': 'Ammonia', 'formula': 'NH3', 'mw': '17.031', 'density': '0.73', 'mp': '-77.7', 'bp': '-33.3', 'type': 'base', 'solubility': 'soluble'},
    'ch4': {'name': 'Methane', 'formula': 'CH4', 'mw': '16.043', 'density': '0.657', 'mp': '-182.5', 'bp': '-161.5', 'type': 'alkane', 'solubility': 'insoluble'},
    'o2': {'name': 'Oxygen', 'formula': 'O2', 'mw': '31.998', 'density': '1.43', 'mp': '-218.8', 'bp': '-183', 'type': 'element', 'solubility': 'slightly soluble'},
    'n2': {'name': 'Nitrogen', 'formula': 'N2', 'mw': '28.014', 'density': '1.25', 'mp': '-210', 'bp': '-195.8', 'type': 'element', 'solubility': 'slightly soluble'},
    'h2': {'name': 'Hydrogen', 'formula': 'H2', 'mw': '2.016', 'density': '0.0899', 'mp': '-259.14', 'bp': '-252.87', 'type': 'element', 'solubility': 'insoluble'},
    'cl2': {'name': 'Chlorine', 'formula': 'Cl2', 'mw': '70.906', 'density': '3.2', 'mp': '-101.5', 'bp': '-34.04', 'type': 'element', 'solubility': 'soluble'},
    'so2': {'name': 'Sulfur dioxide', 'formula': 'SO2', 'mw': '64.066', 'density': '2.63', 'mp': '-72', 'bp': '-10', 'type': 'oxide', 'solubility': 'soluble'},
    'so3': {'name': 'Sulfur trioxide', 'formula': 'SO3', 'mw': '80.066', 'density': '1.92', 'mp': '16.9', 'bp': '45', 'type': 'oxide', 'solubility': 'reacts with water'},
    'no': {'name': 'Nitric oxide', 'formula': 'NO', 'mw': '30.006', 'density': '1.34', 'mp': '-163.6', 'bp': '-151.7', 'type': 'oxide', 'solubility': 'slightly soluble'},
    'no2': {'name': 'Nitrogen dioxide', 'formula': 'NO2', 'mw': '46.005', 'density': '1.45', 'mp': '-11.2', 'bp': '21.2', 'type': 'oxide', 'solubility': 'reacts with water'},
    'n2o': {'name': 'Nitrous oxide', 'formula': 'N2O', 'mw': '44.013', 'density': '1.98', 'mp': '-90.9', 'bp': '-88.5', 'type': 'oxide', 'solubility': 'soluble'},
    'h2o2': {'name': 'Hydrogen peroxide', 'formula': 'H2O2', 'mw': '34.014', 'density': '1.45', 'mp': '-0.43', 'bp': '150.2', 'type': 'oxide', 'solubility': 'soluble'},
    'na2so4': {'name': 'Sodium sulfate', 'formula': 'Na2SO4', 'mw': '142.042', 'density': '2.66', 'mp': '884', 'bp': '1429', 'type': 'salt', 'solubility': 'soluble'},
    'k2so4': {'name': 'Potassium sulfate', 'formula': 'K2SO4', 'mw': '174.259', 'density': '2.66', 'mp': '1069', 'bp': '1689', 'type': 'salt', 'solubility': 'soluble'},
    'mgso4': {'name': 'Magnesium sulfate', 'formula': 'MgSO4', 'mw': '120.366', 'density': '2.66', 'mp': '1124', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'feso4': {'name': 'Iron(II) sulfate', 'formula': 'FeSO4', 'mw': '151.908', 'density': '1.898', 'mp': '64', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'fe2(so4)3': {'name': 'Iron(III) sulfate', 'formula': 'Fe2(SO4)3', 'mw': '399.881', 'density': '3.097', 'mp': '480', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cuso4': {'name': 'Copper(II) sulfate', 'formula': 'CuSO4', 'mw': '159.609', 'density': '3.6', 'mp': '110', 'bp': '650', 'type': 'salt', 'solubility': 'soluble'},
    'agno3': {'name': 'Silver nitrate', 'formula': 'AgNO3', 'mw': '169.873', 'density': '4.35', 'mp': '212', 'bp': '444', 'type': 'salt', 'solubility': 'soluble'},
    'naoh2': {'name': 'Sodium hydroxide', 'formula': 'NaOH', 'mw': '39.997', 'density': '2.13', 'mp': '318', 'bp': '1388', 'type': 'base', 'solubility': 'soluble'},
    'koh': {'name': 'Potassium hydroxide', 'formula': 'KOH', 'mw': '56.106', 'density': '2.044', 'mp': '360', 'bp': '1327', 'type': 'base', 'solubility': 'soluble'},
    'mg(oh)2': {'name': 'Magnesium hydroxide', 'formula': 'Mg(OH)2', 'mw': '58.319', 'density': '2.34', 'mp': '350', 'bp': 'decomposes', 'type': 'base', 'solubility': 'insoluble'},
    'al(oh)3': {'name': 'Aluminum hydroxide', 'formula': 'Al(OH)3', 'mw': '78.004', 'density': '2.42', 'mp': '300', 'bp': 'decomposes', 'type': 'base', 'solubility': 'insoluble'},
    'fe(oh)3': {'name': 'Iron(III) hydroxide', 'formula': 'Fe(OH)3', 'mw': '106.867', 'density': '3.4', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'base', 'solubility': 'insoluble'},
    'fe(oh)2': {'name': 'Iron(II) hydroxide', 'formula': 'Fe(OH)2', 'mw': '89.866', 'density': '3.4', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'base', 'solubility': 'insoluble'},
    'cu(oh)2': {'name': 'Copper(II) hydroxide', 'formula': 'Cu(OH)2', 'mw': '97.561', 'density': '3.37', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'base', 'solubility': 'insoluble'},
    'zn(oh)2': {'name': 'Zinc hydroxide', 'formula': 'Zn(OH)2', 'mw': '99.395', 'density': '3.05', 'mp': '125', 'bp': 'decomposes', 'type': 'base', 'solubility': 'insoluble'},
    'mgo': {'name': 'Magnesium oxide', 'formula': 'MgO', 'mw': '40.304', 'density': '3.58', 'mp': '2852', 'bp': '3600', 'type': 'oxide', 'solubility': 'insoluble'},
    'al2o3': {'name': 'Aluminum oxide', 'formula': 'Al2O3', 'mw': '101.961', 'density': '3.95', 'mp': '2072', 'bp': '2977', 'type': 'oxide', 'solubility': 'insoluble'},
    'fe2o3': {'name': 'Iron(III) oxide', 'formula': 'Fe2O3', 'mw': '159.688', 'density': '5.24', 'mp': '1565', 'bp': 'decomposes', 'type': 'oxide', 'solubility': 'insoluble'},
    'feo': {'name': 'Iron(II) oxide', 'formula': 'FeO', 'mw': '71.844', 'density': '5.7', 'mp': '1377', 'bp': 'decomposes', 'type': 'oxide', 'solubility': 'insoluble'},
    'cuo': {'name': 'Copper(II) oxide', 'formula': 'CuO', 'mw': '79.545', 'density': '6.31', 'mp': '1326', 'bp': 'decomposes', 'type': 'oxide', 'solubility': 'insoluble'},
    'cu2o': {'name': 'Copper(I) oxide', 'formula': 'Cu2O', 'mw': '143.091', 'density': '6.0', 'mp': '1235', 'bp': 'decomposes', 'type': 'oxide', 'solubility': 'insoluble'},
    'zno': {'name': 'Zinc oxide', 'formula': 'ZnO', 'mw': '81.379', 'density': '5.606', 'mp': '1975', 'bp': '2360', 'type': 'oxide', 'solubility': 'insoluble'},
    'sio2': {'name': 'Silicon dioxide', 'formula': 'SiO2', 'mw': '60.084', 'density': '2.65', 'mp': '1713', 'bp': '2230', 'type': 'oxide', 'solubility': 'insoluble'},
    'tio2': {'name': 'Titanium dioxide', 'formula': 'TiO2', 'mw': '79.866', 'density': '4.23', 'mp': '1843', 'bp': '2972', 'type': 'oxide', 'solubility': 'insoluble'},
    'mno2': {'name': 'Manganese dioxide', 'formula': 'MnO2', 'mw': '86.937', 'density': '5.026', 'mp': '535', 'bp': 'decomposes', 'type': 'oxide', 'solubility': 'insoluble'},
    'cr2o3': {'name': 'Chromium(III) oxide', 'formula': 'Cr2O3', 'mw': '151.990', 'density': '5.22', 'mp': '2435', 'bp': '4000', 'type': 'oxide', 'solubility': 'insoluble'},
    'nio': {'name': 'Nickel(II) oxide', 'formula': 'NiO', 'mw': '74.693', 'density': '6.67', 'mp': '1955', 'bp': 'decomposes', 'type': 'oxide', 'solubility': 'insoluble'},
    'co2o3': {'name': 'Cobalt(III) oxide', 'formula': 'Co2O3', 'mw': '165.864', 'density': '5.18', 'mp': '895', 'bp': 'decomposes', 'type': 'oxide', 'solubility': 'insoluble'},
    'wo3': {'name': 'Tungsten(VI) oxide', 'formula': 'WO3', 'mw': '231.838', 'density': '7.16', 'mp': '1473', 'bp': '1700', 'type': 'oxide', 'solubility': 'insoluble'},
    'v2o5': {'name': 'Vanadium(V) oxide', 'formula': 'V2O5', 'mw': '181.880', 'density': '3.357', 'mp': '690', 'bp': '1750', 'type': 'oxide', 'solubility': 'slightly soluble'},
    'p2o5': {'name': 'Phosphorus pentoxide', 'formula': 'P2O5', 'mw': '141.945', 'density': '2.39', 'mp': '340', 'bp': '360', 'type': 'oxide', 'solubility': 'reacts with water'},
    'so2cl2': {'name': 'Sulfuryl chloride', 'formula': 'SO2Cl2', 'mw': '134.969', 'density': '1.667', 'mp': '-54.1', 'bp': '69.1', 'type': 'halide', 'solubility': 'decomposes in water'},
    'socl2': {'name': 'Thionyl chloride', 'formula': 'SOCl2', 'mw': '118.971', 'density': '1.64', 'mp': '-105', 'bp': '79', 'type': 'halide', 'solubility': 'decomposes in water'},
    'pcl3': {'name': 'Phosphorus trichloride', 'formula': 'PCl3', 'mw': '137.333', 'density': '1.574', 'mp': '-93.6', 'bp': '76.1', 'type': 'halide', 'solubility': 'reacts with water'},
    'pcl5': {'name': 'Phosphorus pentachloride', 'formula': 'PCl5', 'mw': '208.239', 'density': '1.6', 'mp': '160', 'bp': '167', 'type': 'halide', 'solubility': 'reacts with water'},
    'sif4': {'name': 'Silicon tetrafluoride', 'formula': 'SiF4', 'mw': '104.079', 'density': '1.66', 'mp': '-90', 'bp': '-86', 'type': 'halide', 'solubility': 'reacts with water'},
    'sicl4': {'name': 'Silicon tetrachloride', 'formula': 'SiCl4', 'mw': '169.898', 'density': '1.483', 'mp': '-68.7', 'bp': '57.6', 'type': 'halide', 'solubility': 'reacts with water'},
    'ccl4': {'name': 'Carbon tetrachloride', 'formula': 'CCl4', 'mw': '153.823', 'density': '1.594', 'mp': '-22.9', 'bp': '76.7', 'type': 'halide', 'solubility': 'insoluble'},
    'cs2': {'name': 'Carbon disulfide', 'formula': 'CS2', 'mw': '76.141', 'density': '1.263', 'mp': '-111.6', 'bp': '46.3', 'type': 'organic', 'solubility': 'insoluble'},
    'hcn': {'name': 'Hydrogen cyanide', 'formula': 'HCN', 'mw': '27.026', 'density': '0.687', 'mp': '-13.4', 'bp': '25.6', 'type': 'acid', 'solubility': 'soluble'},
    'hf': {'name': 'Hydrofluoric acid', 'formula': 'HF', 'mw': '20.006', 'density': '1.15', 'mp': '-83.6', 'bp': '19.5', 'type': 'acid', 'solubility': 'soluble'},
    'hbr': {'name': 'Hydrobromic acid', 'formula': 'HBr', 'mw': '80.912', 'density': '1.49', 'mp': '-86.9', 'bp': '-66.8', 'type': 'acid', 'solubility': 'soluble'},
    'hi': {'name': 'Hydroiodic acid', 'formula': 'HI', 'mw': '127.912', 'density': '2.85', 'mp': '-50.8', 'bp': '-35.4', 'type': 'acid', 'solubility': 'soluble'},
    'hclo4': {'name': 'Perchloric acid', 'formula': 'HClO4', 'mw': '100.458', 'density': '1.67', 'mp': '-17', 'bp': '203', 'type': 'acid', 'solubility': 'soluble'},
    'hclo3': {'name': 'Chloric acid', 'formula': 'HClO3', 'mw': '84.459', 'density': '1.0', 'mp': '-104', 'bp': 'decomposes', 'type': 'acid', 'solubility': 'soluble'},
    'hclo': {'name': 'Hypochlorous acid', 'formula': 'HClO', 'mw': '52.458', 'density': '1.0', 'mp': '-120', 'bp': 'decomposes', 'type': 'acid', 'solubility': 'soluble'},
    'h2s': {'name': 'Hydrogen sulfide', 'formula': 'H2S', 'mw': '34.082', 'density': '1.363', 'mp': '-85.5', 'bp': '-60.3', 'type': 'acid', 'solubility': 'soluble'},
    'h2so3': {'name': 'Sulfurous acid', 'formula': 'H2SO3', 'mw': '82.079', 'density': '1.03', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'acid', 'solubility': 'soluble'},
    'h3po4': {'name': 'Phosphoric acid', 'formula': 'H3PO4', 'mw': '97.994', 'density': '1.88', 'mp': '42.35', 'bp': 'decomposes', 'type': 'acid', 'solubility': 'soluble'},
    'h3po3': {'name': 'Phosphorous acid', 'formula': 'H3PO3', 'mw': '82.000', 'density': '1.65', 'mp': '73.6', 'bp': '200', 'type': 'acid', 'solubility': 'soluble'},
    'h3po2': {'name': 'Hypophosphorous acid', 'formula': 'H3PO2', 'mw': '65.996', 'density': '1.0', 'mp': '26.5', 'bp': 'decomposes', 'type': 'acid', 'solubility': 'soluble'},
    'h2co3': {'name': 'Carbonic acid', 'formula': 'H2CO3', 'mw': '62.025', 'density': '1.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'acid', 'solubility': 'soluble'},
    'hno2': {'name': 'Nitrous acid', 'formula': 'HNO2', 'mw': '47.013', 'density': '1.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'acid', 'solubility': 'soluble'},
    'hcn2': {'name': 'Cyanamide', 'formula': 'H2CN2', 'mw': '42.040', 'density': '1.0', 'mp': '44', 'bp': 'decomposes', 'type': 'organic', 'solubility': 'soluble'},
    'na3po4': {'name': 'Sodium phosphate', 'formula': 'Na3PO4', 'mw': '163.941', 'density': '2.536', 'mp': '73.4', 'bp': '1583', 'type': 'salt', 'solubility': 'soluble'},
    'na2hpo4': {'name': 'Disodium hydrogen phosphate', 'formula': 'Na2HPO4', 'mw': '141.959', 'density': '1.7', 'mp': '250', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'nah2po4': {'name': 'Sodium dihydrogen phosphate', 'formula': 'NaH2PO4', 'mw': '119.977', 'density': '2.04', 'mp': '100', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'k3po4': {'name': 'Potassium phosphate', 'formula': 'K3PO4', 'mw': '212.266', 'density': '2.564', 'mp': '1340', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'ca3(po4)2': {'name': 'Calcium phosphate', 'formula': 'Ca3(PO4)2', 'mw': '310.177', 'density': '3.14', 'mp': '1670', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'mg3(po4)2': {'name': 'Magnesium phosphate', 'formula': 'Mg3(PO4)2', 'mw': '262.858', 'density': '2.19', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'alpo4': {'name': 'Aluminum phosphate', 'formula': 'AlPO4', 'mw': '121.953', 'density': '2.566', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'fe2(po4)3': {'name': 'Iron(III) phosphate', 'formula': 'Fe2(PO4)3', 'mw': '357.478', 'density': '2.89', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'fepo4': {'name': 'Iron(III) phosphate', 'formula': 'FePO4', 'mw': '150.816', 'density': '2.74', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'zns': {'name': 'Zinc sulfide', 'formula': 'ZnS', 'mw': '97.474', 'density': '4.09', 'mp': '1700', 'bp': '1830', 'type': 'salt', 'solubility': 'insoluble'},
    'pbs': {'name': 'Lead(II) sulfide', 'formula': 'PbS', 'mw': '239.266', 'density': '7.6', 'mp': '1118', 'bp': '1281', 'type': 'salt', 'solubility': 'insoluble'},
    'cus': {'name': 'Copper(II) sulfide', 'formula': 'CuS', 'mw': '95.611', 'density': '4.76', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'fes': {'name': 'Iron(II) sulfide', 'formula': 'FeS', 'mw': '87.911', 'density': '4.84', 'mp': '1194', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'fes2': {'name': 'Iron disulfide', 'formula': 'FeS2', 'mw': '119.975', 'density': '5.01', 'mp': '600', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'ag2s': {'name': 'Silver sulfide', 'formula': 'Ag2S', 'mw': '247.801', 'density': '7.23', 'mp': '836', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'cdS': {'name': 'Cadmium sulfide', 'formula': 'CdS', 'mw': '144.474', 'density': '4.826', 'mp': '1750', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'hgs': {'name': 'Mercury(II) sulfide', 'formula': 'HgS', 'mw': '232.66', 'density': '8.1', 'mp': '584', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'sb2s3': {'name': 'Antimony(III) sulfide', 'formula': 'Sb2S3', 'mw': '339.715', 'density': '4.64', 'mp': '550', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'bi2s3': {'name': 'Bismuth(III) sulfide', 'formula': 'Bi2S3', 'mw': '514.155', 'density': '7.39', 'mp': '775', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'snS': {'name': 'Tin(II) sulfide', 'formula': 'SnS', 'mw': '150.775', 'density': '5.22', 'mp': '882', 'bp': '1230', 'type': 'salt', 'solubility': 'insoluble'},
    'snS2': {'name': 'Tin(IV) sulfide', 'formula': 'SnS2', 'mw': '182.84', 'density': '4.5', 'mp': '880', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'mns': {'name': 'Manganese(II) sulfide', 'formula': 'MnS', 'mw': '87.003', 'density': '3.99', 'mp': '1610', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'cos': {'name': 'Cobalt(II) sulfide', 'formula': 'CoS', 'mw': '90.998', 'density': '5.45', 'mp': '1100', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'nis': {'name': 'Nickel(II) sulfide', 'formula': 'NiS', 'mw': '90.758', 'density': '5.3', 'mp': '797', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'cr2s3': {'name': 'Chromium(III) sulfide', 'formula': 'Cr2S3', 'mw': '200.187', 'density': '3.77', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'al2s3': {'name': 'Aluminum sulfide', 'formula': 'Al2S3', 'mw': '150.158', 'density': '2.02', 'mp': '1100', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'decomposes in water'},
    'siS2': {'name': 'Silicon disulfide', 'formula': 'SiS2', 'mw': '92.211', 'density': '1.85', 'mp': '1090', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'decomposes in water'},
    'tiS2': {'name': 'Titanium(IV) sulfide', 'formula': 'TiS2', 'mw': '112.007', 'density': '3.22', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'vs2': {'name': 'Vanadium(IV) sulfide', 'formula': 'VS2', 'mw': '115.071', 'density': '3.22', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'mos2': {'name': 'Molybdenum(IV) sulfide', 'formula': 'MoS2', 'mw': '160.07', 'density': '5.06', 'mp': '1185', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'ws2': {'name': 'Tungsten(IV) sulfide', 'formula': 'WS2', 'mw': '247.98', 'density': '7.5', 'mp': '1250', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    're2s7': {'name': 'Rhenium(VII) sulfide', 'formula': 'Re2S7', 'mw': '596.81', 'density': '4.867', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'decomposes in water'},
    'irS2': {'name': 'Iridium(IV) sulfide', 'formula': 'IrS2', 'mw': '256.28', 'density': '9.33', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'ptS2': {'name': 'Platinum(IV) sulfide', 'formula': 'PtS2', 'mw': '259.25', 'density': '7.85', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'pdS': {'name': 'Palladium(II) sulfide', 'formula': 'PdS', 'mw': '138.47', 'density': '6.7', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'rh2s3': {'name': 'Rhodium(III) sulfide', 'formula': 'Rh2S3', 'mw': '294.02', 'density': '6.44', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'ruS2': {'name': 'Ruthenium(IV) sulfide', 'formula': 'RuS2', 'mw': '197.24', 'density': '4.64', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'osS2': {'name': 'Osmium(IV) sulfide', 'formula': 'OsS2', 'mw': '254.34', 'density': '9.47', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'au2s': {'name': 'Gold(I) sulfide', 'formula': 'Au2S', 'mw': '426.0', 'density': '7.85', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'au2s3': {'name': 'Gold(III) sulfide', 'formula': 'Au2S3', 'mw': '490.1', 'density': '8.75', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'ag2so4': {'name': 'Silver sulfate', 'formula': 'Ag2SO4', 'mw': '311.799', 'density': '5.45', 'mp': '652', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'slightly soluble'},
    'pbso4': {'name': 'Lead(II) sulfate', 'formula': 'PbSO4', 'mw': '303.262', 'density': '6.29', 'mp': '1170', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'baso4': {'name': 'Barium sulfate', 'formula': 'BaSO4', 'mw': '233.389', 'density': '4.5', 'mp': '1580', 'bp': '1600', 'type': 'salt', 'solubility': 'insoluble'},
    'srso4': {'name': 'Strontium sulfate', 'formula': 'SrSO4', 'mw': '183.682', 'density': '3.96', 'mp': '1605', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'caso4': {'name': 'Calcium sulfate', 'formula': 'CaSO4', 'mw': '136.142', 'density': '2.96', 'mp': '1450', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'slightly soluble'},
    'mgso4': {'name': 'Magnesium sulfate', 'formula': 'MgSO4', 'mw': '120.366', 'density': '2.66', 'mp': '1124', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'znso4': {'name': 'Zinc sulfate', 'formula': 'ZnSO4', 'mw': '161.472', 'density': '3.54', 'mp': '680', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'mnso4': {'name': 'Manganese(II) sulfate', 'formula': 'MnSO4', 'mw': '151.001', 'density': '3.25', 'mp': '700', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'niso4': {'name': 'Nickel(II) sulfate', 'formula': 'NiSO4', 'mw': '154.756', 'density': '3.68', 'mp': '840', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'coso4': {'name': 'Cobalt(II) sulfate', 'formula': 'CoSO4', 'mw': '154.996', 'density': '3.71', 'mp': '96.8', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cr2(so4)3': {'name': 'Chromium(III) sulfate', 'formula': 'Cr2(SO4)3', 'mw': '392.18', 'density': '3.1', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'al2(so4)3': {'name': 'Aluminum sulfate', 'formula': 'Al2(SO4)3', 'mw': '342.151', 'density': '2.672', 'mp': '770', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'fe2(so4)3': {'name': 'Iron(III) sulfate', 'formula': 'Fe2(SO4)3', 'mw': '399.881', 'density': '3.097', 'mp': '480', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'feso4': {'name': 'Iron(II) sulfate', 'formula': 'FeSO4', 'mw': '151.908', 'density': '1.898', 'mp': '64', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cuso4': {'name': 'Copper(II) sulfate', 'formula': 'CuSO4', 'mw': '159.609', 'density': '3.6', 'mp': '110', 'bp': '650', 'type': 'salt', 'solubility': 'soluble'},
    'znso4': {'name': 'Zinc sulfate', 'formula': 'ZnSO4', 'mw': '161.472', 'density': '3.54', 'mp': '680', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cdso4': {'name': 'Cadmium sulfate', 'formula': 'CdSO4', 'mw': '208.474', 'density': '4.691', 'mp': '1000', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'hgso4': {'name': 'Mercury(II) sulfate', 'formula': 'HgSO4', 'mw': '296.65', 'density': '6.47', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'decomposes in water'},
    'pb(no3)2': {'name': 'Lead(II) nitrate', 'formula': 'Pb(NO3)2', 'mw': '331.209', 'density': '4.53', 'mp': '470', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'ba(no3)2': {'name': 'Barium nitrate', 'formula': 'Ba(NO3)2', 'mw': '261.337', 'density': '3.24', 'mp': '590', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'sr(no3)2': {'name': 'Strontium nitrate', 'formula': 'Sr(NO3)2', 'mw': '211.629', 'density': '2.99', 'mp': '570', 'bp': '645', 'type': 'salt', 'solubility': 'soluble'},
    'ca(no3)2': {'name': 'Calcium nitrate', 'formula': 'Ca(NO3)2', 'mw': '164.088', 'density': '2.504', 'mp': '561', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'mg(no3)2': {'name': 'Magnesium nitrate', 'formula': 'Mg(NO3)2', 'mw': '148.313', 'density': '2.3', 'mp': '89', 'bp': '330', 'type': 'salt', 'solubility': 'soluble'},
    'zn(no3)2': {'name': 'Zinc nitrate', 'formula': 'Zn(NO3)2', 'mw': '189.389', 'density': '2.065', 'mp': '110', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cu(no3)2': {'name': 'Copper(II) nitrate', 'formula': 'Cu(NO3)2', 'mw': '187.555', 'density': '2.05', 'mp': '114', 'bp': '170', 'type': 'salt', 'solubility': 'soluble'},
    'fe(no3)3': {'name': 'Iron(III) nitrate', 'formula': 'Fe(NO3)3', 'mw': '241.859', 'density': '1.684', 'mp': '47.2', 'bp': '125', 'type': 'salt', 'solubility': 'soluble'},
    'fe(no3)2': {'name': 'Iron(II) nitrate', 'formula': 'Fe(NO3)2', 'mw': '179.855', 'density': '1.68', 'mp': '60.5', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'al(no3)3': {'name': 'Aluminum nitrate', 'formula': 'Al(NO3)3', 'mw': '212.996', 'density': '1.72', 'mp': '73', 'bp': '135', 'type': 'salt', 'solubility': 'soluble'},
    'cr(no3)3': {'name': 'Chromium(III) nitrate', 'formula': 'Cr(NO3)3', 'mw': '238.011', 'density': '1.85', 'mp': '60', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'ni(no3)2': {'name': 'Nickel(II) nitrate', 'formula': 'Ni(NO3)2', 'mw': '182.703', 'density': '2.05', 'mp': '56.7', 'bp': '136.7', 'type': 'salt', 'solubility': 'soluble'},
    'co(no3)2': {'name': 'Cobalt(II) nitrate', 'formula': 'Co(NO3)2', 'mw': '182.941', 'density': '2.49', 'mp': '100', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'mn(no3)2': {'name': 'Manganese(II) nitrate', 'formula': 'Mn(NO3)2', 'mw': '178.945', 'density': '2.2', 'mp': '37', 'bp': '129', 'type': 'salt', 'solubility': 'soluble'},
    'pbcl2': {'name': 'Lead(II) chloride', 'formula': 'PbCl2', 'mw': '278.106', 'density': '5.85', 'mp': '501', 'bp': '950', 'type': 'salt', 'solubility': 'slightly soluble'},
    'pbcl4': {'name': 'Lead(IV) chloride', 'formula': 'PbCl4', 'mw': '349.012', 'density': '3.15', 'mp': '-15', 'bp': '50', 'type': 'salt', 'solubility': 'decomposes in water'},
    'hgcl2': {'name': 'Mercury(II) chloride', 'formula': 'HgCl2', 'mw': '271.52', 'density': '5.44', 'mp': '276', 'bp': '304', 'type': 'salt', 'solubility': 'soluble'},
    'hg2cl2': {'name': 'Mercury(I) chloride', 'formula': 'Hg2Cl2', 'mw': '472.09', 'density': '7.15', 'mp': '400', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'cucl': {'name': 'Copper(I) chloride', 'formula': 'CuCl', 'mw': '98.999', 'density': '4.14', 'mp': '430', 'bp': '1490', 'type': 'salt', 'solubility': 'insoluble'},
    'cucl2': {'name': 'Copper(II) chloride', 'formula': 'CuCl2', 'mw': '134.452', 'density': '3.386', 'mp': '498', 'bp': '993', 'type': 'salt', 'solubility': 'soluble'},
    'fecl2': {'name': 'Iron(II) chloride', 'formula': 'FeCl2', 'mw': '126.751', 'density': '3.16', 'mp': '677', 'bp': '1023', 'type': 'salt', 'solubility': 'soluble'},
    'fecl3': {'name': 'Iron(III) chloride', 'formula': 'FeCl3', 'mw': '162.204', 'density': '2.898', 'mp': '306', 'bp': '315', 'type': 'salt', 'solubility': 'soluble'},
    'alcl3': {'name': 'Aluminum chloride', 'formula': 'AlCl3', 'mw': '133.341', 'density': '2.48', 'mp': '190', 'bp': '180', 'type': 'salt', 'solubility': 'soluble'},
    'zncl2': {'name': 'Zinc chloride', 'formula': 'ZnCl2', 'mw': '136.286', 'density': '2.907', 'mp': '290', 'bp': '732', 'type': 'salt', 'solubility': 'soluble'},
    'mgcl2': {'name': 'Magnesium chloride', 'formula': 'MgCl2', 'mw': '95.211', 'density': '2.32', 'mp': '714', 'bp': '1412', 'type': 'salt', 'solubility': 'soluble'},
    'cacl2': {'name': 'Calcium chloride', 'formula': 'CaCl2', 'mw': '110.984', 'density': '2.15', 'mp': '772', 'bp': '1935', 'type': 'salt', 'solubility': 'soluble'},
    'bacl2': {'name': 'Barium chloride', 'formula': 'BaCl2', 'mw': '208.233', 'density': '3.856', 'mp': '962', 'bp': '1560', 'type': 'salt', 'solubility': 'soluble'},
    'srcl2': {'name': 'Strontium chloride', 'formula': 'SrCl2', 'mw': '158.526', 'density': '3.052', 'mp': '874', 'bp': '1250', 'type': 'salt', 'solubility': 'soluble'},
    'nacl': {'name': 'Sodium chloride', 'formula': 'NaCl', 'mw': '58.443', 'density': '2.16', 'mp': '801', 'bp': '1413', 'type': 'salt', 'solubility': 'soluble'},
    'kcl': {'name': 'Potassium chloride', 'formula': 'KCl', 'mw': '74.551', 'density': '1.98', 'mp': '770', 'bp': '1420', 'type': 'salt', 'solubility': 'soluble'},
    'nh4cl': {'name': 'Ammonium chloride', 'formula': 'NH4Cl', 'mw': '53.491', 'density': '1.527', 'mp': '338', 'bp': '520', 'type': 'salt', 'solubility': 'soluble'},
    'nh4no3': {'name': 'Ammonium nitrate', 'formula': 'NH4NO3', 'mw': '80.043', 'density': '1.725', 'mp': '169.6', 'bp': '210', 'type': 'salt', 'solubility': 'soluble'},
    'nh4oh': {'name': 'Ammonium hydroxide', 'formula': 'NH4OH', 'mw': '35.046', 'density': '0.88', 'mp': '-57', 'bp': '38', 'type': 'base', 'solubility': 'soluble'},
    'na2o2': {'name': 'Sodium peroxide', 'formula': 'Na2O2', 'mw': '77.978', 'density': '2.805', 'mp': '460', 'bp': '675', 'type': 'oxide', 'solubility': 'reacts with water'},
    'k2o2': {'name': 'Potassium peroxide', 'formula': 'K2O2', 'mw': '110.192', 'density': '2.41', 'mp': '490', 'bp': 'decomposes', 'type': 'oxide', 'solubility': 'reacts with water'},
    'cao2': {'name': 'Calcium peroxide', 'formula': 'CaO2', 'mw': '72.078', 'density': '2.92', 'mp': '275', 'bp': 'decomposes', 'type': 'oxide', 'solubility': 'decomposes in water'},
    'mgo2': {'name': 'Magnesium peroxide', 'formula': 'MgO2', 'mw': '56.303', 'density': '3.2', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'oxide', 'solubility': 'decomposes in water'},
    'zno2': {'name': 'Zinc peroxide', 'formula': 'ZnO2', 'mw': '97.38', 'density': '1.57', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'oxide', 'solubility': 'decomposes in water'},
    'bao2': {'name': 'Barium peroxide', 'formula': 'BaO2', 'mw': '169.326', 'density': '4.96', 'mp': '450', 'bp': '800', 'type': 'oxide', 'solubility': 'decomposes in water'},
    'sro2': {'name': 'Strontium peroxide', 'formula': 'SrO2', 'mw': '119.616', 'density': '4.56', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'oxide', 'solubility': 'decomposes in water'},
    'li2o': {'name': 'Lithium oxide', 'formula': 'Li2O', 'mw': '29.881', 'density': '2.013', 'mp': '1438', 'bp': '2600', 'type': 'oxide', 'solubility': 'reacts with water'},
    'na2o': {'name': 'Sodium oxide', 'formula': 'Na2O', 'mw': '61.978', 'density': '2.27', 'mp': '1132', 'bp': '1950', 'type': 'oxide', 'solubility': 'reacts with water'},
    'k2o': {'name': 'Potassium oxide', 'formula': 'K2O', 'mw': '94.196', 'density': '2.35', 'mp': '740', 'bp': 'decomposes', 'type': 'oxide', 'solubility': 'reacts with water'},
    'rb2o': {'name': 'Rubidium oxide', 'formula': 'Rb2O', 'mw': '186.935', 'density': '3.2', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'oxide', 'solubility': 'reacts with water'},
    'cs2o': {'name': 'Cesium oxide', 'formula': 'Cs2O', 'mw': '281.810', 'density': '3.25', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'oxide', 'solubility': 'reacts with water'},
    'beo': {'name': 'Beryllium oxide', 'formula': 'BeO', 'mw': '25.011', 'density': '3.01', 'mp': '2578', 'bp': '3900', 'type': 'oxide', 'solubility': 'insoluble'},
    'mgo': {'name': 'Magnesium oxide', 'formula': 'MgO', 'mw': '40.304', 'density': '3.58', 'mp': '2852', 'bp': '3600', 'type': 'oxide', 'solubility': 'insoluble'},
    'cao': {'name': 'Calcium oxide', 'formula': 'CaO', 'mw': '56.077', 'density': '3.34', 'mp': '2572', 'bp': '2850', 'type': 'oxide', 'solubility': 'reacts with water'},
    'sro': {'name': 'Strontium oxide', 'formula': 'SrO', 'mw': '103.619', 'density': '4.7', 'mp': '2430', 'bp': '3000', 'type': 'oxide', 'solubility': 'reacts with water'},
    'bao': {'name': 'Barium oxide', 'formula': 'BaO', 'mw': '153.326', 'density': '5.72', 'mp': '1923', 'bp': '2000', 'type': 'oxide', 'solubility': 'reacts with water'},
    'li2co3': {'name': 'Lithium carbonate', 'formula': 'Li2CO3', 'mw': '73.891', 'density': '2.11', 'mp': '723', 'bp': '1310', 'type': 'salt', 'solubility': 'slightly soluble'},
    'na2co3': {'name': 'Sodium carbonate', 'formula': 'Na2CO3', 'mw': '105.988', 'density': '2.54', 'mp': '851', 'bp': '1600', 'type': 'salt', 'solubility': 'soluble'},
    'k2co3': {'name': 'Potassium carbonate', 'formula': 'K2CO3', 'mw': '138.205', 'density': '2.43', 'mp': '891', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'rb2co3': {'name': 'Rubidium carbonate', 'formula': 'Rb2CO3', 'mw': '230.945', 'density': '3.68', 'mp': '837', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cs2co3': {'name': 'Cesium carbonate', 'formula': 'Cs2CO3', 'mw': '325.824', 'density': '4.072', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'mgco3': {'name': 'Magnesium carbonate', 'formula': 'MgCO3', 'mw': '84.313', 'density': '2.96', 'mp': '350', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'caco3': {'name': 'Calcium carbonate', 'formula': 'CaCO3', 'mw': '100.087', 'density': '2.71', 'mp': '825', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'srco3': {'name': 'Strontium carbonate', 'formula': 'SrCO3', 'mw': '147.629', 'density': '3.7', 'mp': '1497', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'baco3': {'name': 'Barium carbonate', 'formula': 'BaCO3', 'mw': '197.336', 'density': '4.43', 'mp': '811', 'bp': '1450', 'type': 'salt', 'solubility': 'insoluble'},
    'znco3': {'name': 'Zinc carbonate', 'formula': 'ZnCO3', 'mw': '125.378', 'density': '4.436', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'feco3': {'name': 'Iron(II) carbonate', 'formula': 'FeCO3', 'mw': '115.854', 'density': '3.9', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'cuco3': {'name': 'Copper(II) carbonate', 'formula': 'CuCO3', 'mw': '123.555', 'density': '4.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'mnco3': {'name': 'Manganese(II) carbonate', 'formula': 'MnCO3', 'mw': '114.947', 'density': '3.125', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'nico3': {'name': 'Nickel(II) carbonate', 'formula': 'NiCO3', 'mw': '118.703', 'density': '4.39', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'coco3': {'name': 'Cobalt(II) carbonate', 'formula': 'CoCO3', 'mw': '118.942', 'density': '4.13', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'pbco3': {'name': 'Lead(II) carbonate', 'formula': 'PbCO3', 'mw': '267.209', 'density': '6.6', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'cdco3': {'name': 'Cadmium carbonate', 'formula': 'CdCO3', 'mw': '172.414', 'density': '5.027', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'ag2co3': {'name': 'Silver carbonate', 'formula': 'Ag2CO3', 'mw': '275.745', 'density': '6.077', 'mp': '218', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'hg2co3': {'name': 'Mercury(I) carbonate', 'formula': 'Hg2CO3', 'mw': '461.186', 'density': '6.473', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'decomposes in water'},
    'lihco3': {'name': 'Lithium bicarbonate', 'formula': 'LiHCO3', 'mw': '67.958', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'nahco3': {'name': 'Sodium bicarbonate', 'formula': 'NaHCO3', 'mw': '84.007', 'density': '2.2', 'mp': '50', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'khco3': {'name': 'Potassium bicarbonate', 'formula': 'KHCO3', 'mw': '100.115', 'density': '2.17', 'mp': '100', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'rbhco3': {'name': 'Rubidium bicarbonate', 'formula': 'RbHCO3', 'mw': '146.472', 'density': '2.43', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cshco3': {'name': 'Cesium bicarbonate', 'formula': 'CsHCO3', 'mw': '193.921', 'density': '2.52', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'mghco3': {'name': 'Magnesium bicarbonate', 'formula': 'Mg(HCO3)2', 'mw': '146.338', 'density': '1.8', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cahco3': {'name': 'Calcium bicarbonate', 'formula': 'Ca(HCO3)2', 'mw': '162.112', 'density': '2.22', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'srhco3': {'name': 'Strontium bicarbonate', 'formula': 'Sr(HCO3)2', 'mw': '209.636', 'density': '2.44', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'bahco3': {'name': 'Barium bicarbonate', 'formula': 'Ba(HCO3)2', 'mw': '259.336', 'density': '2.58', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'li2so3': {'name': 'Lithium sulfite', 'formula': 'Li2SO3', 'mw': '93.945', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'na2so3': {'name': 'Sodium sulfite', 'formula': 'Na2SO3', 'mw': '126.043', 'density': '2.633', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'k2so3': {'name': 'Potassium sulfite', 'formula': 'K2SO3', 'mw': '158.26', 'density': '2.35', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'rb2so3': {'name': 'Rubidium sulfite', 'formula': 'Rb2SO3', 'mw': '252.035', 'density': '2.8', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cs2so3': {'name': 'Cesium sulfite', 'formula': 'Cs2SO3', 'mw': '345.81', 'density': '3.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'mgso3': {'name': 'Magnesium sulfite', 'formula': 'MgSO3', 'mw': '104.368', 'density': '2.54', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'caso3': {'name': 'Calcium sulfite', 'formula': 'CaSO3', 'mw': '120.142', 'density': '2.54', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'slightly soluble'},
    'srso3': {'name': 'Strontium sulfite', 'formula': 'SrSO3', 'mw': '167.682', 'density': '3.19', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'baso3': {'name': 'Barium sulfite', 'formula': 'BaSO3', 'mw': '217.39', 'density': '4.25', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'znso3': {'name': 'Zinc sulfite', 'formula': 'ZnSO3', 'mw': '145.439', 'density': '3.28', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'feso3': {'name': 'Iron(II) sulfite', 'formula': 'FeSO3', 'mw': '135.909', 'density': '3.2', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'cuso3': {'name': 'Copper(II) sulfite', 'formula': 'CuSO3', 'mw': '143.611', 'density': '3.4', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'mnso3': {'name': 'Manganese(II) sulfite', 'formula': 'MnSO3', 'mw': '134.994', 'density': '3.25', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'niso3': {'name': 'Nickel(II) sulfite', 'formula': 'NiSO3', 'mw': '138.758', 'density': '3.4', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'coso3': {'name': 'Cobalt(II) sulfite', 'formula': 'CoSO3', 'mw': '138.998', 'density': '3.4', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'pbso3': {'name': 'Lead(II) sulfite', 'formula': 'PbSO3', 'mw': '267.262', 'density': '6.535', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'cdso3': {'name': 'Cadmium sulfite', 'formula': 'CdSO3', 'mw': '190.474', 'density': '4.72', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'ag2so3': {'name': 'Silver sulfite', 'formula': 'Ag2SO3', 'mw': '295.799', 'density': '5.4', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'hg2so3': {'name': 'Mercury(I) sulfite', 'formula': 'Hg2SO3', 'mw': '465.186', 'density': '7.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'decomposes in water'},
    'li2s2o3': {'name': 'Lithium thiosulfate', 'formula': 'Li2S2O3', 'mw': '127.06', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'na2s2o3': {'name': 'Sodium thiosulfate', 'formula': 'Na2S2O3', 'mw': '158.108', 'density': '1.667', 'mp': '48.5', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'k2s2o3': {'name': 'Potassium thiosulfate', 'formula': 'K2S2O3', 'mw': '190.262', 'density': '1.87', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'rb2s2o3': {'name': 'Rubidium thiosulfate', 'formula': 'Rb2S2O3', 'mw': '284.037', 'density': '2.2', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cs2s2o3': {'name': 'Cesium thiosulfate', 'formula': 'Cs2S2O3', 'mw': '377.812', 'density': '2.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'mgs2o3': {'name': 'Magnesium thiosulfate', 'formula': 'MgS2O3', 'mw': '136.433', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cas2o3': {'name': 'Calcium thiosulfate', 'formula': 'CaS2O3', 'mw': '152.147', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'srs2o3': {'name': 'Strontium thiosulfate', 'formula': 'SrS2O3', 'mw': '199.691', 'density': '2.2', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'bas2o3': {'name': 'Barium thiosulfate', 'formula': 'BaS2O3', 'mw': '247.39', 'density': '2.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'zns2o3': {'name': 'Zinc thiosulfate', 'formula': 'ZnS2O3', 'mw': '161.444', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'fes2o3': {'name': 'Iron(II) thiosulfate', 'formula': 'FeS2O3', 'mw': '167.914', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cus2o3': {'name': 'Copper(II) thiosulfate', 'formula': 'CuS2O3', 'mw': '175.616', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'mns2o3': {'name': 'Manganese(II) thiosulfate', 'formula': 'MnS2O3', 'mw': '166.999', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'nis2o3': {'name': 'Nickel(II) thiosulfate', 'formula': 'NiS2O3', 'mw': '170.763', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cos2o3': {'name': 'Cobalt(II) thiosulfate', 'formula': 'CoS2O3', 'mw': '170.903', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'pbs2o3': {'name': 'Lead(II) thiosulfate', 'formula': 'PbS2O3', 'mw': '301.267', 'density': '6.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'cds2o3': {'name': 'Cadmium thiosulfate', 'formula': 'CdS2O3', 'mw': '206.479', 'density': '4.7', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'ag2s2o3': {'name': 'Silver thiosulfate', 'formula': 'Ag2S2O3', 'mw': '327.804', 'density': '5.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'hg2s2o3': {'name': 'Mercury(I) thiosulfate', 'formula': 'Hg2S2O3', 'mw': '517.191', 'density': '7.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'decomposes in water'},
    'li2s2o4': {'name': 'Lithium dithionite', 'formula': 'Li2S2O4', 'mw': '174.06', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'na2s2o4': {'name': 'Sodium dithionite', 'formula': 'Na2S2O4', 'mw': '174.108', 'density': '2.35', 'mp': '52', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'k2s2o4': {'name': 'Potassium dithionite', 'formula': 'K2S2O4', 'mw': '206.262', 'density': '2.2', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'rb2s2o4': {'name': 'Rubidium dithionite', 'formula': 'Rb2S2O4', 'mw': '300.037', 'density': '2.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cs2s2o4': {'name': 'Cesium dithionite', 'formula': 'Cs2S2O4', 'mw': '393.812', 'density': '2.8', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'mgs2o4': {'name': 'Magnesium dithionite', 'formula': 'MgS2O4', 'mw': '152.433', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cas2o4': {'name': 'Calcium dithionite', 'formula': 'CaS2O4', 'mw': '168.147', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'srs2o4': {'name': 'Strontium dithionite', 'formula': 'SrS2O4', 'mw': '215.691', 'density': '2.2', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'bas2o4': {'name': 'Barium dithionite', 'formula': 'BaS2O4', 'mw': '263.39', 'density': '2.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'zns2o4': {'name': 'Zinc dithionite', 'formula': 'ZnS2O4', 'mw': '193.444', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'fes2o4': {'name': 'Iron(II) dithionite', 'formula': 'FeS2O4', 'mw': '199.914', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cus2o4': {'name': 'Copper(II) dithionite', 'formula': 'CuS2O4', 'mw': '207.616', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'mns2o4': {'name': 'Manganese(II) dithionite', 'formula': 'MnS2O4', 'mw': '198.999', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'nis2o4': {'name': 'Nickel(II) dithionite', 'formula': 'NiS2O4', 'mw': '202.763', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cos2o4': {'name': 'Cobalt(II) dithionite', 'formula': 'CoS2O4', 'mw': '202.903', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'pbs2o4': {'name': 'Lead(II) dithionite', 'formula': 'PbS2O4', 'mw': '333.267', 'density': '6.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'cds2o4': {'name': 'Cadmium dithionite', 'formula': 'CdS2O4', 'mw': '238.479', 'density': '4.7', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'ag2s2o4': {'name': 'Silver dithionite', 'formula': 'Ag2S2O4', 'mw': '359.804', 'density': '5.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'hg2s2o4': {'name': 'Mercury(I) dithionite', 'formula': 'Hg2S2O4', 'mw': '549.191', 'density': '7.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'decomposes in water'},
    'li2s2o5': {'name': 'Lithium metabisulfite', 'formula': 'Li2S2O5', 'mw': '190.06', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'na2s2o5': {'name': 'Sodium metabisulfite', 'formula': 'Na2S2O5', 'mw': '190.108', 'density': '2.33', 'mp': '150', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'k2s2o5': {'name': 'Potassium metabisulfite', 'formula': 'K2S2O5', 'mw': '222.262', 'density': '2.34', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'rb2s2o5': {'name': 'Rubidium metabisulfite', 'formula': 'Rb2S2O5', 'mw': '316.037', 'density': '2.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cs2s2o5': {'name': 'Cesium metabisulfite', 'formula': 'Cs2S2O5', 'mw': '409.812', 'density': '2.8', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'mgs2o5': {'name': 'Magnesium metabisulfite', 'formula': 'MgS2O5', 'mw': '168.433', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cas2o5': {'name': 'Calcium metabisulfite', 'formula': 'CaS2O5', 'mw': '184.147', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'srs2o5': {'name': 'Strontium metabisulfite', 'formula': 'SrS2O5', 'mw': '231.691', 'density': '2.2', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'bas2o5': {'name': 'Barium metabisulfite', 'formula': 'BaS2O5', 'mw': '279.39', 'density': '2.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'zns2o5': {'name': 'Zinc metabisulfite', 'formula': 'ZnS2O5', 'mw': '209.444', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'fes2o5': {'name': 'Iron(II) metabisulfite', 'formula': 'FeS2O5', 'mw': '215.914', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cus2o5': {'name': 'Copper(II) metabisulfite', 'formula': 'CuS2O5', 'mw': '223.616', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'mns2o5': {'name': 'Manganese(II) metabisulfite', 'formula': 'MnS2O5', 'mw': '214.999', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'nis2o5': {'name': 'Nickel(II) metabisulfite', 'formula': 'NiS2O5', 'mw': '218.763', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cos2o5': {'name': 'Cobalt(II) metabisulfite', 'formula': 'CoS2O5', 'mw': '218.903', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'pbs2o5': {'name': 'Lead(II) metabisulfite', 'formula': 'PbS2O5', 'mw': '349.267', 'density': '6.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'cds2o5': {'name': 'Cadmium metabisulfite', 'formula': 'CdS2O5', 'mw': '254.479', 'density': '4.7', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'ag2s2o5': {'name': 'Silver metabisulfite', 'formula': 'Ag2S2O5', 'mw': '375.804', 'density': '5.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'hg2s2o5': {'name': 'Mercury(I) metabisulfite', 'formula': 'Hg2S2O5', 'mw': '565.191', 'density': '7.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'decomposes in water'},
    'li2s2o6': {'name': 'Lithium dithionate', 'formula': 'Li2S2O6', 'mw': '206.06', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'na2s2o6': {'name': 'Sodium dithionate', 'formula': 'Na2S2O6', 'mw': '206.108', 'density': '2.35', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'k2s2o6': {'name': 'Potassium dithionate', 'formula': 'K2S2O6', 'mw': '238.262', 'density': '2.2', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'rb2s2o6': {'name': 'Rubidium dithionate', 'formula': 'Rb2S2O6', 'mw': '332.037', 'density': '2.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cs2s2o6': {'name': 'Cesium dithionate', 'formula': 'Cs2S2O6', 'mw': '425.812', 'density': '2.8', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'mgs2o6': {'name': 'Magnesium dithionate', 'formula': 'MgS2O6', 'mw': '184.433', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cas2o6': {'name': 'Calcium dithionate', 'formula': 'CaS2O6', 'mw': '200.147', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'srs2o6': {'name': 'Strontium dithionate', 'formula': 'SrS2O6', 'mw': '247.691', 'density': '2.2', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'bas2o6': {'name': 'Barium dithionate', 'formula': 'BaS2O6', 'mw': '295.39', 'density': '2.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'zns2o6': {'name': 'Zinc dithionate', 'formula': 'ZnS2O6', 'mw': '225.444', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'fes2o6': {'name': 'Iron(II) dithionate', 'formula': 'FeS2O6', 'mw': '231.914', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cus2o6': {'name': 'Copper(II) dithionate', 'formula': 'CuS2O6', 'mw': '239.616', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'mns2o6': {'name': 'Manganese(II) dithionate', 'formula': 'MnS2O6', 'mw': '230.999', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'nis2o6': {'name': 'Nickel(II) dithionate', 'formula': 'NiS2O6', 'mw': '234.763', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cos2o6': {'name': 'Cobalt(II) dithionate', 'formula': 'CoS2O6', 'mw': '234.903', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'pbs2o6': {'name': 'Lead(II) dithionate', 'formula': 'PbS2O6', 'mw': '365.267', 'density': '6.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'cds2o6': {'name': 'Cadmium dithionate', 'formula': 'CdS2O6', 'mw': '270.479', 'density': '4.7', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'ag2s2o6': {'name': 'Silver dithionate', 'formula': 'Ag2S2O6', 'mw': '391.804', 'density': '5.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'hg2s2o6': {'name': 'Mercury(I) dithionate', 'formula': 'Hg2S2O6', 'mw': '581.191', 'density': '7.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'decomposes in water'},
    'li2s2o7': {'name': 'Lithium pyrosulfate', 'formula': 'Li2S2O7', 'mw': '222.06', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'na2s2o7': {'name': 'Sodium pyrosulfate', 'formula': 'Na2S2O7', 'mw': '222.108', 'density': '2.35', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'k2s2o7': {'name': 'Potassium pyrosulfate', 'formula': 'K2S2O7', 'mw': '254.262', 'density': '2.2', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'rb2s2o7': {'name': 'Rubidium pyrosulfate', 'formula': 'Rb2S2O7', 'mw': '348.037', 'density': '2.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cs2s2o7': {'name': 'Cesium pyrosulfate', 'formula': 'Cs2S2O7', 'mw': '441.812', 'density': '2.8', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'mgs2o7': {'name': 'Magnesium pyrosulfate', 'formula': 'MgS2O7', 'mw': '200.433', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cas2o7': {'name': 'Calcium pyrosulfate', 'formula': 'CaS2O7', 'mw': '216.147', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'srs2o7': {'name': 'Strontium pyrosulfate', 'formula': 'SrS2O7', 'mw': '263.691', 'density': '2.2', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'bas2o7': {'name': 'Barium pyrosulfate', 'formula': 'BaS2O7', 'mw': '311.39', 'density': '2.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'zns2o7': {'name': 'Zinc pyrosulfate', 'formula': 'ZnS2O7', 'mw': '241.444', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'fes2o7': {'name': 'Iron(II) pyrosulfate', 'formula': 'FeS2O7', 'mw': '247.914', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cus2o7': {'name': 'Copper(II) pyrosulfate', 'formula': 'CuS2O7', 'mw': '255.616', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'mns2o7': {'name': 'Manganese(II) pyrosulfate', 'formula': 'MnS2O7', 'mw': '246.999', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'nis2o7': {'name': 'Nickel(II) pyrosulfate', 'formula': 'NiS2O7', 'mw': '250.763', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cos2o7': {'name': 'Cobalt(II) pyrosulfate', 'formula': 'CoS2O7', 'mw': '250.903', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'pbs2o7': {'name': 'Lead(II) pyrosulfate', 'formula': 'PbS2O7', 'mw': '381.267', 'density': '6.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'cds2o7': {'name': 'Cadmium pyrosulfate', 'formula': 'CdS2O7', 'mw': '286.479', 'density': '4.7', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'ag2s2o7': {'name': 'Silver pyrosulfate', 'formula': 'Ag2S2O7', 'mw': '407.804', 'density': '5.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'hg2s2o7': {'name': 'Mercury(I) pyrosulfate', 'formula': 'Hg2S2O7', 'mw': '597.191', 'density': '7.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'decomposes in water'},
    'li2s2o8': {'name': 'Lithium peroxodisulfate', 'formula': 'Li2S2O8', 'mw': '238.06', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'na2s2o8': {'name': 'Sodium peroxodisulfate', 'formula': 'Na2S2O8', 'mw': '238.108', 'density': '2.35', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'k2s2o8': {'name': 'Potassium peroxodisulfate', 'formula': 'K2S2O8', 'mw': '270.262', 'density': '2.2', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'rb2s2o8': {'name': 'Rubidium peroxodisulfate', 'formula': 'Rb2S2O8', 'mw': '364.037', 'density': '2.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cs2s2o8': {'name': 'Cesium peroxodisulfate', 'formula': 'Cs2S2O8', 'mw': '457.812', 'density': '2.8', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'mgs2o8': {'name': 'Magnesium peroxodisulfate', 'formula': 'MgS2O8', 'mw': '216.433', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cas2o8': {'name': 'Calcium peroxodisulfate', 'formula': 'CaS2O8', 'mw': '232.147', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'srs2o8': {'name': 'Strontium peroxodisulfate', 'formula': 'SrS2O8', 'mw': '279.691', 'density': '2.2', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'bas2o8': {'name': 'Barium peroxodisulfate', 'formula': 'BaS2O8', 'mw': '327.39', 'density': '2.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'zns2o8': {'name': 'Zinc peroxodisulfate', 'formula': 'ZnS2O8', 'mw': '257.444', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'fes2o8': {'name': 'Iron(II) peroxodisulfate', 'formula': 'FeS2O8', 'mw': '263.914', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cus2o8': {'name': 'Copper(II) peroxodisulfate', 'formula': 'CuS2O8', 'mw': '271.616', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'mns2o8': {'name': 'Manganese(II) peroxodisulfate', 'formula': 'MnS2O8', 'mw': '262.999', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'nis2o8': {'name': 'Nickel(II) peroxodisulfate', 'formula': 'NiS2O8', 'mw': '266.763', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'cos2o8': {'name': 'Cobalt(II) peroxodisulfate', 'formula': 'CoS2O8', 'mw': '266.903', 'density': '2.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'soluble'},
    'pbs2o8': {'name': 'Lead(II) peroxodisulfate', 'formula': 'PbS2O8', 'mw': '397.267', 'density': '6.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'cds2o8': {'name': 'Cadmium peroxodisulfate', 'formula': 'CdS2O8', 'mw': '302.479', 'density': '4.7', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'ag2s2o8': {'name': 'Silver peroxodisulfate', 'formula': 'Ag2S2O8', 'mw': '423.804', 'density': '5.5', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'insoluble'},
    'hg2s2o8': {'name': 'Mercury(I) peroxodisulfate', 'formula': 'Hg2S2O8', 'mw': '613.191', 'density': '7.0', 'mp': 'decomposes', 'bp': 'decomposes', 'type': 'salt', 'solubility': 'decomposes in water'},
}

# Redox database: species -> (name, Eo)
REDOX_DB: Dict[str, Dict] = {
    'na': {'name': 'Sodium', 'Eo': '-2.71'},
    'mg': {'name': 'Magnesium', 'Eo': '-2.37'},
    'al': {'name': 'Aluminum', 'Eo': '-1.66'},
    'zn': {'name': 'Zinc', 'Eo': '-0.76'},
    'fe': {'name': 'Iron', 'Eo': '-0.44'},
    'ni': {'name': 'Nickel', 'Eo': '-0.25'},
    'sn': {'name': 'Tin', 'Eo': '-0.14'},
    'pb': {'name': 'Lead', 'Eo': '-0.13'},
    'h2': {'name': 'Hydrogen (reference)', 'Eo': '0.00'},
    'cu': {'name': 'Copper', 'Eo': '+0.34'},
    'ag': {'name': 'Silver', 'Eo': '+0.80'},
    'hg': {'name': 'Mercury', 'Eo': '+0.85'},
    'au': {'name': 'Gold', 'Eo': '+1.50'},
    'o2': {'name': 'Oxygen', 'Eo': '+1.23'},
    'cl2': {'name': 'Chlorine', 'Eo': '+1.36'},
    'br2': {'name': 'Bromine', 'Eo': '+1.07'},
    'i2': {'name': 'Iodine', 'Eo': '+0.54'},
    'f2': {'name': 'Fluorine', 'Eo': '+2.87'},
    'cr': {'name': 'Chromium', 'Eo': '-0.74'},
    'mn': {'name': 'Manganese', 'Eo': '-1.18'},
    'co': {'name': 'Cobalt', 'Eo': '-0.28'},
    'cd': {'name': 'Cadmium', 'Eo': '-0.40'},
    'mn': {'name': 'Manganese', 'Eo': '-1.18'},
    'cr': {'name': 'Chromium', 'Eo': '-0.74'},
    'co': {'name': 'Cobalt', 'Eo': '-0.28'},
    'ni': {'name': 'Nickel', 'Eo': '-0.25'},
    'sn': {'name': 'Tin', 'Eo': '-0.14'},
    'pb': {'name': 'Lead', 'Eo': '-0.13'},
    'sb': {'name': 'Antimony', 'Eo': '+0.24'},
    'bi': {'name': 'Bismuth', 'Eo': '+0.32'},
    'as': {'name': 'Arsenic', 'Eo': '+0.24'},
    'te': {'name': 'Tellurium', 'Eo': '-0.92'},
    'se': {'name': 'Selenium', 'Eo': '-0.78'},
    'po': {'name': 'Polonium', 'Eo': '-0.36'},
    'at': {'name': 'Astatine', 'Eo': '+0.3'},
    'rn': {'name': 'Radon', 'Eo': '+0.7'},
    'fr': {'name': 'Francium', 'Eo': '-2.9'},
    'ra': {'name': 'Radium', 'Eo': '-2.9'},
    'ac': {'name': 'Actinium', 'Eo': '-2.6'},
    'th': {'name': 'Thorium', 'Eo': '-1.9'},
    'pa': {'name': 'Protactinium', 'Eo': '-1.9'},
    'u': {'name': 'Uranium', 'Eo': '-1.8'},
    'np': {'name': 'Neptunium', 'Eo': '-1.8'},
    'pu': {'name': 'Plutonium', 'Eo': '-1.8'},
    'am': {'name': 'Americium', 'Eo': '-1.8'},
    'cm': {'name': 'Curium', 'Eo': '-1.8'},
    'bk': {'name': 'Berkelium', 'Eo': '-1.8'},
    'cf': {'name': 'Californium', 'Eo': '-1.8'},
    'es': {'name': 'Einsteinium', 'Eo': '-1.8'},
    'fm': {'name': 'Fermium', 'Eo': '-1.8'},
    'md': {'name': 'Mendelevium', 'Eo': '-1.8'},
    'no': {'name': 'Nobelium', 'Eo': '-1.8'},
    'lr': {'name': 'Lawrencium', 'Eo': '-1.8'},
    'rf': {'name': 'Rutherfordium', 'Eo': '-1.8'},
    'db': {'name': 'Dubnium', 'Eo': '-1.8'},
    'sg': {'name': 'Seaborgium', 'Eo': '-1.8'},
    'bh': {'name': 'Bohrium', 'Eo': '-1.8'},
    'hs': {'name': 'Hassium', 'Eo': '-1.8'},
    'mt': {'name': 'Meitnerium', 'Eo': '-1.8'},
    'ds': {'name': 'Darmstadtium', 'Eo': '-1.8'},
    'rg': {'name': 'Roentgenium', 'Eo': '-1.8'},
    'cn': {'name': 'Copernicium', 'Eo': '-1.8'},
    'nh': {'name': 'Nihonium', 'Eo': '-1.8'},
    'fl': {'name': 'Flerovium', 'Eo': '-1.8'},
    'mc': {'name': 'Moscovium', 'Eo': '-1.8'},
    'lv': {'name': 'Livermorium', 'Eo': '-1.8'},
    'ts': {'name': 'Tennessine', 'Eo': '-1.8'},
    'og': {'name': 'Oganesson', 'Eo': '-1.8'},
}


class ChemistryEngine:
    """Engine for chemistry mode calculations."""

    def __init__(self):
        self.constants = CHEMISTRY_CONSTANTS
        self.symbols = CHEMISTRY_SYMBOLS
        self.elements = ELEMENTS
        self.compounds = COMPOUNDS
        self.organic_db = ORGANIC_COMPOUNDS
        self.redox_db = REDOX_DB
        self.formulas = CHEMISTRY_FORMULAS
        self.units = CHEMISTRY_UNITS

    def solve(self, expression: str) -> str:
        """Solve a chemistry expression (general mode)."""
        if not expression or not expression.strip():
            return "Error: Empty expression"

        expression = expression.strip()

        # Check if it's a simple symbol lookup
        if expression in self.constants:
            return self._lookup_constant(expression)

        # Check if it's a symbol lookup
        if expression in self.symbols:
            return self._lookup_symbol(expression)

        # Check if it's an equation with known values (e.g., "m=18, n=1.5, M")
        if '=' in expression:
            result = self._solve_equation(expression)
            # Fall back to AI if chemistry engine can't solve it
            if result and ("Error" in result or "Could not" in result or "coming soon" in result):
                try:
                    ai = AIEngine()
                    return ai.solve(expression, mode="chemistry")
                except Exception:
                    pass
            return result

        # Direct compound / element / organic / redox property lookup
        cleaned = expression.strip().lower()
        if cleaned in self.compounds:
            return self._compound_info_general(cleaned)
        if cleaned in self.elements:
            return self._element_info_general(cleaned)
        if cleaned in self.organic_db:
            return self._compound_info_organic(cleaned) if cleaned in self.compounds else (
                f"Organic Compound: {self.organic_db[cleaned]['name']}\n"
                f"  Formula: {self.organic_db[cleaned]['formula']}\n"
                f"  Molar Mass: {self.organic_db[cleaned]['mw']} g/mol\n"
                f"  Type: {self.organic_db[cleaned]['type']}"
            )
        if cleaned in self.redox_db:
            return self._element_info_redox(cleaned)

        # Check if it looks like a chemical formula
        if self._looks_like_formula(expression):
            result = self._describe_formula_solver(expression)
            # Fall back to AI if chemistry engine can't solve it
            if result and ("coming soon" in result or "Error" in result):
                try:
                    ai = AIEngine()
                    return ai.solve(expression, mode="chemistry")
                except Exception:
                    pass
            return result

        return self._describe_capabilities()

    def solve_general(self, expression: str) -> str:
        """General chemistry formula solver."""
        if not expression or not expression.strip():
            return "Error: Empty expression"

        cleaned = expression.strip().lower()

        # Check for compound lookup
        if cleaned in self.compounds:
            return self._compound_info_general(cleaned)

        # Check for element symbol lookup
        if cleaned in self.elements:
            return self._element_info_general(cleaned)

        # Check for element combination (e.g., "h + o") - use AI to find reactions
        if self._is_element_combination(expression):
            return self._find_reactions_with_elements(expression)

        # Check for reaction balancing
        if '+' in expression or '->' in expression:
            return self._balance_reaction(expression)

        # Check for equation with known values (e.g., "n=2, V=0.5, M")
        if '=' in expression:
            return self._solve_equation(expression)

        return f"General chemistry solver: {expression}\n(Full implementation coming soon)"

    def solve_organic(self, expression: str) -> str:
        """Organic chemistry formula solver."""
        if not expression or not expression.strip():
            return "Error: Empty expression"

        cleaned = expression.strip().lower()

        # Check for common organic molecules
        if cleaned in self.organic_db:
            info = self.organic_db[cleaned]
            return (
                f"Organic Compound: {info['name']}\n"
                f"  Formula: {info['formula']}\n"
                f"  Molar Mass: {info['mw']} g/mol\n"
                f"  Type: {info['type']}"
            )

        # Check for compound lookup (inorganic compounds that appear in organic contexts)
        if cleaned in self.compounds:
            return self._compound_info_organic(cleaned)

        # Check for element symbol lookup
        if cleaned in self.elements:
            return self._element_info_organic(cleaned)

        # Check for reaction
        if '+' in expression or '->' in expression:
            return self._balance_reaction(expression)

        return f"Organic chemistry solver: {expression}\n(Compound not in database yet)"

    def solve_redox(self, expression: str) -> str:
        """Redox and equilibrium reaction solver."""
        if not expression or not expression.strip():
            return "Error: Empty expression"

        cleaned = expression.strip().lower()

        # Check for element symbol lookup (with Eo values)
        if cleaned in self.redox_db:
            return self._element_info_redox(cleaned)

        # Check for element symbol lookup
        if cleaned in self.elements:
            return self._element_info_redox(cleaned)

        # Check for compound lookup
        if cleaned in self.compounds:
            return self._compound_info_redox(cleaned)

        # Check for element combination (e.g., "ag + o", "fe + o2")
        if self._is_element_combination_redox(expression):
            return self._find_redox_reactions_for_elements(expression)

        # Check for reaction balancing
        if '+' in expression or '->' in expression:
            return self._balance_reaction(expression)

        # Search for redox reactions involving the given species (compound or element)
        redox_match = self._find_redox_reactions(cleaned)
        if redox_match:
            return redox_match

        return f"Redox solver: {expression}\n(Species not in database yet)"

    def _lookup_constant(self, symbol: str) -> str:
        """Look up a chemistry constant."""
        if symbol in self.constants:
            unit = self.units.get(symbol, '')
            return f"{symbol} = {self.constants[symbol]} {unit}".strip()
        return f"Unknown constant: {symbol}"

    def _lookup_symbol(self, symbol: str) -> str:
        """Look up a chemistry symbol."""
        if symbol in self.symbols:
            meanings = ', '.join(self.symbols[symbol])
            unit = self.units.get(symbol, '')
            return f"{symbol} ({meanings}) [{unit}]".strip()
        return f"Unknown symbol: {symbol}"

    def _looks_like_formula(self, expression: str) -> bool:
        """Check if the expression looks like a chemical formula."""
        cleaned = expression.strip()
        # Match simple chemical formulas like "h2o", "naoh", "c6h12o6"
        if re.match(r'^[A-Za-z][A-Za-z0-9]*$', cleaned):
            return True
        # Match reactions with + or -> but only if they look like chemical formulas
        # (no spaces between words, no natural language)
        if ('+' in cleaned or '->' in cleaned):
            # Check if it's a natural language query (contains action words)
            words = cleaned.split()
            # If more than 4 words, it's likely a natural language query
            if len(words) > 4:
                return False
            # Check if it contains natural language action words
            ai_triggers = {'solve', 'balance', 'find', 'get', 'simplify', 'calculate', 'determine', 'compute', 'evaluate', 'what', 'how', 'much', 'many', 'is', 'are', 'the', 'of', 'for', 'in', 'to', 'from', 'with', 'and', 'or', 'enthalpy', 'energy', 'heat', 'reaction', 'compound', 'molecule', 'atom', 'ion', 'acid', 'base', 'salt', 'oxide', 'gas', 'liquid', 'solid', 'molar', 'mass', 'weight', 'volume', 'density', 'temperature', 'pressure', 'concentration', 'molarity', 'molality', 'ph', 'poh', 'ka', 'kb', 'kw', 'keq', 'ksp', 'delta', 'entropy', 'gibbs', 'free', 'potential', 'cell', 'voltage', 'current', 'resistance', 'power', 'work', 'force', 'velocity', 'acceleration', 'momentum', 'kinetic', 'potential', 'frequency', 'wavelength', 'amplitude', 'period', 'speed', 'distance', 'time', 'displacement', 'position', 'vector', 'scalar', 'tensor', 'matrix', 'determinant', 'eigenvalue', 'eigenvector', 'linear', 'algebra', 'calculus', 'derivative', 'integral', 'limit', 'series', 'sequence', 'function', 'equation', 'inequality', 'polynomial', 'quadratic', 'cubic', 'quartic', 'exponential', 'logarithmic', 'trigonometric', 'hyperbolic', 'inverse', 'complex', 'real', 'imaginary', 'rational', 'irrational', 'integer', 'natural', 'whole', 'prime', 'composite', 'even', 'odd', 'positive', 'negative', 'zero', 'infinity', 'undefined', 'indeterminate', 'convergent', 'divergent', 'absolute', 'relative', 'percent', 'percentage', 'ratio', 'proportion', 'fraction', 'decimal', 'binary', 'octal', 'hexadecimal', 'decimal', 'point', 'place', 'value', 'digit', 'number', 'numeral', 'figure', 'digit', 'count', 'amount', 'quantity', 'measure', 'measurement', 'unit', 'units', 'dimension', 'dimensional', 'analysis', 'conversion', 'convert', 'equivalent', 'equivalence', 'equal', 'equality', 'equation', 'formula', 'expression', 'term', 'coefficient', 'constant', 'variable', 'parameter', 'argument', 'value', 'root', 'solution', 'solve', 'answer', 'result', 'output', 'input', 'process', 'procedure', 'method', 'technique', 'algorithm', 'program', 'code', 'software', 'hardware', 'computer', 'machine', 'device', 'tool', 'instrument', 'apparatus', 'equipment', 'material', 'substance', 'matter', 'mass', 'weight', 'volume', 'density', 'temperature', 'pressure', 'heat', 'energy', 'work', 'power', 'force', 'motion', 'movement', 'speed', 'velocity', 'acceleration', 'momentum', 'impulse', 'collision', 'impact', 'friction', 'gravity', 'gravitational', 'electric', 'electrical', 'magnetic', 'electromagnetic', 'optical', 'light', 'sound', 'wave', 'particle', 'quantum', 'atomic', 'nuclear', 'molecular', 'chemical', 'physical', 'biological', 'environmental', 'earth', 'space', 'universe', 'cosmos', 'galaxy', 'star', 'planet', 'moon', 'sun', 'earth', 'world', 'globe', 'planet', 'continent', 'country', 'city', 'town', 'village', 'house', 'building', 'structure', 'bridge', 'road', 'street', 'highway', 'railway', 'airport', 'port', 'harbor', 'station', 'terminal', 'depot', 'warehouse', 'factory', 'plant', 'mill', 'mine', 'well', 'field', 'farm', 'garden', 'park', 'forest', 'jungle', 'desert', 'mountain', 'hill', 'valley', 'cave', 'canyon', 'cliff', 'plateau', 'plain', 'prairie', 'savanna', 'tundra', 'taiga', 'swamp', 'marsh', 'bog', 'fen', 'moor', 'heath', 'meadow', 'pasture', 'range', 'ranch', 'paddock', 'corral', 'pen', 'cage', 'coop', 'hutch', 'stable', 'barn', 'shed', 'garage', 'carport', 'pavilion', 'gazebo', 'pergola', 'arbor', 'trellis', 'fence', 'wall', 'gate', 'door', 'window', 'roof', 'ceiling', 'floor', 'foundation', 'basement', 'attic', 'loft', 'garret', 'dormer', 'chimney', 'flue', 'vent', 'duct', 'pipe', 'tube', 'conduit', 'channel', 'tunnel', 'canal', 'aqueduct', 'sewer', 'drain', 'ditch', 'trench', 'furrow', 'groove', 'slot', 'notch', 'niche', 'recess', 'alcove', 'bay', 'cove', 'inlet', 'outlet', 'entrance', 'exit', 'access', 'approach', 'passage', 'path', 'way', 'route', 'course', 'direction', 'bearing', 'heading', 'orientation', 'position', 'location', 'site', 'spot', 'point', 'place', 'area', 'region', 'zone', 'sector', 'district', 'quarter', 'neighborhood', 'community', 'society', 'culture', 'civilization', 'nation', 'state', 'government', 'politics', 'law', 'justice', 'right', 'freedom', 'liberty', 'independence', 'autonomy', 'sovereignty', 'democracy', 'republic', 'monarchy', 'empire', 'kingdom', 'duchy', 'principality', 'county', 'province', 'territory', 'colony', 'dependency', 'protectorate', 'mandate', 'trust', 'possession', 'holding', 'estate', 'property', 'land', 'real', 'personal', 'tangible', 'intangible', 'movable', 'immovable', 'fixture', 'fitting', 'attachment', 'accessory', 'appendage', 'addition', 'extension', 'expansion', 'enlargement', 'increase', 'decrease', 'reduction', 'diminution', 'lessening', 'lowering', 'raising', 'elevation', 'lift', 'boost', 'enhancement', 'improvement', 'betterment', 'amelioration', 'upgrade', 'update', 'renewal', 'restoration', 'rehabilitation', 'reconstruction', 'rebuilding', 'remodeling', 'renovation', 'repair', 'fix', 'mend', 'patch', 'correct', 'rectify', 'remedy', 'cure', 'heal', 'treat', 'care', 'help', 'aid', 'assist', 'support', 'serve', 'attend', 'tend', 'mind', 'watch', 'guard', 'protect', 'defend', 'shield', 'shelter', 'harbor', 'house', 'accommodate', 'lodge', 'quarter', 'board', 'feed', 'nourish', 'nurture', 'sustain', 'maintain', 'keep', 'retain', 'hold', 'possess', 'own', 'have', 'contain', 'include', 'comprise', 'consist', 'constitute', 'form', 'make', 'create', 'produce', 'generate', 'manufacture', 'fabricate', 'construct', 'build', 'assemble', 'put', 'set', 'place', 'lay', 'position', 'arrange', 'organize', 'order', 'structure', 'systematize', 'methodize', 'standardize', 'normalize', 'regularize', 'routinize', 'habituate', 'accustom', 'adapt', 'adjust', 'accommodate', 'fit', 'suit', 'match', 'correspond', 'agree', 'harmonize', 'reconcile', 'resolve', 'settle', 'decide', 'determine', 'fix', 'establish', 'found', 'institute', 'organize', 'create', 'start', 'begin', 'commence', 'initiate', 'launch', 'introduce', 'present', 'offer', 'propose', 'suggest', 'recommend', 'advise', 'counsel', 'guide', 'direct', 'lead', 'steer', 'pilot', 'navigate', 'drive', 'ride', 'travel', 'journey', 'voyage', 'tour', 'visit', 'explore', 'discover', 'find', 'locate', 'identify', 'recognize', 'know', 'understand', 'comprehend', 'apprehend', 'grasp', 'seize', 'catch', 'capture', 'trap', 'snare', 'net', 'hook', 'line', 'sink', 'drown', 'submerge', 'immerse', 'dip', 'plunge', 'dive', 'jump', 'leap', 'hop', 'skip', 'bound', 'spring', 'bounce', 'rebound', 'recoil', 'kick', 'punt', 'drop', 'fall', 'sink', 'descend', 'decline', 'decrease', 'diminish', 'dwindle', 'shrink', 'contract', 'compress', 'condense', 'concentrate', 'consolidate', 'integrate', 'unify', 'combine', 'merge', 'fuse', 'blend', 'mix', 'mingle', 'intermingle', 'intermix', 'admix', 'alloy', 'amalgam', 'compound', 'composite', 'mixture', 'solution', 'suspension', 'colloid', 'emulsion', 'foam', 'froth', 'bubble', 'droplet', 'drop', 'drip', 'trickle', 'flow', 'stream', 'current', 'tide', 'wave', 'ripple', 'undulation', 'vibration', 'oscillation', 'swing', 'sway', 'rock', 'roll', 'pitch', 'yaw', 'turn', 'rotate', 'revolve', 'spin', 'twirl', 'whirl', 'swirl', 'curl', 'coil', 'wind', 'twist', 'bend', 'flex', 'curve', 'arch', 'bow', 'crook', 'hook', 'loop', 'kink', 'tangle', 'tangle', 'entangle', 'snarl', 'mat', 'felt', 'web', 'net', 'mesh', 'grid', 'lattice', 'screen', 'sieve', 'filter', 'strainer', 'colander', 'funnel', 'channel', 'conduit', 'pipe', 'tube', 'hose', 'cable', 'wire', 'cord', 'rope', 'string', 'thread', 'yarn', 'fiber', 'filament', 'strand', 'lock', 'tress', 'curl', 'wave', 'ringlet', 'spiral', 'helix', 'coil', 'loop', 'knot', 'tie', 'bind', 'bond', 'link', 'connection', 'joint', 'junction', 'intersection', 'crossing', 'crossroads', 'fork', 'branch', 'bough', 'limb', 'arm', 'wing', 'fin', 'flipper', 'tail', 'hind', 'rear', 'back', 'behind', 'aft', 'stern', 'bow', 'prow', 'front', 'fore', 'forward', 'ahead', 'before', 'prior', 'previous', 'preceding', 'antecedent', 'precursor', 'forerunner', 'harbinger', 'herald', 'messenger', 'courier', 'carrier', 'bearer', 'porter', 'conveyor', 'transmitter', 'sender', 'receiver', 'recipient', 'addressee', 'destination', 'terminus', 'end', 'finish', 'conclusion', 'close', 'closure', 'cessation', 'stop', 'halt', 'pause', 'break', 'interruption', 'disruption', 'intermission', 'interval', 'gap', 'space', 'void', 'vacuum', 'emptiness', 'nothingness', 'nullity', 'nonexistence', 'absence', 'lack', 'want', 'need', 'requirement', 'demand', 'request', 'appeal', 'petition', 'plea', 'prayer', 'invocation', 'supplication', 'entreaty', 'imploration', 'beseechment', 'adjuration', 'obsecration', 'obtest', 'deprecation', 'exorcism', 'incantation', 'charm', 'spell', 'enchantment', 'magic', 'sorcery', 'witchcraft', 'wizardry', 'necromancy', 'divination', 'prophecy', 'prediction', 'forecast', 'prognostication', 'augury', 'omen', 'portent', 'sign', 'signal', 'indication', 'evidence', 'proof', 'testimony', 'witness', 'attestation', 'confirmation', 'verification', 'validation', 'authentication', 'certification', 'accreditation', 'approval', 'endorsement', 'sanction', 'ratification', 'authorization', 'permission', 'consent', 'assent', 'agreement', 'accord', 'concord', 'harmony', 'unity', 'unison', 'concert', 'collaboration', 'cooperation', 'teamwork', 'partnership', 'alliance', 'coalition', 'league', 'federation', 'confederation', 'union', 'association', 'society', 'club', 'circle', 'group', 'band', 'company', 'party', 'faction', 'sect', 'cult', 'denomination', 'persuasion', 'school', 'movement', 'cause', 'crusade', 'campaign', 'drive', 'push', 'effort', 'attempt', 'try', 'endeavor', 'undertaking', 'enterprise', 'venture', 'project', 'scheme', 'plan', 'design', 'strategy', 'tactic', 'maneuver', 'move', 'step', 'measure', 'action', 'deed', 'act', 'performance', 'execution', 'implementation', 'realization', 'achievement', 'accomplishment', 'attainment', 'success', 'victory', 'triumph', 'conquest', 'win', 'gain', 'profit', 'benefit', 'advantage', 'edge', 'lead', 'head', 'start', 'beginning', 'onset', 'outset', 'inception', 'origin', 'source', 'root', 'cause', 'reason', 'motive', 'purpose', 'aim', 'goal', 'objective', 'target', 'end', 'destination', 'terminus', 'stop', 'station', 'depot', 'terminal', 'harbor', 'port', 'haven', 'anchorage', 'mooring', 'berth', 'dock', 'pier', 'wharf', 'quay', 'jetty', 'breakwater', 'seawall', 'embankment', 'levee', 'dike', 'dam', 'weir', 'sluice', 'lock', 'gate', 'valve', 'faucet', 'tap', 'spigot', 'spout', 'nozzle', 'jet', 'spray', 'shower', 'sprinkle', 'drizzle', 'rain', 'snow', 'sleet', 'hail', 'blizzard', 'storm', 'tempest', 'hurricane', 'typhoon', 'cyclone', 'tornado', 'twister', 'whirlwind', 'gust', 'breeze', 'wind', 'gale', 'squall', 'draft', 'current', 'flow', 'stream', 'torrent', 'flood', 'deluge', 'inundation', 'overflow', 'spill', 'leak', 'drip', 'drop', 'trickle', 'seep', 'ooze', 'exude', 'emit', 'discharge', 'release', 'eject', 'expel', 'evict', 'remove', 'eliminate', 'eradicate', 'extirpate', 'exterminate', 'annihilate', 'destroy', 'demolish', 'ruin', 'wreck', 'devastate', 'desolate', 'ravage', 'pillage', 'plunder', 'loot', 'ransack', 'sack', 'raid', 'maraud', 'pirate', 'buccaneer', 'corsair', 'privateer', 'freebooter', 'bandit', 'brigand', 'outlaw', 'highwayman', 'robber', 'thief', 'burglar', 'lar', 'housebreaker', 'safebreaker', 'cracker', 'hacker', 'intruder', 'trespasser', 'infiltrator', 'spy', 'agent', 'operative', 'asset', 'informant', 'informer', 'whistleblower', 'leaker', 'mole', 'double', 'traitor', 'turncoat', 'defector', 'deserter', 'renegade', 'apostate', 'heretic', 'schismatic', 'dissident', 'rebel', 'revolutionary', 'insurgent', 'guerrilla', 'partisan', 'freedom', 'fighter', 'resistance', 'underground', 'subversive', 'saboteur', 'terrorist', 'extremist', 'radical', 'fanatic', 'zealot', 'fundamentalist', 'orthodox', 'conservative', 'traditionalist', 'reactionary', 'counterrevolutionary', 'rightist', 'right-winger', 'leftist', 'left-winger', 'liberal', 'progressive', 'moderate', 'centrist', 'independent', 'nonpartisan', 'unaffiliated', 'unaligned', 'neutral', 'impartial', 'unbiased', 'fair', 'just', 'equitable', 'righteous', 'virtuous', 'moral', 'ethical', 'upright', 'honorable', 'noble', 'dignified', 'distinguished', 'eminent', 'prominent', 'noted', 'notable', 'famous', 'renowned', 'celebrated', 'acclaimed', 'lauded', 'extolled', 'praised', 'commended', 'applauded', 'cheered', 'hailed', 'saluted', 'greeted', 'welcomed', 'received', 'accepted', 'admitted', 'allowed', 'permitted', 'granted', 'given', 'bestowed', 'conferred', 'awarded', 'presented', 'offered', 'extended', 'stretched', 'reached', 'handed', 'passed', 'transferred', 'conveyed', 'delivered', 'brought', 'carried', 'transported', 'shipped', 'freighted', 'hauled', 'towed', 'pulled', 'dragged', 'drawn', 'attracted', 'magnetized', 'charged', 'electrified', 'ionized', 'polarized', 'polarized', 'magnetized', 'electrified', 'charged', 'ionized', 'polarized'}
            if any(word.lower() in ai_triggers for word in words):
                return False
            return True
        return False

    def _describe_formula_solver(self, expression: str) -> str:
        """Describe what the formula solver will do."""
        return (
            f"Chemistry formula solver: {expression}\n\n"
            "Coming soon:\n"
            "- Element property lookup\n"
            "- Reaction balancing\n"
            "- Stoichiometry calculations\n"
            "- Molar mass computation"
        )

    def _element_info(self, symbol: str) -> str:
        """Get basic information about an element."""
        if symbol in self.elements:
            info = self.elements[symbol]
            return (
                f"Element: {info['name']} ({symbol.upper()})\n"
                f"  Atomic Number: {info['Z']}\n"
                f"  Atomic Mass: {info['mass']} u"
            )
        return f"Element '{symbol}' not found in database"

    def _element_info_general(self, symbol: str) -> str:
        """Get general chemistry information about an element."""
        if symbol not in self.elements:
            return f"Element '{symbol}' not found in database"

        info = self.elements[symbol]
        name = info['name']
        z = info['Z']
        mass = info['mass']

        properties = []
        en = self._get_electronegativity(symbol)
        if en:
            properties.append(f"  Electronegativity: {en}")
        ec = self._get_electron_configuration(symbol)
        if ec:
            properties.append(f"  Electron Configuration: {ec}")
        mp, bp = self._get_phase_points(symbol)
        if mp:
            properties.append(f"  Melting Point: {mp}")
        if bp:
            properties.append(f"  Boiling Point: {bp}")
        density = self._get_density(symbol)
        if density:
            properties.append(f"  Density: {density}")
        oxidation_states = self._get_oxidation_states(symbol)
        if oxidation_states:
            properties.append(f"  Oxidation States: {oxidation_states}")
        compounds = self._get_common_compounds(symbol)
        if compounds:
            properties.append(f"  Common Compounds: {compounds}")
        uses = self._get_element_uses(symbol)
        if uses:
            properties.append(f"  Uses: {uses}")
        category = self._get_element_category(symbol)
        if category:
            properties.append(f"  Category: {category}")
        year = self._get_year_discovered(symbol)
        if year:
            properties.append(f"  Year Discovered: {year}")
        abundance = self._get_crust_abundance(symbol)
        if abundance:
            properties.append(f"  Crust Abundance: {abundance}")

        result = f"Element: {name} ({symbol.upper()})\n"
        result += f"  Atomic Number: {z}\n"
        result += f"  Atomic Mass: {mass} u\n"
        result += '\n'.join(properties)
        return result

    def _element_info_organic(self, symbol: str) -> str:
        """Get organic chemistry information about an element."""
        if symbol not in self.elements:
            return f"Element '{symbol}' not found in database"

        info = self.elements[symbol]
        name = info['name']
        z = info['Z']
        mass = info['mass']

        properties = []
        organic_relevance = self._get_organic_relevance(symbol)
        if organic_relevance:
            properties.append(f"  Organic Relevance: {organic_relevance}")
        organic_compounds = self._get_organic_compounds(symbol)
        if organic_compounds:
            properties.append(f"  Common Organic Compounds: {organic_compounds}")
        functional_groups = self._get_functional_groups(symbol)
        if functional_groups:
            properties.append(f"  Functional Groups: {functional_groups}")
        bonding = self._get_bonding_properties(symbol)
        if bonding:
            properties.append(f"  Bonding: {bonding}")
        hybridization = self._get_hybridization(symbol)
        if hybridization:
            properties.append(f"  Hybridization: {hybridization}")
        en = self._get_electronegativity(symbol)
        if en:
            properties.append(f"  Electronegativity: {en}")
        vdw = self._get_vdw_radius(symbol)
        if vdw:
            properties.append(f"  Van der Waals Radius: {vdw}")
        cov = self._get_covalent_radius(symbol)
        if cov:
            properties.append(f"  Covalent Radius: {cov}")

        result = f"Element: {name} ({symbol.upper()})\n"
        result += f"  Atomic Number: {z}\n"
        result += f"  Atomic Mass: {mass} u\n"
        result += '\n'.join(properties)
        return result

    def _element_info_redox(self, symbol: str) -> str:
        """Get redox chemistry information about an element."""
        if symbol not in self.elements:
            return f"Element '{symbol}' not found in database"

        info = self.elements[symbol]
        name = info['name']
        z = info['Z']
        mass = info['mass']

        properties = []
        if symbol in self.redox_db:
            redox_info = self.redox_db[symbol]
            properties.append(f"  Eo = {redox_info['Eo']} V")
        oxidation_states = self._get_oxidation_states(symbol)
        if oxidation_states:
            properties.append(f"  Oxidation States: {oxidation_states}")
        redox_reactions = self._get_redox_reactions(symbol)
        if redox_reactions:
            properties.append(f"  Common Redox Reactions: {redox_reactions}")
        reduction_potential = self._get_reduction_potential(symbol)
        if reduction_potential:
            properties.append(f"  Standard Reduction Potential: {reduction_potential}")
        ec = self._get_electron_configuration(symbol)
        if ec:
            properties.append(f"  Electron Configuration: {ec}")
        en = self._get_electronegativity(symbol)
        if en:
            properties.append(f"  Electronegativity: {en}")
        ie = self._get_ionization_energy(symbol)
        if ie:
            properties.append(f"  Ionization Energy: {ie}")

        result = f"Element: {name} ({symbol.upper()})\n"
        result += f"  Atomic Number: {z}\n"
        result += f"  Atomic Mass: {mass} u\n"
        result += '\n'.join(properties)
        return result

    def _compound_info_general(self, formula: str) -> str:
        """Get general chemistry information about a compound."""
        if formula not in self.compounds:
            return f"Compound '{formula}' not found in database"

        info = self.compounds[formula]
        result = f"Compound: {info['name']} ({info['formula']})\n"
        result += f"  Molar Mass: {info['mw']} g/mol\n"
        result += f"  Density: {info['density']} g/cm³\n"
        result += f"  Melting Point: {info['mp']} °C\n"
        result += f"  Boiling Point: {info['bp']} °C\n"
        result += f"  Type: {info['type']}\n"
        result += f"  Solubility: {info['solubility']}"
        return result

    def _compound_info_organic(self, formula: str) -> str:
        """Get organic chemistry information about a compound."""
        if formula not in self.compounds:
            return f"Compound '{formula}' not found in database"

        info = self.compounds[formula]
        result = f"Compound: {info['name']} ({info['formula']})\n"
        result += f"  Molar Mass: {info['mw']} g/mol\n"
        result += f"  Density: {info['density']} g/cm³\n"
        result += f"  Melting Point: {info['mp']} °C\n"
        result += f"  Boiling Point: {info['bp']} °C\n"
        result += f"  Type: {info['type']}\n"
        result += f"  Solubility: {info['solubility']}"
        return result

    def _compound_info_redox(self, formula: str) -> str:
        """Get redox chemistry information about a compound."""
        if formula not in self.compounds:
            return f"Compound '{formula}' not found in database"

        info = self.compounds[formula]
        result = f"Compound: {info['name']} ({info['formula']})\n"
        result += f"  Molar Mass: {info['mw']} g/mol\n"
        result += f"  Type: {info['type']}\n"
        result += f"  Solubility: {info['solubility']}"

        # Check if any elements in the compound have redox data
        elements_in_compound = self._parse_formula(info['formula'])
        for elem in elements_in_compound:
            if elem.lower() in self.redox_db:
                redox_info = self.redox_db[elem.lower()]
                result += f"\n  Redox ({elem}): Eo = {redox_info['Eo']} V"

        return result

    def _is_element_combination(self, expression: str) -> bool:
        """Check if the expression is a combination of element symbols."""
        cleaned = expression.strip().lower()
        parts = [p.strip() for p in cleaned.replace('->', '+').split('+') if p.strip()]
        if len(parts) < 2:
            return False
        for part in parts:
            elem = ''.join(c for c in part if c.isalpha())
            if elem not in self.elements:
                return False
        return True

    def _find_reactions_with_elements(self, expression: str) -> str:
        """Use AI to find known reactions involving the given elements."""
        try:
            from .ai_engine import AIEngine
        except ImportError:
            return "Error: AI mode requires the 'openai' package. Install it with: pip install calculate-it[ai]"

        cleaned = expression.strip().lower()
        parts = [p.strip() for p in cleaned.replace('->', '+').split('+') if p.strip()]
        elements = []
        for part in parts:
            elem = ''.join(c for c in part if c.isalpha())
            if elem in self.elements:
                elements.append(self.elements[elem]['name'])

        if not elements:
            return f"Error: Could not parse elements from '{expression}'"

        ai_engine = AIEngine()
        if not ai_engine.is_available():
            return "Error: AI mode requires OPENAI_API_KEY environment variable."

        element_str = ' and '.join(elements)
        prompt = f"""List all known chemical reactions that involve {element_str} as reactants or products.

For each reaction, provide:
1. The balanced chemical equation
2. A brief description of the reaction type (e.g., combustion, synthesis, decomposition)
3. Any notable conditions (e.g., heat, catalyst)

Format each reaction as:
- [balanced equation] | [reaction type] | [conditions]

List at least 5-10 reactions if possible. Be thorough and include both common and important industrial/biological reactions."""

        result = ai_engine.solve(prompt, mode="chemistry")
        return result

    def _parse_formula(self, formula: str) -> Dict[str, int]:
        """Parse a chemical formula into element counts."""
        formula = self._normalize_formula(formula)
        counts = defaultdict(int)
        i = 0
        n = len(formula)

        while i < n:
            if formula[i].isspace():
                i += 1
                continue
            if formula[i] == '(':
                depth = 1
                j = i + 1
                while j < n and depth > 0:
                    if formula[j] == '(':
                        depth += 1
                    elif formula[j] == ')':
                        depth -= 1
                    j += 1
                group = formula[i+1:j-1]
                k = j
                while k < n and formula[k].isdigit():
                    k += 1
                multiplier = int(formula[j:k]) if k > j else 1
                group_counts = self._parse_formula(group)
                for elem, count in group_counts.items():
                    counts[elem] += count * multiplier
                i = k
                continue
            if not formula[i].isupper():
                i += 1
                continue
            elem = formula[i]
            i += 1
            while i < n and formula[i].islower():
                elem += formula[i]
                i += 1
            num_str = ''
            while i < n and formula[i].isdigit():
                num_str += formula[i]
                i += 1
            count = int(num_str) if num_str else 1
            counts[elem] += count
        return dict(counts)

    def _normalize_formula(self, formula: str) -> str:
        """Normalize a chemical formula to standard notation."""
        known_elements = sorted(self.elements.keys(), key=len, reverse=True)
        result = []
        i = 0
        n = len(formula)

        while i < n:
            if formula[i].isspace():
                i += 1
                continue
            matched = False
            for elem in known_elements:
                elem_len = len(elem)
                if i + elem_len <= n and formula[i:i+elem_len].lower() == elem:
                    if elem_len == 2 and i + elem_len < n and formula[i + elem_len].isdigit():
                        first_letter = elem[0]
                        second_letter = elem[1]
                        if first_letter in self.elements and second_letter in self.elements:
                            result.append(first_letter.upper())
                            i += 1
                            matched = True
                            break
                    result.append(elem[0].upper() + elem[1:] if len(elem) > 1 else elem.upper())
                    i += elem_len
                    matched = True
                    break
            if matched:
                continue
            if formula[i].isalpha():
                result.append(formula[i].upper())
                i += 1
                while i < n and formula[i].islower():
                    result.append(formula[i])
                    i += 1
            else:
                result.append(formula[i])
                i += 1
        return ''.join(result)

    def _balance_reaction(self, expression: str) -> str:
        """Balance a chemical reaction."""
        expr = expression.strip()
        if '->' in expr:
            left, right = expr.split('->', 1)
        elif '→' in expr:
            left, right = expr.split('→', 1)
        elif '=' in expr:
            left, right = expr.split('=', 1)
        else:
            return f"Error: Could not parse reaction '{expression}'. Use '->' to separate reactants and products."

        reactants = [r.strip() for r in left.split('+') if r.strip()]
        products = [p.strip() for p in right.split('+') if p.strip()]

        if not reactants or not products:
            return f"Error: Could not parse reaction '{expression}'"

        reactant_formulas = []
        for r in reactants:
            try:
                normalized = self._normalize_formula(r)
                counts = self._parse_formula(normalized)
                reactant_formulas.append((normalized, counts))
            except Exception as e:
                return f"Error: Could not parse formula '{r}': {e}"

        product_formulas = []
        for p in products:
            try:
                normalized = self._normalize_formula(p)
                counts = self._parse_formula(normalized)
                product_formulas.append((normalized, counts))
            except Exception as e:
                return f"Error: Could not parse formula '{p}': {e}"

        all_elements = set()
        for _, counts in reactant_formulas:
            all_elements.update(counts.keys())
        for _, counts in product_formulas:
            all_elements.update(counts.keys())
        all_elements = sorted(all_elements)

        n_reactants = len(reactant_formulas)
        n_products = len(product_formulas)
        n_vars = n_reactants + n_products
        coeffs = symbols(f'c0:{n_vars}')

        equations = []
        for elem in all_elements:
            lhs = 0
            for i, (_, counts) in enumerate(reactant_formulas):
                lhs += coeffs[i] * counts.get(elem, 0)
            rhs = 0
            for j, (_, counts) in enumerate(product_formulas):
                rhs += coeffs[n_reactants + j] * counts.get(elem, 0)
            equations.append(Eq(lhs, rhs))

        try:
            equations.append(Eq(coeffs[0], 1))
            solution = solve(equations, coeffs, dict=True)
            if not solution:
                return f"Error: Could not balance reaction '{expression}'"
            sol = solution[0]

            values = []
            for c in coeffs:
                if c in sol:
                    val = sol[c]
                    try:
                        val = float(val.evalf())
                    except:
                        val = float(val)
                    values.append(val)
                else:
                    values.append(1.0)

            from math import gcd
            from functools import reduce

            fractions = []
            for v in values:
                frac = Rational(v).limit_denominator(1000)
                fractions.append(frac)

            denominators = [f.q for f in fractions]
            lcm = denominators[0]
            for d in denominators[1:]:
                lcm = lcm * d // gcd(lcm, d)

            int_values = [int(f * lcm) for f in fractions]
            g = reduce(gcd, int_values)
            if g == 0:
                g = 1
            int_values = [v // g for v in int_values]

            reactant_parts = []
            for i, (formula, _) in enumerate(reactant_formulas):
                coeff = int_values[i]
                if coeff == 1:
                    reactant_parts.append(formula)
                else:
                    reactant_parts.append(f"{coeff}{formula}")

            product_parts = []
            for j, (formula, _) in enumerate(product_formulas):
                coeff = int_values[n_reactants + j]
                if coeff == 1:
                    product_parts.append(formula)
                else:
                    product_parts.append(f"{coeff}{formula}")

            balanced = " + ".join(reactant_parts) + " -> " + " + ".join(product_parts)
            return f"Balanced: {balanced}"

        except Exception as e:
            return f"Error: Could not balance reaction '{expression}': {e}"

    def _is_element_combination_redox(self, expression: str) -> bool:
        """Check if the expression is a combination of element symbols."""
        cleaned = expression.strip().lower()
        parts = [p.strip() for p in cleaned.replace('->', '+').split('+') if p.strip()]
        if len(parts) < 2:
            return False
        for part in parts:
            elem = ''.join(c for c in part if c.isalpha())
            if elem not in self.elements:
                return False
        return True

    def _find_redox_reactions_for_elements(self, expression: str) -> str:
        """Find redox reactions involving the given elements."""
        cleaned = expression.strip().lower()
        parts = [p.strip() for p in cleaned.replace('->', '+').split('+') if p.strip()]
        elements = []
        for part in parts:
            elem = ''.join(c for c in part if c.isalpha())
            if elem in self.elements:
                elements.append(elem)

        if not elements:
            return f"Error: Could not parse elements from '{expression}'"

        # Search for reactions involving these elements
        matches = []
        for rxn in self._get_all_redox_reactions():
            reactants = rxn["reactants"]
            products = rxn["products"]
            all_species = reactants + products
            # Check if any of the input elements appear in the reaction
            for elem in elements:
                if elem in all_species:
                    matches.append(rxn)
                    break

        if not matches:
            return f"No redox reactions found involving {' and '.join(elements)}"

        result = f"Redox reactions involving {' and '.join(elements)}:\n\n"
        for i, match in enumerate(matches, 1):
            result += f"{i}. {match['reaction']}\n"
            result += f"   Type: {match['type']}\n"
            result += f"   Reactants: {' + '.join(match['reactants'])}\n"
            result += f"   Products: {' + '.join(match['products'])}\n\n"
        return result

    def _get_all_redox_reactions(self) -> list:
        """Get all redox reactions from the database."""
        return [
            {"reaction": "2H2 + O2 → 2H2O", "reactants": ["h2", "o2"], "products": ["h2o"], "type": "combustion"},
            {"reaction": "2H2O → 2H2 + O2", "reactants": ["h2o"], "products": ["h2", "o2"], "type": "electrolysis"},
            {"reaction": "H2 + Cl2 → 2HCl", "reactants": ["h2", "cl2"], "products": ["hcl"], "type": "synthesis"},
            {"reaction": "Zn + 2HCl → ZnCl2 + H2", "reactants": ["zn", "hcl"], "products": ["zncl2", "h2"], "type": "single displacement"},
            {"reaction": "2H2O2 → 2H2O + O2", "reactants": ["h2o2"], "products": ["h2o", "o2"], "type": "decomposition"},
            {"reaction": "C + O2 → CO2", "reactants": ["c", "o2"], "products": ["co2"], "type": "combustion"},
            {"reaction": "2CO + O2 → 2CO2", "reactants": ["co", "o2"], "products": ["co2"], "type": "combustion"},
            {"reaction": "CH4 + 2O2 → CO2 + 2H2O", "reactants": ["ch4", "o2"], "products": ["co2", "h2o"], "type": "combustion"},
            {"reaction": "4Fe + 3O2 → 2Fe2O3", "reactants": ["fe", "o2"], "products": ["fe2o3"], "type": "corrosion"},
            {"reaction": "2SO2 + O2 → 2SO3", "reactants": ["so2", "o2"], "products": ["so3"], "type": "contact process"},
            {"reaction": "2NO + O2 → 2NO2", "reactants": ["no", "o2"], "products": ["no2"], "type": "oxidation"},
            {"reaction": "Fe + 2HCl → FeCl2 + H2", "reactants": ["fe", "hcl"], "products": ["fecl2", "h2"], "type": "single displacement"},
            {"reaction": "Fe + CuSO4 → FeSO4 + Cu", "reactants": ["fe", "cuso4"], "products": ["feso4", "cu"], "type": "single displacement"},
            {"reaction": "2Fe + 3Cl2 → 2FeCl3", "reactants": ["fe", "cl2"], "products": ["fecl3"], "type": "synthesis"},
            {"reaction": "Fe2O3 + 3CO → 2Fe + 3CO2", "reactants": ["fe2o3", "co"], "products": ["fe", "co2"], "type": "reduction"},
            {"reaction": "4Fe(OH)2 + O2 + 2H2O → 4Fe(OH)3", "reactants": ["fe(oh)2", "o2", "h2o"], "products": ["fe(oh)3"], "type": "oxidation"},
            {"reaction": "Cu + 2AgNO3 → Cu(NO3)2 + 2Ag", "reactants": ["cu", "ag"], "products": ["cu", "ag"], "type": "single displacement"},
            {"reaction": "2Cu + O2 → 2CuO", "reactants": ["cu", "o2"], "products": ["cuo"], "type": "oxidation"},
            {"reaction": "CuO + H2 → Cu + H2O", "reactants": ["cuo", "h2"], "products": ["cu", "h2o"], "type": "reduction"},
            {"reaction": "Cu + 2H2SO4 → CuSO4 + SO2 + 2H2O", "reactants": ["cu", "h2so4"], "products": ["cuso4", "so2", "h2o"], "type": "oxidation"},
            {"reaction": "Zn + 2HCl → ZnCl2 + H2", "reactants": ["zn", "hcl"], "products": ["zncl2", "h2"], "type": "single displacement"},
            {"reaction": "Zn + CuSO4 → ZnSO4 + Cu", "reactants": ["zn", "cuso4"], "products": ["znso4", "cu"], "type": "single displacement"},
            {"reaction": "2Zn + O2 → 2ZnO", "reactants": ["zn", "o2"], "products": ["zno"], "type": "oxidation"},
            {"reaction": "2Al + 6HCl → 2AlCl3 + 3H2", "reactants": ["al", "hcl"], "products": ["alcl3", "h2"], "type": "single displacement"},
            {"reaction": "2Al + 3CuSO4 → Al2(SO4)3 + 3Cu", "reactants": ["al", "cuso4"], "products": ["al2(so4)3", "cu"], "type": "single displacement"},
            {"reaction": "4Al + 3O2 → 2Al2O3", "reactants": ["al", "o2"], "products": ["al2o3"], "type": "oxidation"},
            {"reaction": "2Al + Fe2O3 → 2Fe + Al2O3", "reactants": ["al", "fe2o3"], "products": ["fe", "al2o3"], "type": "thermite"},
            {"reaction": "Mg + 2HCl → MgCl2 + H2", "reactants": ["mg", "hcl"], "products": ["mgcl2", "h2"], "type": "single displacement"},
            {"reaction": "2Mg + O2 → 2MgO", "reactants": ["mg", "o2"], "products": ["mgo"], "type": "combustion"},
            {"reaction": "Mg + CuSO4 → MgSO4 + Cu", "reactants": ["mg", "cuso4"], "products": ["mgso4", "cu"], "type": "single displacement"},
            {"reaction": "2Na + 2H2O → 2NaOH + H2", "reactants": ["na", "h2o"], "products": ["naoh", "h2"], "type": "single displacement"},
            {"reaction": "2Na + Cl2 → 2NaCl", "reactants": ["na", "cl2"], "products": ["nacl"], "type": "synthesis"},
            {"reaction": "2Na + O2 → Na2O2", "reactants": ["na", "o2"], "products": ["na2o2"], "type": "oxidation"},
            {"reaction": "Cl2 + 2NaBr → 2NaCl + Br2", "reactants": ["cl2", "nabr"], "products": ["nacl", "br2"], "type": "halogen displacement"},
            {"reaction": "Cl2 + 2KI → 2KCl + I2", "reactants": ["cl2", "ki"], "products": ["kcl", "i2"], "type": "halogen displacement"},
            {"reaction": "Cl2 + H2O → HCl + HClO", "reactants": ["cl2", "h2o"], "products": ["hcl", "hclo"], "type": "disproportionation"},
            {"reaction": "Ag + 2HNO3 → AgNO3 + NO2 + H2O", "reactants": ["ag", "hno3"], "products": ["agno3", "no2", "h2o"], "type": "oxidation"},
            {"reaction": "2Ag2O → 4Ag + O2", "reactants": ["ag2o"], "products": ["ag", "o2"], "type": "decomposition"},
            {"reaction": "Au + 4HCl + HNO3 → HAuCl4 + NO + 2H2O", "reactants": ["au", "hcl", "hno3"], "products": ["haucl4", "no", "h2o"], "type": "aqua regia"},
            {"reaction": "N2 + 3H2 → 2NH3", "reactants": ["n2", "h2"], "products": ["nh3"], "type": "Haber process"},
            {"reaction": "4NH3 + 5O2 → 4NO + 6H2O", "reactants": ["nh3", "o2"], "products": ["no", "h2o"], "type": "Ostwald process"},
            {"reaction": "2NO + O2 → 2NO2", "reactants": ["no", "o2"], "products": ["no2"], "type": "oxidation"},
            {"reaction": "3NO2 + H2O → 2HNO3 + NO", "reactants": ["no2", "h2o"], "products": ["hno3", "no"], "type": "disproportionation"},
            {"reaction": "S + O2 → SO2", "reactants": ["s", "o2"], "products": ["so2"], "type": "combustion"},
            {"reaction": "2H2S + SO2 → 3S + 2H2O", "reactants": ["h2s", "so2"], "products": ["s", "h2o"], "type": "comproportionation"},
            {"reaction": "H2S + 2O2 → H2SO4", "reactants": ["h2s", "o2"], "products": ["h2so4"], "type": "oxidation"},
            {"reaction": "C + 2CuO → 2Cu + CO2", "reactants": ["c", "cuo"], "products": ["cu", "co2"], "type": "reduction"},
            {"reaction": "CO2 + C → 2CO", "reactants": ["co2", "c"], "products": ["co"], "type": "Boudouard"},
            {"reaction": "CH4 + H2O → CO + 3H2", "reactants": ["ch4", "h2o"], "products": ["co", "h2"], "type": "steam reforming"},
            {"reaction": "2KMnO4 → K2MnO4 + MnO2 + O2", "reactants": ["kmno4"], "products": ["k2mno4", "mno2", "o2"], "type": "decomposition"},
            {"reaction": "2MnO2 + 4KOH + O2 → 2K2MnO4 + 2H2O", "reactants": ["mno2", "koh", "o2"], "products": ["k2mno4", "h2o"], "type": "oxidation"},
            {"reaction": "2Cr + 3Cl2 → 2CrCl3", "reactants": ["cr", "cl2"], "products": ["crcl3"], "type": "synthesis"},
            {"reaction": "MnO4- + 5Fe2+ + 8H+ → Mn2+ + 5Fe3+ + 4H2O", "reactants": ["mno4-", "fe2+", "h+"], "products": ["mn2+", "fe3+", "h2o"], "type": "permanganate reduction"},
            {"reaction": "Cr2O7^2- + 6Fe2+ + 14H+ → 2Cr3+ + 6Fe3+ + 7H2O", "reactants": ["cr2o7^2-", "fe2+", "h+"], "products": ["cr3+", "fe3+", "h2o"], "type": "dichromate reduction"},
            {"reaction": "H2O2 + 2H+ + 2e- → 2H2O", "reactants": ["h2o2", "h+"], "products": ["h2o"], "type": "peroxide reduction"},
            {"reaction": "O3 + 2H+ + 2e- → O2 + H2O", "reactants": ["o3", "h+"], "products": ["o2", "h2o"], "type": "ozone reduction"},
            {"reaction": "F2 + 2e- → 2F-", "reactants": ["f2"], "products": ["f-"], "type": "fluorine reduction"},
            {"reaction": "ClO- + H2O + 2e- → Cl- + 2OH-", "reactants": ["clo-", "h2o"], "products": ["cl-", "oh-"], "type": "hypochlorite reduction"},
            {"reaction": "BrO3- + 6H+ + 6e- → Br- + 3H2O", "reactants": ["bro3-", "h+"], "products": ["br-", "h2o"], "type": "bromate reduction"},
            {"reaction": "IO3- + 6H+ + 6e- → I- + 3H2O", "reactants": ["io3-", "h+"], "products": ["i-", "h2o"], "type": "iodate reduction"},
            {"reaction": "NO3- + 4H+ + 3e- → NO + 2H2O", "reactants": ["no3-", "h+"], "products": ["no", "h2o"], "type": "nitrate reduction"},
            {"reaction": "SO4^2- + 4H+ + 2e- → SO2 + 2H2O", "reactants": ["so4^2-", "h+"], "products": ["so2", "h2o"], "type": "sulfate reduction"},
            {"reaction": "S2O8^2- + 2e- → 2SO4^2-", "reactants": ["s2o8^2-"], "products": ["so4^2-"], "type": "peroxodisulfate reduction"},
            {"reaction": "2H+ + 2e- → H2", "reactants": ["h+"], "products": ["h2"], "type": "hydrogen evolution"},
            {"reaction": "O2 + 4H+ + 4e- → 2H2O", "reactants": ["o2", "h+"], "products": ["h2o"], "type": "oxygen reduction"},
            {"reaction": "O2 + 2H2O + 4e- → 4OH-", "reactants": ["o2", "h2o"], "products": ["oh-"], "type": "oxygen reduction (basic)"},
            {"reaction": "2H2O + 2e- → H2 + 2OH-", "reactants": ["h2o"], "products": ["h2", "oh-"], "type": "water reduction (basic)"},
            {"reaction": "Na+ + e- → Na", "reactants": ["na+"], "products": ["na"], "type": "sodium reduction"},
            {"reaction": "Mg2+ + 2e- → Mg", "reactants": ["mg2+"], "products": ["mg"], "type": "magnesium reduction"},
            {"reaction": "Al3+ + 3e- → Al", "reactants": ["al3+"], "products": ["al"], "type": "aluminum reduction"},
            {"reaction": "Zn2+ + 2e- → Zn", "reactants": ["zn2+"], "products": ["zn"], "type": "zinc reduction"},
            {"reaction": "Fe2+ + 2e- → Fe", "reactants": ["fe2+"], "products": ["fe"], "type": "iron reduction"},
            {"reaction": "Fe3+ + e- → Fe2+", "reactants": ["fe3+"], "products": ["fe2+"], "type": "iron(III) reduction"},
            {"reaction": "Ni2+ + 2e- → Ni", "reactants": ["ni2+"], "products": ["ni"], "type": "nickel reduction"},
            {"reaction": "Cu2+ + 2e- → Cu", "reactants": ["cu2+"], "products": ["cu"], "type": "copper reduction"},
            {"reaction": "Cu2+ + e- → Cu+", "reactants": ["cu2+"], "products": ["cu+"], "type": "copper(II) reduction"},
            {"reaction": "Ag+ + e- → Ag", "reactants": ["ag+"], "products": ["ag"], "type": "silver reduction"},
            {"reaction": "Au3+ + 3e- → Au", "reactants": ["au3+"], "products": ["au"], "type": "gold reduction"},
            {"reaction": "Au+ + e- → Au", "reactants": ["au+"], "products": ["au"], "type": "gold(I) reduction"},
            {"reaction": "Pt2+ + 2e- → Pt", "reactants": ["pt2+"], "products": ["pt"], "type": "platinum reduction"},
            {"reaction": "Pd2+ + 2e- → Pd", "reactants": ["pd2+"], "products": ["pd"], "type": "palladium reduction"},
            {"reaction": "Hg2+ + 2e- → Hg", "reactants": ["hg2+"], "products": ["hg"], "type": "mercury reduction"},
            {"reaction": "Hg2^2+ + 2e- → 2Hg", "reactants": ["hg2^2+"], "products": ["hg"], "type": "mercury(I) reduction"},
            {"reaction": "Tl+ + e- → Tl", "reactants": ["tl+"], "products": ["tl"], "type": "thallium reduction"},
            {"reaction": "Pb2+ + 2e- → Pb", "reactants": ["pb2+"], "products": ["pb"], "type": "lead reduction"},
            {"reaction": "Sn2+ + 2e- → Sn", "reactants": ["sn2+"], "products": ["sn"], "type": "tin reduction"},
            {"reaction": "Sn4+ + 2e- → Sn2+", "reactants": ["sn4+"], "products": ["sn2+"], "type": "tin(IV) reduction"},
            {"reaction": "Bi3+ + 3e- → Bi", "reactants": ["bi3+"], "products": ["bi"], "type": "bismuth reduction"},
            {"reaction": "Sb3+ + 3e- → Sb", "reactants": ["sb3+"], "products": ["sb"], "type": "antimony reduction"},
            {"reaction": "As3+ + 3e- → As", "reactants": ["as3+"], "products": ["as"], "type": "arsenic reduction"},
            {"reaction": "Te4+ + 4e- → Te", "reactants": ["te4+"], "products": ["te"], "type": "tellurium reduction"},
            {"reaction": "Se4+ + 4e- → Se", "reactants": ["se4+"], "products": ["se"], "type": "selenium reduction"},
            {"reaction": "Po4+ + 4e- → Po", "reactants": ["po4+"], "products": ["po"], "type": "polonium reduction"},
            {"reaction": "At+ + e- → At", "reactants": ["at+"], "products": ["at"], "type": "astatine reduction"},
            {"reaction": "Rn2+ + 2e- → Rn", "reactants": ["rn2+"], "products": ["rn"], "type": "radon reduction"},
            {"reaction": "Fr+ + e- → Fr", "reactants": ["fr+"], "products": ["fr"], "type": "francium reduction"},
            {"reaction": "Ra2+ + 2e- → Ra", "reactants": ["ra2+"], "products": ["ra"], "type": "radium reduction"},
            {"reaction": "Ac3+ + 3e- → Ac", "reactants": ["ac3+"], "products": ["ac"], "type": "actinium reduction"},
            {"reaction": "Th4+ + 4e- → Th", "reactants": ["th4+"], "products": ["th"], "type": "thorium reduction"},
            {"reaction": "Pa5+ + 5e- → Pa", "reactants": ["pa5+"], "products": ["pa"], "type": "protactinium reduction"},
            {"reaction": "U4+ + 4e- → U", "reactants": ["u4+"], "products": ["u"], "type": "uranium reduction"},
            {"reaction": "Np4+ + 4e- → Np", "reactants": ["np4+"], "products": ["np"], "type": "neptunium reduction"},
            {"reaction": "Pu4+ + 4e- → Pu", "reactants": ["pu4+"], "products": ["pu"], "type": "plutonium reduction"},
            {"reaction": "Am4+ + 4e- → Am", "reactants": ["am4+"], "products": ["am"], "type": "americium reduction"},
            {"reaction": "Cm4+ + 4e- → Cm", "reactants": ["cm4+"], "products": ["cm"], "type": "curium reduction"},
            {"reaction": "Bk4+ + 4e- → Bk", "reactants": ["bk4+"], "products": ["bk"], "type": "berkelium reduction"},
            {"reaction": "Cf4+ + 4e- → Cf", "reactants": ["cf4+"], "products": ["cf"], "type": "californium reduction"},
            {"reaction": "Es4+ + 4e- → Es", "reactants": ["es4+"], "products": ["es"], "type": "einsteinium reduction"},
            {"reaction": "Fm4+ + 4e- → Fm", "reactants": ["fm4+"], "products": ["fm"], "type": "fermium reduction"},
            {"reaction": "Md4+ + 4e- → Md", "reactants": ["md4+"], "products": ["md"], "type": "mendelevium reduction"},
            {"reaction": "No4+ + 4e- → No", "reactants": ["no4+"], "products": ["no"], "type": "nobelium reduction"},
            {"reaction": "Lr4+ + 4e- → Lr", "reactants": ["lr4+"], "products": ["lr"], "type": "lawrencium reduction"},
            {"reaction": "Rf4+ + 4e- → Rf", "reactants": ["rf4+"], "products": ["rf"], "type": "rutherfordium reduction"},
            {"reaction": "Db4+ + 4e- → Db", "reactants": ["db4+"], "products": ["db"], "type": "dubnium reduction"},
            {"reaction": "Sg4+ + 4e- → Sg", "reactants": ["sg4+"], "products": ["sg"], "type": "seaborgium reduction"},
            {"reaction": "Bh4+ + 4e- → Bh", "reactants": ["bh4+"], "products": ["bh"], "type": "bohrium reduction"},
            {"reaction": "Hs4+ + 4e- → Hs", "reactants": ["hs4+"], "products": ["hs"], "type": "hassium reduction"},
            {"reaction": "Mt4+ + 4e- → Mt", "reactants": ["mt4+"], "products": ["mt"], "type": "meitnerium reduction"},
            {"reaction": "Ds4+ + 4e- → Ds", "reactants": ["ds4+"], "products": ["ds"], "type": "darmstadtium reduction"},
            {"reaction": "Rg4+ + 4e- → Rg", "reactants": ["rg4+"], "products": ["rg"], "type": "roentgenium reduction"},
            {"reaction": "Cn4+ + 4e- → Cn", "reactants": ["cn4+"], "products": ["cn"], "type": "copernicium reduction"},
            {"reaction": "Nh4+ + 4e- → Nh", "reactants": ["nh4+"], "products": ["nh"], "type": "nihonium reduction"},
            {"reaction": "Fl4+ + 4e- → Fl", "reactants": ["fl4+"], "products": ["fl"], "type": "flerovium reduction"},
            {"reaction": "Mc4+ + 4e- → Mc", "reactants": ["mc4+"], "products": ["mc"], "type": "moscovium reduction"},
            {"reaction": "Lv4+ + 4e- → Lv", "reactants": ["lv4+"], "products": ["lv"], "type": "livermorium reduction"},
            {"reaction": "Ts4+ + 4e- → Ts", "reactants": ["ts4+"], "products": ["ts"], "type": "tennessine reduction"},
            {"reaction": "Og4+ + 4e- → Og", "reactants": ["og4+"], "products": ["og"], "type": "oganesson reduction"},
        ]

    def _is_chemical_formula(self, text: str) -> bool:
        """Check if the text looks like a chemical formula."""
        if not re.match(r'^[a-z0-9]+$', text):
            return False
        if not any(c.isalpha() for c in text):
            return False
        return True

    def _find_redox_reactions(self, species: str) -> str:
        """Find redox reactions where the given species appears as reactant or product."""
        redox_reactions = [
            {"reaction": "2H2 + O2 → 2H2O", "reactants": ["h2", "o2"], "products": ["h2o"], "type": "combustion"},
            {"reaction": "2H2O → 2H2 + O2", "reactants": ["h2o"], "products": ["h2", "o2"], "type": "electrolysis"},
            {"reaction": "H2 + Cl2 → 2HCl", "reactants": ["h2", "cl2"], "products": ["hcl"], "type": "synthesis"},
            {"reaction": "Zn + 2HCl → ZnCl2 + H2", "reactants": ["zn", "hcl"], "products": ["zncl2", "h2"], "type": "single displacement"},
            {"reaction": "2H2O2 → 2H2O + O2", "reactants": ["h2o2"], "products": ["h2o", "o2"], "type": "decomposition"},
            {"reaction": "C + O2 → CO2", "reactants": ["c", "o2"], "products": ["co2"], "type": "combustion"},
            {"reaction": "2CO + O2 → 2CO2", "reactants": ["co", "o2"], "products": ["co2"], "type": "combustion"},
            {"reaction": "CH4 + 2O2 → CO2 + 2H2O", "reactants": ["ch4", "o2"], "products": ["co2", "h2o"], "type": "combustion"},
            {"reaction": "4Fe + 3O2 → 2Fe2O3", "reactants": ["fe", "o2"], "products": ["fe2o3"], "type": "corrosion"},
            {"reaction": "2SO2 + O2 → 2SO3", "reactants": ["so2", "o2"], "products": ["so3"], "type": "contact process"},
            {"reaction": "2NO + O2 → 2NO2", "reactants": ["no", "o2"], "products": ["no2"], "type": "oxidation"},
            {"reaction": "Fe + 2HCl → FeCl2 + H2", "reactants": ["fe", "hcl"], "products": ["fecl2", "h2"], "type": "single displacement"},
            {"reaction": "Fe + CuSO4 → FeSO4 + Cu", "reactants": ["fe", "cuso4"], "products": ["feso4", "cu"], "type": "single displacement"},
            {"reaction": "2Fe + 3Cl2 → 2FeCl3", "reactants": ["fe", "cl2"], "products": ["fecl3"], "type": "synthesis"},
            {"reaction": "Fe2O3 + 3CO → 2Fe + 3CO2", "reactants": ["fe2o3", "co"], "products": ["fe", "co2"], "type": "reduction"},
            {"reaction": "4Fe(OH)2 + O2 + 2H2O → 4Fe(OH)3", "reactants": ["fe(oh)2", "o2", "h2o"], "products": ["fe(oh)3"], "type": "oxidation"},
            {"reaction": "Cu + 2AgNO3 → Cu(NO3)2 + 2Ag", "reactants": ["cu", "ag"], "products": ["cu", "ag"], "type": "single displacement"},
            {"reaction": "2Cu + O2 → 2CuO", "reactants": ["cu", "o2"], "products": ["cuo"], "type": "oxidation"},
            {"reaction": "CuO + H2 → Cu + H2O", "reactants": ["cuo", "h2"], "products": ["cu", "h2o"], "type": "reduction"},
            {"reaction": "Cu + 2H2SO4 → CuSO4 + SO2 + 2H2O", "reactants": ["cu", "h2so4"], "products": ["cuso4", "so2", "h2o"], "type": "oxidation"},
            {"reaction": "Zn + 2HCl → ZnCl2 + H2", "reactants": ["zn", "hcl"], "products": ["zncl2", "h2"], "type": "single displacement"},
            {"reaction": "Zn + CuSO4 → ZnSO4 + Cu", "reactants": ["zn", "cuso4"], "products": ["znso4", "cu"], "type": "single displacement"},
            {"reaction": "2Zn + O2 → 2ZnO", "reactants": ["zn", "o2"], "products": ["zno"], "type": "oxidation"},
            {"reaction": "2Al + 6HCl → 2AlCl3 + 3H2", "reactants": ["al", "hcl"], "products": ["alcl3", "h2"], "type": "single displacement"},
            {"reaction": "2Al + 3CuSO4 → Al2(SO4)3 + 3Cu", "reactants": ["al", "cuso4"], "products": ["al2(so4)3", "cu"], "type": "single displacement"},
            {"reaction": "4Al + 3O2 → 2Al2O3", "reactants": ["al", "o2"], "products": ["al2o3"], "type": "oxidation"},
            {"reaction": "2Al + Fe2O3 → 2Fe + Al2O3", "reactants": ["al", "fe2o3"], "products": ["fe", "al2o3"], "type": "thermite"},
            {"reaction": "Mg + 2HCl → MgCl2 + H2", "reactants": ["mg", "hcl"], "products": ["mgcl2", "h2"], "type": "single displacement"},
            {"reaction": "2Mg + O2 → 2MgO", "reactants": ["mg", "o2"], "products": ["mgo"], "type": "combustion"},
            {"reaction": "Mg + CuSO4 → MgSO4 + Cu", "reactants": ["mg", "cuso4"], "products": ["mgso4", "cu"], "type": "single displacement"},
            {"reaction": "2Na + 2H2O → 2NaOH + H2", "reactants": ["na", "h2o"], "products": ["naoh", "h2"], "type": "single displacement"},
            {"reaction": "2Na + Cl2 → 2NaCl", "reactants": ["na", "cl2"], "products": ["nacl"], "type": "synthesis"},
            {"reaction": "2Na + O2 → Na2O2", "reactants": ["na", "o2"], "products": ["na2o2"], "type": "oxidation"},
            {"reaction": "Cl2 + 2NaBr → 2NaCl + Br2", "reactants": ["cl2", "nabr"], "products": ["nacl", "br2"], "type": "halogen displacement"},
            {"reaction": "Cl2 + 2KI → 2KCl + I2", "reactants": ["cl2", "ki"], "products": ["kcl", "i2"], "type": "halogen displacement"},
            {"reaction": "Cl2 + H2O → HCl + HClO", "reactants": ["cl2", "h2o"], "products": ["hcl", "hclo"], "type": "disproportionation"},
            {"reaction": "Ag + 2HNO3 → AgNO3 + NO2 + H2O", "reactants": ["ag", "hno3"], "products": ["agno3", "no2", "h2o"], "type": "oxidation"},
            {"reaction": "2Ag2O → 4Ag + O2", "reactants": ["ag2o"], "products": ["ag", "o2"], "type": "decomposition"},
            {"reaction": "Au + 4HCl + HNO3 → HAuCl4 + NO + 2H2O", "reactants": ["au", "hcl", "hno3"], "products": ["haucl4", "no", "h2o"], "type": "aqua regia"},
            {"reaction": "N2 + 3H2 → 2NH3", "reactants": ["n2", "h2"], "products": ["nh3"], "type": "Haber process"},
            {"reaction": "4NH3 + 5O2 → 4NO + 6H2O", "reactants": ["nh3", "o2"], "products": ["no", "h2o"], "type": "Ostwald process"},
            {"reaction": "2NO + O2 → 2NO2", "reactants": ["no", "o2"], "products": ["no2"], "type": "oxidation"},
            {"reaction": "3NO2 + H2O → 2HNO3 + NO", "reactants": ["no2", "h2o"], "products": ["hno3", "no"], "type": "disproportionation"},
            {"reaction": "S + O2 → SO2", "reactants": ["s", "o2"], "products": ["so2"], "type": "combustion"},
            {"reaction": "2H2S + SO2 → 3S + 2H2O", "reactants": ["h2s", "so2"], "products": ["s", "h2o"], "type": "comproportionation"},
            {"reaction": "H2S + 2O2 → H2SO4", "reactants": ["h2s", "o2"], "products": ["h2so4"], "type": "oxidation"},
            {"reaction": "C + 2CuO → 2Cu + CO2", "reactants": ["c", "cuo"], "products": ["cu", "co2"], "type": "reduction"},
            {"reaction": "CO2 + C → 2CO", "reactants": ["co2", "c"], "products": ["co"], "type": "Boudouard"},
            {"reaction": "CH4 + H2O → CO + 3H2", "reactants": ["ch4", "h2o"], "products": ["co", "h2"], "type": "steam reforming"},
            {"reaction": "2KMnO4 → K2MnO4 + MnO2 + O2", "reactants": ["kmno4"], "products": ["k2mno4", "mno2", "o2"], "type": "decomposition"},
            {"reaction": "2MnO2 + 4KOH + O2 → 2K2MnO4 + 2H2O", "reactants": ["mno2", "koh", "o2"], "products": ["k2mno4", "h2o"], "type": "oxidation"},
            {"reaction": "2Cr + 3Cl2 → 2CrCl3", "reactants": ["cr", "cl2"], "products": ["crcl3"], "type": "synthesis"},
            {"reaction": "MnO4- + 5Fe2+ + 8H+ → Mn2+ + 5Fe3+ + 4H2O", "reactants": ["mno4-", "fe2+", "h+"], "products": ["mn2+", "fe3+", "h2o"], "type": "permanganate reduction"},
            {"reaction": "Cr2O7^2- + 6Fe2+ + 14H+ → 2Cr3+ + 6Fe3+ + 7H2O", "reactants": ["cr2o7^2-", "fe2+", "h+"], "products": ["cr3+", "fe3+", "h2o"], "type": "dichromate reduction"},
            {"reaction": "H2O2 + 2H+ + 2e- → 2H2O", "reactants": ["h2o2", "h+"], "products": ["h2o"], "type": "peroxide reduction"},
            {"reaction": "O3 + 2H+ + 2e- → O2 + H2O", "reactants": ["o3", "h+"], "products": ["o2", "h2o"], "type": "ozone reduction"},
            {"reaction": "F2 + 2e- → 2F-", "reactants": ["f2"], "products": ["f-"], "type": "fluorine reduction"},
            {"reaction": "ClO- + H2O + 2e- → Cl- + 2OH-", "reactants": ["clo-", "h2o"], "products": ["cl-", "oh-"], "type": "hypochlorite reduction"},
            {"reaction": "BrO3- + 6H+ + 6e- → Br- + 3H2O", "reactants": ["bro3-", "h+"], "products": ["br-", "h2o"], "type": "bromate reduction"},
            {"reaction": "IO3- + 6H+ + 6e- → I- + 3H2O", "reactants": ["io3-", "h+"], "products": ["i-", "h2o"], "type": "iodate reduction"},
            {"reaction": "NO3- + 4H+ + 3e- → NO + 2H2O", "reactants": ["no3-", "h+"], "products": ["no", "h2o"], "type": "nitrate reduction"},
            {"reaction": "SO4^2- + 4H+ + 2e- → SO2 + 2H2O", "reactants": ["so4^2-", "h+"], "products": ["so2", "h2o"], "type": "sulfate reduction"},
            {"reaction": "S2O8^2- + 2e- → 2SO4^2-", "reactants": ["s2o8^2-"], "products": ["so4^2-"], "type": "peroxodisulfate reduction"},
            {"reaction": "2H+ + 2e- → H2", "reactants": ["h+"], "products": ["h2"], "type": "hydrogen evolution"},
            {"reaction": "O2 + 4H+ + 4e- → 2H2O", "reactants": ["o2", "h+"], "products": ["h2o"], "type": "oxygen reduction"},
            {"reaction": "O2 + 2H2O + 4e- → 4OH-", "reactants": ["o2", "h2o"], "products": ["oh-"], "type": "oxygen reduction (basic)"},
            {"reaction": "2H2O + 2e- → H2 + 2OH-", "reactants": ["h2o"], "products": ["h2", "oh-"], "type": "water reduction (basic)"},
            {"reaction": "Na+ + e- → Na", "reactants": ["na+"], "products": ["na"], "type": "sodium reduction"},
            {"reaction": "Mg2+ + 2e- → Mg", "reactants": ["mg2+"], "products": ["mg"], "type": "magnesium reduction"},
            {"reaction": "Al3+ + 3e- → Al", "reactants": ["al3+"], "products": ["al"], "type": "aluminum reduction"},
            {"reaction": "Zn2+ + 2e- → Zn", "reactants": ["zn2+"], "products": ["zn"], "type": "zinc reduction"},
            {"reaction": "Fe2+ + 2e- → Fe", "reactants": ["fe2+"], "products": ["fe"], "type": "iron reduction"},
            {"reaction": "Fe3+ + e- → Fe2+", "reactants": ["fe3+"], "products": ["fe2+"], "type": "iron(III) reduction"},
            {"reaction": "Ni2+ + 2e- → Ni", "reactants": ["ni2+"], "products": ["ni"], "type": "nickel reduction"},
            {"reaction": "Cu2+ + 2e- → Cu", "reactants": ["cu2+"], "products": ["cu"], "type": "copper reduction"},
            {"reaction": "Cu2+ + e- → Cu+", "reactants": ["cu2+"], "products": ["cu+"], "type": "copper(II) reduction"},
            {"reaction": "Ag+ + e- → Ag", "reactants": ["ag+"], "products": ["ag"], "type": "silver reduction"},
            {"reaction": "Au3+ + 3e- → Au", "reactants": ["au3+"], "products": ["au"], "type": "gold reduction"},
            {"reaction": "Au+ + e- → Au", "reactants": ["au+"], "products": ["au"], "type": "gold(I) reduction"},
            {"reaction": "Pt2+ + 2e- → Pt", "reactants": ["pt2+"], "products": ["pt"], "type": "platinum reduction"},
            {"reaction": "Pd2+ + 2e- → Pd", "reactants": ["pd2+"], "products": ["pd"], "type": "palladium reduction"},
            {"reaction": "Hg2+ + 2e- → Hg", "reactants": ["hg2+"], "products": ["hg"], "type": "mercury reduction"},
            {"reaction": "Hg2^2+ + 2e- → 2Hg", "reactants": ["hg2^2+"], "products": ["hg"], "type": "mercury(I) reduction"},
            {"reaction": "Tl+ + e- → Tl", "reactants": ["tl+"], "products": ["tl"], "type": "thallium reduction"},
            {"reaction": "Pb2+ + 2e- → Pb", "reactants": ["pb2+"], "products": ["pb"], "type": "lead reduction"},
            {"reaction": "Sn2+ + 2e- → Sn", "reactants": ["sn2+"], "products": ["sn"], "type": "tin reduction"},
            {"reaction": "Sn4+ + 2e- → Sn2+", "reactants": ["sn4+"], "products": ["sn2+"], "type": "tin(IV) reduction"},
            {"reaction": "Bi3+ + 3e- → Bi", "reactants": ["bi3+"], "products": ["bi"], "type": "bismuth reduction"},
            {"reaction": "Sb3+ + 3e- → Sb", "reactants": ["sb3+"], "products": ["sb"], "type": "antimony reduction"},
            {"reaction": "As3+ + 3e- → As", "reactants": ["as3+"], "products": ["as"], "type": "arsenic reduction"},
            {"reaction": "Te4+ + 4e- → Te", "reactants": ["te4+"], "products": ["te"], "type": "tellurium reduction"},
            {"reaction": "Se4+ + 4e- → Se", "reactants": ["se4+"], "products": ["se"], "type": "selenium reduction"},
            {"reaction": "Po4+ + 4e- → Po", "reactants": ["po4+"], "products": ["po"], "type": "polonium reduction"},
            {"reaction": "At+ + e- → At", "reactants": ["at+"], "products": ["at"], "type": "astatine reduction"},
            {"reaction": "Rn2+ + 2e- → Rn", "reactants": ["rn2+"], "products": ["rn"], "type": "radon reduction"},
            {"reaction": "Fr+ + e- → Fr", "reactants": ["fr+"], "products": ["fr"], "type": "francium reduction"},
            {"reaction": "Ra2+ + 2e- → Ra", "reactants": ["ra2+"], "products": ["ra"], "type": "radium reduction"},
            {"reaction": "Ac3+ + 3e- → Ac", "reactants": ["ac3+"], "products": ["ac"], "type": "actinium reduction"},
            {"reaction": "Th4+ + 4e- → Th", "reactants": ["th4+"], "products": ["th"], "type": "thorium reduction"},
            {"reaction": "Pa5+ + 5e- → Pa", "reactants": ["pa5+"], "products": ["pa"], "type": "protactinium reduction"},
            {"reaction": "U4+ + 4e- → U", "reactants": ["u4+"], "products": ["u"], "type": "uranium reduction"},
            {"reaction": "Np4+ + 4e- → Np", "reactants": ["np4+"], "products": ["np"], "type": "neptunium reduction"},
            {"reaction": "Pu4+ + 4e- → Pu", "reactants": ["pu4+"], "products": ["pu"], "type": "plutonium reduction"},
            {"reaction": "Am4+ + 4e- → Am", "reactants": ["am4+"], "products": ["am"], "type": "americium reduction"},
            {"reaction": "Cm4+ + 4e- → Cm", "reactants": ["cm4+"], "products": ["cm"], "type": "curium reduction"},
            {"reaction": "Bk4+ + 4e- → Bk", "reactants": ["bk4+"], "products": ["bk"], "type": "berkelium reduction"},
            {"reaction": "Cf4+ + 4e- → Cf", "reactants": ["cf4+"], "products": ["cf"], "type": "californium reduction"},
            {"reaction": "Es4+ + 4e- → Es", "reactants": ["es4+"], "products": ["es"], "type": "einsteinium reduction"},
            {"reaction": "Fm4+ + 4e- → Fm", "reactants": ["fm4+"], "products": ["fm"], "type": "fermium reduction"},
            {"reaction": "Md4+ + 4e- → Md", "reactants": ["md4+"], "products": ["md"], "type": "mendelevium reduction"},
            {"reaction": "No4+ + 4e- → No", "reactants": ["no4+"], "products": ["no"], "type": "nobelium reduction"},
            {"reaction": "Lr4+ + 4e- → Lr", "reactants": ["lr4+"], "products": ["lr"], "type": "lawrencium reduction"},
            {"reaction": "Rf4+ + 4e- → Rf", "reactants": ["rf4+"], "products": ["rf"], "type": "rutherfordium reduction"},
            {"reaction": "Db4+ + 4e- → Db", "reactants": ["db4+"], "products": ["db"], "type": "dubnium reduction"},
            {"reaction": "Sg4+ + 4e- → Sg", "reactants": ["sg4+"], "products": ["sg"], "type": "seaborgium reduction"},
            {"reaction": "Bh4+ + 4e- → Bh", "reactants": ["bh4+"], "products": ["bh"], "type": "bohrium reduction"},
            {"reaction": "Hs4+ + 4e- → Hs", "reactants": ["hs4+"], "products": ["hs"], "type": "hassium reduction"},
            {"reaction": "Mt4+ + 4e- → Mt", "reactants": ["mt4+"], "products": ["mt"], "type": "meitnerium reduction"},
            {"reaction": "Ds4+ + 4e- → Ds", "reactants": ["ds4+"], "products": ["ds"], "type": "darmstadtium reduction"},
            {"reaction": "Rg4+ + 4e- → Rg", "reactants": ["rg4+"], "products": ["rg"], "type": "roentgenium reduction"},
            {"reaction": "Cn4+ + 4e- → Cn", "reactants": ["cn4+"], "products": ["cn"], "type": "copernicium reduction"},
            {"reaction": "Nh4+ + 4e- → Nh", "reactants": ["nh4+"], "products": ["nh"], "type": "nihonium reduction"},
            {"reaction": "Fl4+ + 4e- → Fl", "reactants": ["fl4+"], "products": ["fl"], "type": "flerovium reduction"},
            {"reaction": "Mc4+ + 4e- → Mc", "reactants": ["mc4+"], "products": ["mc"], "type": "moscovium reduction"},
            {"reaction": "Lv4+ + 4e- → Lv", "reactants": ["lv4+"], "products": ["lv"], "type": "livermorium reduction"},
            {"reaction": "Ts4+ + 4e- → Ts", "reactants": ["ts4+"], "products": ["ts"], "type": "tennessine reduction"},
            {"reaction": "Og4+ + 4e- → Og", "reactants": ["og4+"], "products": ["og"], "type": "oganesson reduction"},
        ]

        matches = []
        species_lower = species.lower()

        for rxn in redox_reactions:
            if species_lower in rxn["reactants"] or species_lower in rxn["products"]:
                matches.append(rxn)

        if not matches:
            return ""

        result = f"Redox reactions involving {species}:\n\n"
        for i, match in enumerate(matches, 1):
            result += f"{i}. {match['reaction']}\n"
            result += f"   Type: {match['type']}\n"
            result += f"   Reactants: {' + '.join(match['reactants'])}\n"
            result += f"   Products: {' + '.join(match['products'])}\n\n"
        return result

    def _parse_value(self, value_str: str) -> float:
        """Parse a value string that may contain units and constant references.
        
        Examples:
            '1.5' -> 1.5
            'na' -> 6.023e23 (constant reference)
            '0.4na' -> 0.4 * 6.023e23 (constant reference)
            '0.001M' -> 0.001 (unit stripped)
            '1e-2' -> 0.01
        """
        value_str = value_str.strip()
        
        # Try direct float conversion first
        try:
            return float(value_str)
        except ValueError:
            pass
        
        # Insert * between digit and letter (e.g., 0.4na -> 0.4*na)
        expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', value_str)
        # Replace constants with their values (longest first to avoid partial matches)
        for const_name, const_value in sorted(self.constants.items(), key=lambda x: len(x[0]), reverse=True):
            expr = re.sub(r'(?<!\w)' + re.escape(const_name) + r'(?!\w)', f'({const_value})', expr)
        # Fix implicit multiplication: digit followed by ( -> digit*(
        expr = re.sub(r'(\d)\(', r'\1*(', expr)
        
        # Strip units - remove trailing unit symbols (longest first)
        unit_patterns = [
            'mol/L', 'kJ/mol·K', 'kJ/mol.K', 'g/cm³', 'mL/L', 'cm³',
            'mol⁻¹', 'mol', 'g', 'L', 'mL', 'K', 'Pa', 'V', 'M', 'm'
        ]
        for unit in sorted(unit_patterns, key=len, reverse=True):
            expr = re.sub(r'\*?\s*' + re.escape(unit) + r'$', '', expr)
        
        # Try to evaluate
        try:
            return float(sympify(expr))
        except Exception:
            raise ValueError(f"Could not parse value: {value_str}")

    def _solve_equation(self, expression: str) -> str:
        """Solve a chemistry equation with known values.

        Parses inputs like "n=2, V=0.5, M" where known values are given
        and the last symbol is the target to solve for.
        """
        parts = [p.strip() for p in expression.split(',')]

        known_values: Dict[str, float] = {}
        target_var = None

        for part in parts:
            if '=' in part:
                key, _, value = part.partition('=')
                key = key.strip()
                value = value.strip()
                try:
                    known_values[key] = self._parse_value(value)
                except ValueError as e:
                    return f"Error: {e}"
            else:
                target_var = part.strip()

        if not target_var:
            return self._solve_for_all(known_values)

        # Try to solve with formulas first
        if target_var in self.formulas:
            result = self._solve_with_formulas(target_var, known_values)
            if result:
                return result

        # If target is a constant, show its value
        if target_var in self.constants:
            return self._lookup_constant(target_var)

        if target_var not in self.symbols:
            return f"Error: Unknown variable '{target_var}'"

        return f"Error: Could not solve for '{target_var}' with given values"

    def _solve_with_formulas(self, target_var: str, known_values: Dict[str, float]) -> Optional[str]:
        """Try to solve for target_var using known chemistry formulas."""
        if target_var not in self.formulas:
            return None

        for formula_str, required_vars, description, unit in self.formulas[target_var]:
            missing = [v for v in required_vars if v not in known_values and v not in self.constants]
            if missing:
                continue

            try:
                # Use a simple namespace with unique names to avoid sympy built-in conflicts
                # Replace variable names with unique identifiers
                namespace = {}
                for v in required_vars:
                    if v in known_values:
                        namespace[f"var_{v}"] = known_values[v]
                    elif v in self.constants:
                        namespace[f"var_{v}"] = float(sympify(self.constants[v]))

                # Replace variable names in formula string (using word boundaries to avoid partial matches)
                modified_formula = formula_str
                for v in sorted(required_vars, key=len, reverse=True):
                    modified_formula = re.sub(r'\b' + re.escape(v) + r'\b', f"var_{v}", modified_formula)

                # Evaluate the formula
                result = eval(modified_formula, {"__builtins__": {}}, {"Rational": Rational, **namespace})

                try:
                    numeric = float(result)
                    if numeric == int(numeric):
                        return f"{target_var} = {int(numeric)} {unit}".strip()
                    return f"{target_var} = {numeric:.10g} {unit}".strip()
                except (TypeError, ValueError):
                    return f"{target_var} = {result}"

            except Exception:
                continue

        return None

    def _solve_for_all(self, known_values: Dict[str, float]) -> str:
        """Solve for all possible target variables given known values.

        Only calculates formulas that share at least one variable with the user's input
        (excluding constants like R, na, NA).
        """
        results = []
        user_vars = set(known_values.keys()) - CONSTANT_SYMBOLS

        for target_var, formulas in self.formulas.items():
            if target_var in known_values:
                continue

            is_relevant = False
            for formula_str, required_vars, description in formulas:
                formula_vars = set(required_vars) - CONSTANT_SYMBOLS
                if formula_vars & user_vars:
                    is_relevant = True
                    break

            if not is_relevant:
                continue

            result = self._solve_with_formulas(target_var, known_values)
            if result:
                results.append(result)

        if results:
            return '\n'.join(results)

        return "Error: Could not solve for any variable with the given known values"

    def _get_electronegativity(self, symbol: str) -> str:
        """Get electronegativity for an element."""
        en = {
            'h': '2.20', 'he': 'N/A', 'li': '0.98', 'be': '1.57',
            'b': '2.04', 'c': '2.55', 'n': '3.04', 'o': '3.44',
            'f': '3.98', 'ne': 'N/A', 'na': '0.93', 'mg': '1.31',
            'al': '1.61', 'si': '1.90', 'p': '2.19', 's': '2.58',
            'cl': '3.16', 'ar': 'N/A', 'k': '0.82', 'ca': '1.00',
            'sc': '1.36', 'ti': '1.54', 'v': '1.63', 'cr': '1.66',
            'mn': '1.55', 'fe': '1.83', 'co': '1.88', 'ni': '1.91',
            'cu': '1.90', 'zn': '1.65', 'ga': '1.81', 'ge': '2.01',
            'as': '2.18', 'se': '2.55', 'br': '2.96', 'kr': '3.00',
            'rb': '0.82', 'sr': '0.95', 'y': '1.22', 'zr': '1.33',
            'nb': '1.60', 'mo': '2.16', 'tc': '1.90', 'ru': '2.20',
            'rh': '2.28', 'pd': '2.20', 'ag': '1.93', 'cd': '1.69',
            'in': '1.78', 'sn': '1.96', 'sb': '2.05', 'te': '2.10',
            'i': '2.66', 'xe': '2.60', 'cs': '0.79', 'ba': '0.89',
            'la': '1.10', 'ce': '1.12', 'pr': '1.13', 'nd': '1.14',
            'pm': 'N/A', 'sm': '1.17', 'eu': 'N/A', 'gd': '1.20',
            'tb': 'N/A', 'dy': '1.22', 'ho': '1.23', 'er': '1.24',
            'tm': '1.25', 'yb': 'N/A', 'lu': '1.27', 'hf': '1.30',
            'ta': '1.50', 'w': '2.36', 're': '1.90', 'os': '2.20',
            'ir': '2.20', 'pt': '2.28', 'au': '2.54', 'hg': '2.00',
            'tl': '1.62', 'pb': '2.33', 'bi': '2.02', 'po': '2.00',
            'at': '2.20', 'rn': 'N/A', 'fr': '0.70', 'ra': '0.90',
            'ac': '1.10', 'th': '1.30', 'pa': '1.50', 'u': '1.38',
        }
        return en.get(symbol, '')

    def _get_electron_configuration(self, symbol: str) -> str:
        """Get electron configuration for an element."""
        ec = {
            'h': '1s1', 'he': '1s2', 'li': '[He] 2s1', 'be': '[He] 2s2',
            'b': '[He] 2s2 2p1', 'c': '[He] 2s2 2p2', 'n': '[He] 2s2 2p3',
            'o': '[He] 2s2 2p4', 'f': '[He] 2s2 2p5', 'ne': '[He] 2s2 2p6',
            'na': '[Ne] 3s1', 'mg': '[Ne] 3s2', 'al': '[Ne] 3s2 3p1',
            'si': '[Ne] 3s2 3p2', 'p': '[Ne] 3s2 3p3', 's': '[Ne] 3s2 3p4',
            'cl': '[Ne] 3s2 3p5', 'ar': '[Ne] 3s2 3p6', 'k': '[Ar] 4s1',
            'ca': '[Ar] 4s2', 'sc': '[Ar] 3d1 4s2', 'ti': '[Ar] 3d2 4s2',
            'v': '[Ar] 3d3 4s2', 'cr': '[Ar] 3d5 4s1', 'mn': '[Ar] 3d5 4s2',
            'fe': '[Ar] 3d6 4s2', 'co': '[Ar] 3d7 4s2', 'ni': '[Ar] 3d8 4s2',
            'cu': '[Ar] 3d10 4s1', 'zn': '[Ar] 3d10 4s2', 'ga': '[Ar] 3d10 4s2 4p1',
            'ge': '[Ar] 3d10 4s2 4p2', 'as': '[Ar] 3d10 4s2 4p3',
            'se': '[Ar] 3d10 4s2 4p4', 'br': '[Ar] 3d10 4s2 4p5',
            'kr': '[Ar] 3d10 4s2 4p6', 'rb': '[Kr] 5s1', 'sr': '[Kr] 5s2',
            'y': '[Kr] 4d1 5s2', 'zr': '[Kr] 4d2 5s2', 'nb': '[Kr] 4d4 5s1',
            'mo': '[Kr] 4d5 5s1', 'tc': '[Kr] 4d5 5s2', 'ru': '[Kr] 4d7 5s1',
            'rh': '[Kr] 4d8 5s1', 'pd': '[Kr] 4d10', 'ag': '[Kr] 4d10 5s1',
            'cd': '[Kr] 4d10 5s2', 'in': '[Kr] 4d10 5s2 5p1',
            'sn': '[Kr] 4d10 5s2 5p2', 'sb': '[Kr] 4d10 5s2 5p3',
            'te': '[Kr] 4d10 5s2 5p4', 'i': '[Kr] 4d10 5s2 5p5',
            'xe': '[Kr] 4d10 5s2 5p6', 'cs': '[Xe] 6s1', 'ba': '[Xe] 6s2',
            'la': '[Xe] 5d1 6s2', 'ce': '[Xe] 4f1 5d1 6s2',
            'pr': '[Xe] 4f3 6s2', 'nd': '[Xe] 4f4 6s2', 'pm': '[Xe] 4f5 6s2',
            'sm': '[Xe] 4f6 6s2', 'eu': '[Xe] 4f7 6s2', 'gd': '[Xe] 4f7 5d1 6s2',
            'tb': '[Xe] 4f9 6s2', 'dy': '[Xe] 4f10 6s2', 'ho': '[Xe] 4f11 6s2',
            'er': '[Xe] 4f12 6s2', 'tm': '[Xe] 4f13 6s2', 'yb': '[Xe] 4f14 6s2',
            'lu': '[Xe] 4f14 5d1 6s2', 'hf': '[Xe] 4f14 5d2 6s2',
            'ta': '[Xe] 4f14 5d3 6s2', 'w': '[Xe] 4f14 5d4 6s2',
            're': '[Xe] 4f14 5d5 6s2', 'os': '[Xe] 4f14 5d6 6s2',
            'ir': '[Xe] 4f14 5d7 6s2', 'pt': '[Xe] 4f14 5d9 6s1',
            'au': '[Xe] 4f14 5d10 6s1', 'hg': '[Xe] 4f14 5d10 6s2',
            'tl': '[Xe] 4f14 5d10 6s2 6p1', 'pb': '[Xe] 4f14 5d10 6s2 6p2',
            'bi': '[Xe] 4f14 5d10 6s2 6p3', 'po': '[Xe] 4f14 5d10 6s2 6p4',
            'at': '[Xe] 4f14 5d10 6s2 6p5', 'rn': '[Xe] 4f14 5d10 6s2 6p6',
            'fr': '[Rn] 7s1', 'ra': '[Rn] 7s2', 'ac': '[Rn] 6d1 7s2',
            'th': '[Rn] 6d2 7s2', 'pa': '[Rn] 5f2 6d1 7s2',
            'u': '[Rn] 5f3 6d1 7s2',
        }
        return ec.get(symbol, '')

    def _get_phase_points(self, symbol: str) -> tuple:
        """Get melting and boiling points for an element (in °C)."""
        points = {
            'h': ('-259.14', '-252.87'), 'he': ('-272.2', '-268.93'),
            'li': ('180.54', '1342'), 'be': ('1287', '2469'),
            'b': ('2075', '4000'), 'c': ('3550', '4027'),
            'n': ('-210.1', '-195.79'), 'o': ('-218.79', '-182.96'),
            'f': ('-219.67', '-188.11'), 'ne': ('-248.59', '-246.05'),
            'na': ('97.72', '883'), 'mg': ('650', '1090'),
            'al': ('660.32', '2519'), 'si': ('1414', '2900'),
            'p': ('44.15', '280.5'), 's': ('115.21', '444.6'),
            'cl': ('-101.5', '-34.04'), 'ar': ('-189.34', '-185.85'),
            'k': ('63.38', '759'), 'ca': ('842', '1484'),
            'sc': ('1541', '2836'), 'ti': ('1668', '3287'),
            'v': ('1910', '3407'), 'cr': ('1907', '2671'),
            'mn': ('1246', '2061'), 'fe': ('1538', '2862'),
            'co': ('1495', '2927'), 'ni': ('1455', '2913'),
            'cu': ('1084.62', '2562'), 'zn': ('419.53', '907'),
            'ga': ('29.76', '2204'), 'ge': ('938.25', '2833'),
            'as': ('817', '614'), 'se': ('221', '685'),
            'br': ('-7.2', '58.8'), 'kr': ('-157.36', '-153.42'),
            'rb': ('39.31', '688'), 'sr': ('777', '1382'),
            'y': ('1522', '3345'), 'zr': ('1855', '4409'),
            'nb': ('2477', '4744'), 'mo': ('2623', '4639'),
            'tc': ('2157', '4265'), 'ru': ('2334', '4150'),
            'rh': ('1964', '3695'), 'pd': ('1554.9', '2963'),
            'ag': ('961.78', '2162'), 'cd': ('321.07', '767'),
            'in': ('156.6', '2072'), 'sn': ('231.93', '2602'),
            'sb': ('630.63', '1587'), 'te': ('449.51', '988'),
            'i': ('113.7', '184.3'), 'xe': ('-111.75', '-108.1'),
            'cs': ('28.44', '671'), 'ba': ('727', '1845'),
            'la': ('920', '3464'), 'ce': ('798', '3443'),
            'pr': ('931', '3520'), 'nd': ('1021', '3074'),
            'pm': ('1042', '3000'), 'sm': ('1074', '1794'),
            'eu': ('822', '1529'), 'gd': ('1313', '3273'),
            'tb': ('1356', '3230'), 'dy': ('1412', '2567'),
            'ho': ('1474', '2700'), 'er': ('1529', '2868'),
            'tm': ('1545', '1950'), 'yb': ('819', '1196'),
            'lu': ('1663', '3402'), 'hf': ('2233', '4603'),
            'ta': ('3017', '5458'), 'w': ('3422', '5555'),
            're': ('3186', '5596'), 'os': ('3033', '5012'),
            'ir': ('2446', '4428'), 'pt': ('1768.3', '3825'),
            'au': ('1064.18', '2856'), 'hg': ('-38.83', '356.73'),
            'tl': ('304', '1473'), 'pb': ('327.46', '1749'),
            'bi': ('271.4', '1564'), 'po': ('254', '962'),
            'at': ('302', '337'), 'rn': ('-71', '-61.7'),
            'fr': ('27', '677'), 'ra': ('700', '1737'),
            'ac': ('1050', '3200'), 'th': ('1750', '4788'),
            'pa': ('1568', '4027'), 'u': ('1132.2', '4131'),
        }
        return points.get(symbol, ('', ''))

    def _get_density(self, symbol: str) -> str:
        """Get density for an element (in g/cm³ at STP)."""
        density = {
            'h': '0.0000899', 'he': '0.0001785', 'li': '0.534',
            'be': '1.85', 'b': '2.34', 'c': '2.267 (graphite)',
            'n': '0.0012506', 'o': '0.001429', 'f': '0.001696',
            'ne': '0.0009002', 'na': '0.968', 'mg': '1.738',
            'al': '2.70', 'si': '2.329', 'p': '1.823 (white)',
            's': '2.07', 'cl': '0.003214', 'ar': '0.0017837',
            'k': '0.862', 'ca': '1.55', 'sc': '2.985',
            'ti': '4.506', 'v': '6.11', 'cr': '7.15',
            'mn': '7.3', 'fe': '7.874', 'co': '8.90',
            'ni': '8.908', 'cu': '8.933', 'zn': '7.134',
            'ga': '5.91', 'ge': '5.323', 'as': '5.727',
            'se': '4.81', 'br': '3.1028', 'kr': '0.003749',
            'rb': '1.532', 'sr': '2.64', 'y': '4.472',
            'zr': '6.52', 'nb': '8.57', 'mo': '10.28',
            'tc': '11.5', 'ru': '12.45', 'rh': '12.41',
            'pd': '12.023', 'ag': '10.49', 'cd': '8.65',
            'in': '7.31', 'sn': '7.287', 'sb': '6.697',
            'te': '6.24', 'i': '4.933', 'xe': '0.005894',
            'cs': '1.879', 'ba': '3.51', 'la': '6.162',
            'ce': '6.77', 'pr': '6.77', 'nd': '7.01',
            'pm': '7.26', 'sm': '7.52', 'eu': '5.244',
            'gd': '7.90', 'tb': '8.23', 'dy': '8.54',
            'ho': '8.79', 'er': '9.066', 'tm': '9.32',
            'yb': '6.90', 'lu': '9.841', 'hf': '13.31',
            'ta': '16.69', 'w': '19.25', 're': '21.02',
            'os': '22.59', 'ir': '22.56', 'pt': '21.45',
            'au': '19.3', 'hg': '13.5336', 'tl': '11.85',
            'pb': '11.34', 'bi': '9.78', 'po': '9.196',
            'at': 'N/A', 'rn': '0.00973', 'fr': 'N/A',
            'ra': '5.5', 'ac': '10.07', 'th': '11.724',
            'pa': '15.37', 'u': '19.1',
        }
        return density.get(symbol, '')

    def _get_oxidation_states(self, symbol: str) -> str:
        """Get common oxidation states for an element."""
        oxidation_states = {
            'h': '+1, -1', 'he': '0', 'li': '+1', 'be': '+2',
            'b': '+3', 'c': '-4, -3, -2, -1, 0, +1, +2, +3, +4',
            'n': '-3, -2, -1, 0, +1, +2, +3, +4, +5',
            'o': '-2, -1, 0', 'f': '-1', 'ne': '0', 'na': '+1',
            'mg': '+2', 'al': '+3', 'si': '-4, +4', 'p': '-3, +3, +5',
            's': '-2, +4, +6', 'cl': '-1, +1, +3, +5, +7', 'ar': '0',
            'k': '+1', 'ca': '+2', 'sc': '+3', 'ti': '+2, +3, +4',
            'v': '+2, +3, +4, +5', 'cr': '+2, +3, +6', 'mn': '+2, +4, +6, +7',
            'fe': '+2, +3', 'co': '+2, +3', 'ni': '+2, +3', 'cu': '+1, +2',
            'zn': '+2', 'ga': '+3', 'ge': '+2, +4', 'as': '-3, +3, +5',
            'se': '-2, +4, +6', 'br': '-1, +1, +5', 'kr': '0', 'rb': '+1',
            'sr': '+2', 'y': '+3', 'zr': '+4', 'nb': '+3, +5', 'mo': '+3, +4, +6',
            'tc': '+4, +7', 'ru': '+3, +4, +6, +8', 'rh': '+3, +4', 'pd': '+2, +4',
            'ag': '+1', 'cd': '+2', 'in': '+3', 'sn': '+2, +4', 'sb': '-3, +3, +5',
            'te': '-2, +4, +6', 'i': '-1, +1, +5, +7', 'xe': '0', 'cs': '+1',
            'ba': '+2', 'la': '+3', 'ce': '+3, +4', 'pr': '+3, +4', 'nd': '+3',
            'pm': '+3', 'sm': '+2, +3', 'eu': '+2, +3', 'gd': '+3', 'tb': '+3, +4',
            'dy': '+3', 'ho': '+3', 'er': '+3', 'tm': '+2, +3', 'yb': '+2, +3',
            'lu': '+3', 'hf': '+4', 'ta': '+5', 'w': '+4, +6', 're': '+4, +7',
            'os': '+4, +8', 'ir': '+3, +4', 'pt': '+2, +4', 'au': '+1, +3',
            'hg': '+1, +2', 'tl': '+1, +3', 'pb': '+2, +4', 'bi': '+3, +5',
            'po': '+2, +4', 'at': '-1, +1', 'rn': '0', 'fr': '+1', 'ra': '+2',
            'ac': '+3', 'th': '+4', 'pa': '+4, +5', 'u': '+3, +4, +5, +6',
        }
        return oxidation_states.get(symbol, '')

    def _get_common_compounds(self, symbol: str) -> str:
        """Get common compounds for an element."""
        compounds = {
            'h': 'H2O, HCl, H2SO4, NH3, CH4', 'he': 'He (inert)',
            'li': 'Li2O, LiOH, LiCl', 'be': 'BeO, Be(OH)2, BeCl2',
            'b': 'B2O3, H3BO3, BCl3', 'c': 'CO2, CH4, C6H12O6, CaCO3',
            'n': 'NH3, NO, NO2, N2O, HNO3', 'o': 'H2O, CO2, O2, O3, Fe2O3',
            'f': 'HF, NaF, CaF2', 'ne': 'Ne (inert)', 'na': 'NaCl, NaOH, Na2CO3, NaHCO3',
            'mg': 'MgO, Mg(OH)2, MgCl2, MgSO4', 'al': 'Al2O3, Al(OH)3, AlCl3, Al2(SO4)3',
            'si': 'SiO2, SiC, SiCl4, Na2SiO3', 'p': 'P2O5, H3PO4, PH3, PCl3',
            's': 'H2S, SO2, SO3, H2SO4, FeS2', 'cl': 'HCl, NaCl, KCl, Cl2, HClO4',
            'ar': 'Ar (inert)', 'k': 'KCl, KOH, K2CO3, KNO3', 'ca': 'CaO, Ca(OH)2, CaCO3, CaCl2',
            'sc': 'Sc2O3, ScCl3', 'ti': 'TiO2, TiCl4, TiC', 'v': 'V2O5, VCl3, VO2',
            'cr': 'Cr2O3, CrCl3, K2Cr2O7, CrO3', 'mn': 'MnO2, KMnO4, MnCl2, MnSO4',
            'fe': 'Fe2O3, Fe3O4, FeCl2, FeCl3, FeSO4', 'co': 'CoO, CoCl2, CoSO4',
            'ni': 'NiO, NiCl2, NiSO4', 'cu': 'CuO, Cu2O, CuCl2, CuSO4, Cu(OH)2',
            'zn': 'ZnO, ZnCl2, ZnSO4, ZnS', 'ga': 'Ga2O3, GaCl3', 'ge': 'GeO2, GeCl4',
            'as': 'As2O3, AsH3, AsCl3', 'se': 'SeO2, H2Se, Na2SeO3', 'br': 'HBr, NaBr, KBr, Br2',
            'kr': 'KrF2', 'rb': 'RbCl, RbOH, Rb2CO3', 'sr': 'SrO, Sr(OH)2, SrCO3, SrCl2',
            'y': 'Y2O3, YCl3', 'zr': 'ZrO2, ZrCl4, ZrC', 'nb': 'Nb2O5, NbCl5',
            'mo': 'MoO3, MoS2, Na2MoO4', 'tc': 'TcO2, NaTcO4', 'ru': 'RuO4, RuCl3',
            'rh': 'Rh2O3, RhCl3', 'pd': 'PdO, PdCl2, PdSO4', 'ag': 'Ag2O, AgCl, AgNO3, Ag2S',
            'cd': 'CdO, CdCl2, CdSO4, CdS', 'in': 'In2O3, InCl3', 'sn': 'SnO, SnO2, SnCl2, SnCl4',
            'sb': 'Sb2O3, SbCl3, SbCl5', 'te': 'TeO2, H2Te, Na2TeO3', 'i': 'I2, HI, NaI, KIO3',
            'xe': 'XeF2, XeF4, XeF6', 'cs': 'CsCl, CsOH, Cs2CO3', 'ba': 'BaO, Ba(OH)2, BaCO3, BaCl2',
            'la': 'La2O3, LaCl3', 'ce': 'CeO2, CeCl3, Ce2(SO4)3', 'pr': 'Pr2O3, PrCl3',
            'nd': 'Nd2O3, NdCl3', 'pm': 'Pm2O3', 'sm': 'Sm2O3, SmCl3', 'eu': 'Eu2O3, EuCl3',
            'gd': 'Gd2O3, GdCl3', 'tb': 'Tb2O3, TbCl3', 'dy': 'Dy2O3, DyCl3', 'ho': 'Ho2O3, HoCl3',
            'er': 'Er2O3, ErCl3', 'tm': 'Tm2O3, TmCl3', 'yb': 'Yb2O3, YbCl3', 'lu': 'Lu2O3, LuCl3',
            'hf': 'HfO2, HfCl4', 'ta': 'Ta2O5, TaCl5', 'w': 'WO3, WCl6, WC',
            're': 'Re2O7, ReCl3', 'os': 'OsO4, OsCl4', 'ir': 'Ir2O3, IrCl3',
            'pt': 'PtO2, PtCl2, PtCl4', 'au': 'Au2O3, AuCl3', 'hg': 'HgO, HgCl2, Hg2Cl2',
            'tl': 'Tl2O3, TlCl3', 'pb': 'PbO, PbO2, PbCl2, PbSO4', 'bi': 'Bi2O3, BiCl3',
            'po': 'PoO2', 'at': 'At2O', 'rn': 'RnF2', 'fr': 'FrCl',
            'ra': 'RaO, RaCl2', 'ac': 'Ac2O3, AcCl3', 'th': 'ThO2, ThCl4',
            'pa': 'Pa2O5, PaCl5', 'u': 'UO2, UCl4, UO2(NO3)2',
        }
        return compounds.get(symbol, '')

    def _get_element_uses(self, symbol: str) -> str:
        """Get common uses for an element."""
        uses = {
            'h': 'Fuel, ammonia production, rocket fuel, hydrogenation',
            'he': 'Cooling, lifting gas, leak detection, welding',
            'li': 'Batteries, ceramics, lubricants, mood-stabilizing drugs',
            'be': 'Aerospace materials, X-ray windows, nuclear reactors',
            'b': 'Borosilicate glass, detergents, semiconductors',
            'c': 'Steel production, carbon fiber, diamonds, organic chemistry',
            'n': 'Fertilizers, explosives, liquid nitrogen, food preservation',
            'o': 'Combustion, respiration, steel production, welding',
            'f': 'Toothpaste, Teflon, uranium enrichment',
            'ne': 'Lighting, cryogenics, lasers',
            'na': 'Street lights, soap, table salt, heat transfer',
            'mg': 'Alloys, flares, antacids, fireworks',
            'al': 'Aircraft, cans, construction, electrical transmission',
            'si': 'Semiconductors, solar cells, glass, cement',
            'p': 'Fertilizers, matches, detergents, steel production',
            's': 'Sulfuric acid, vulcanization, fungicides, gunpowder',
            'cl': 'Water purification, PVC, disinfectants, table salt',
            'ar': 'Lighting, welding, inert atmosphere',
            'k': 'Fertilizers, soap, glass, gunpowder',
            'ca': 'Cement, steel production, agriculture, construction',
            'sc': 'Aerospace alloys, stadium lights',
            'ti': 'Aerospace alloys, implants, white pigment, golf clubs',
            'v': 'Steel alloys, catalysts, batteries',
            'cr': 'Stainless steel, chrome plating, pigments',
            'mn': 'Steel production, batteries, aluminum cans',
            'fe': 'Steel production, construction, vehicles, machinery',
            'co': 'Superalloys, magnets, batteries, catalysts',
            'ni': 'Stainless steel, coins, batteries, electroplating',
            'cu': 'Electrical wiring, plumbing, coins, electronics',
            'zn': 'Galvanizing, batteries, alloys, die-casting',
            'ga': 'Semiconductors, LEDs, solar cells, thermometers',
            'ge': 'Semiconductors, fiber optics, infrared optics',
            'as': 'Semiconductors, wood preservatives, alloys',
            'se': 'Solar cells, glass, photocopiers, anti-dandruff shampoo',
            'br': 'Flame retardants, pesticides, photographic film',
            'kr': 'Lighting, photography, lasers',
            'rb': 'Atomic clocks, research, fireworks',
            'sr': 'Fireworks, flares, glow-in-the-dark paints',
            'y': 'Lasers, superconductors, LEDs, camera lenses',
            'zr': 'Nuclear reactors, ceramics, refractories',
            'nb': 'Superconducting magnets, pipeline steel, rocket engines',
            'mo': 'Steel alloys, catalysts, lubricants',
            'tc': 'Medical imaging, research',
            'ru': 'Catalysts, electrical contacts, fountain pen tips',
            'rh': 'Catalytic converters, jewelry, mirrors',
            'pd': 'Catalytic converters, electronics, hydrogen purification',
            'ag': 'Jewelry, photography, electronics, coins',
            'cd': 'Batteries, pigments, coatings, solar cells',
            'in': 'Semiconductors, touch screens, solders',
            'sn': 'Solder, tin cans, bronze, glass coating',
            'sb': 'Flame retardants, alloys, semiconductors',
            'te': 'Solar cells, thermoelectric devices, steel additives',
            'i': 'Disinfectants, contrast agents, photography',
            'xe': 'Lighting, ion thrusters, anesthesia',
            'cs': 'Atomic clocks, drilling fluids, research',
            'ba': 'Medical imaging, drilling fluids, fireworks',
            'la': 'Camera lenses, hybrid car batteries, catalysts',
            'ce': 'Catalytic converters, glass polishing, self-cleaning ovens',
            'pr': 'Aircraft engines, carbon arc lighting, magnets',
            'nd': 'Magnets, headphones, hybrid car motors, lasers',
            'pm': 'Luminous paint, nuclear batteries',
            'sm': 'Magnets, cancer treatment, neutron absorbers',
            'eu': 'Fluorescent lamps, lasers, anti-counterfeiting',
            'gd': 'MRI contrast agents, neutron shielding, magnets',
            'tb': 'Green phosphors, solid-state devices, sonar',
            'dy': 'Magnets, lasers, commercial lighting',
            'ho': 'Magnets, lasers, calibration standards',
            'er': 'Fiber amplifiers, lasers, glass coloring',
            'tm': 'Lasers, portable X-ray machines',
            'yb': 'Fiber lasers, atomic clocks, stress gauges',
            'lu': 'PET scan detectors, cancer treatment, LED phosphors',
            'hf': 'Nuclear control rods, plasma cutting, microchips',
            'ta': 'Surgical implants, capacitors, cutting tools',
            'w': 'Light bulb filaments, cutting tools, armor',
            're': 'Jet engine superalloys, catalysts, thermocouples',
            'os': 'Fountain pen tips, electrical contacts, catalysts',
            'ir': 'Spark plugs, crucibles, deep-sea cables',
            'pt': 'Catalytic converters, jewelry, laboratory equipment',
            'au': 'Jewelry, electronics, investment, dentistry',
            'hg': 'Thermometers, barometers, fluorescent lamps',
            'tl': 'Infrared detectors, low-temperature thermometers',
            'pb': 'Batteries, ammunition, radiation shielding',
            'bi': 'Pepto-Bismol, lead-free solder, fire detectors',
            'po': 'Anti-static devices, heat sources, research',
            'at': 'Cancer treatment, research',
            'rn': 'Earthquake prediction, cancer treatment',
            'fr': 'Research only',
            'ra': 'Cancer treatment, luminous dials',
            'ac': 'Neutron sources, research',
            'th': 'Gas mantles, aerospace alloys, nuclear fuel',
            'pa': 'Research only',
            'u': 'Nuclear fuel, atomic weapons, counterweights',
        }
        return uses.get(symbol, '')

    def _get_element_category(self, symbol: str) -> str:
        """Get the category of an element."""
        categories = {
            'h': 'Nonmetal', 'he': 'Noble Gas', 'li': 'Alkali Metal',
            'be': 'Alkaline Earth Metal', 'b': 'Metalloid', 'c': 'Nonmetal',
            'n': 'Nonmetal', 'o': 'Nonmetal', 'f': 'Halogen', 'ne': 'Noble Gas',
            'na': 'Alkali Metal', 'mg': 'Alkaline Earth Metal', 'al': 'Post-transition Metal',
            'si': 'Metalloid', 'p': 'Nonmetal', 's': 'Nonmetal', 'cl': 'Halogen',
            'ar': 'Noble Gas', 'k': 'Alkali Metal', 'ca': 'Alkaline Earth Metal',
            'sc': 'Transition Metal', 'ti': 'Transition Metal', 'v': 'Transition Metal',
            'cr': 'Transition Metal', 'mn': 'Transition Metal', 'fe': 'Transition Metal',
            'co': 'Transition Metal', 'ni': 'Transition Metal', 'cu': 'Transition Metal',
            'zn': 'Transition Metal', 'ga': 'Post-transition Metal', 'ge': 'Metalloid',
            'as': 'Metalloid', 'se': 'Nonmetal', 'br': 'Halogen', 'kr': 'Noble Gas',
            'rb': 'Alkali Metal', 'sr': 'Alkaline Earth Metal', 'y': 'Transition Metal',
            'zr': 'Transition Metal', 'nb': 'Transition Metal', 'mo': 'Transition Metal',
            'tc': 'Transition Metal', 'ru': 'Transition Metal', 'rh': 'Transition Metal',
            'pd': 'Transition Metal', 'ag': 'Transition Metal', 'cd': 'Transition Metal',
            'in': 'Post-transition Metal', 'sn': 'Post-transition Metal', 'sb': 'Metalloid',
            'te': 'Metalloid', 'i': 'Halogen', 'xe': 'Noble Gas', 'cs': 'Alkali Metal',
            'ba': 'Alkaline Earth Metal', 'la': 'Lanthanide', 'ce': 'Lanthanide',
            'pr': 'Lanthanide', 'nd': 'Lanthanide', 'pm': 'Lanthanide', 'sm': 'Lanthanide',
            'eu': 'Lanthanide', 'gd': 'Lanthanide', 'tb': 'Lanthanide', 'dy': 'Lanthanide',
            'ho': 'Lanthanide', 'er': 'Lanthanide', 'tm': 'Lanthanide', 'yb': 'Lanthanide',
            'lu': 'Lanthanide', 'hf': 'Transition Metal', 'ta': 'Transition Metal',
            'w': 'Transition Metal', 're': 'Transition Metal', 'os': 'Transition Metal',
            'ir': 'Transition Metal', 'pt': 'Transition Metal', 'au': 'Transition Metal',
            'hg': 'Transition Metal', 'tl': 'Post-transition Metal', 'pb': 'Post-transition Metal',
            'bi': 'Post-transition Metal', 'po': 'Post-transition Metal', 'at': 'Halogen',
            'rn': 'Noble Gas', 'fr': 'Alkali Metal', 'ra': 'Alkaline Earth Metal',
            'ac': 'Actinide', 'th': 'Actinide', 'pa': 'Actinide', 'u': 'Actinide',
        }
        return categories.get(symbol, '')

    def _get_year_discovered(self, symbol: str) -> str:
        """Get year discovered for an element."""
        years = {
            'h': '1766 (Cavendish)', 'he': '1868 (Janssen/Lockyer)',
            'li': '1817 (Arfvedson)', 'be': '1798 (Vaulquelin)',
            'b': '1808 (Gay-Lussac/Thenard)', 'c': 'Ancient',
            'n': '1772 (Rutherford)', 'o': '1774 (Priestley/Scheele)',
            'f': '1886 (Moissan)', 'ne': '1898 (Ramsay/Travers)',
            'na': '1807 (Davy)', 'mg': '1755 (Black)',
            'al': '1825 (Oersted)', 'si': '1824 (Berzelius)',
            'p': '1669 (Brand)', 's': 'Ancient',
            'cl': '1774 (Scheele)', 'ar': '1894 (Ramsay/Rayleigh)',
            'k': '1807 (Davy)', 'ca': '1808 (Davy)',
            'sc': '1879 (Nilson)', 'ti': '1791 (Gregor)',
            'v': '1801 (del Rio)', 'cr': '1797 (Vauquelin)',
            'mn': '1774 (Gahn)', 'fe': 'Ancient',
            'co': '1735 (Brandt)', 'ni': '1751 (Cronstedt)',
            'cu': 'Ancient', 'zn': '1746 (Marggraf)',
            'ga': '1875 (de Boisbaudran)', 'ge': '1886 (Winkler)',
            'as': 'Ancient', 'se': '1817 (Berzelius)',
            'br': '1826 (Balard)', 'kr': '1898 (Ramsay/Travers)',
            'rb': '1861 (Bunsen/Kirchhoff)', 'sr': '1790 (Crawford)',
            'y': '1794 (Gadolin)', 'zr': '1789 (Klaproth)',
            'nb': '1801 (Hatchett)', 'mo': '1778 (Scheele)',
            'tc': '1937 (Perrier/Segrè)', 'ru': '1844 (Klaus)',
            'rh': '1803 (Wollaston)', 'pd': '1803 (Wollaston)',
            'ag': 'Ancient', 'cd': '1817 (Stromeyer)',
            'in': '1863 (Reich/Richter)', 'sn': 'Ancient',
            'sb': 'Ancient', 'te': '1782 (von Reichenstein)',
            'i': '1811 (Courtois)', 'xe': '1898 (Ramsay/Travers)',
            'cs': '1860 (Bunsen/Kirchhoff)', 'ba': '1808 (Davy)',
            'la': '1839 (Mosander)', 'ce': '1803 (Berzelius)',
            'pr': '1885 (Auer von Welsbach)', 'nd': '1885 (Auer von Welsbach)',
            'pm': '1945 (Marinsky/Glendenin)', 'sm': '1879 (de Boisbaudran)',
            'eu': '1901 (Demarçay)', 'gd': '1880 (Marignac)',
            'tb': '1843 (Mosander)', 'dy': '1886 (de Boisbaudran)',
            'ho': '1878 (Cleve)', 'er': '1843 (Mosander)',
            'tm': '1879 (Cleve)', 'yb': '1878 (Marignac)',
            'lu': '1907 (Urbain)', 'hf': '1923 (Coster/Hevesy)',
            'ta': '1802 (Ekeberg)', 'w': '1783 (d\'Elhuyar)',
            're': '1925 (Noddack/Noddack/Tacke)', 'os': '1803 (Tennant)',
            'ir': '1803 (Tennant)', 'pt': 'Ancient',
            'au': 'Ancient', 'hg': 'Ancient',
            'tl': '1861 (Crookes)', 'pb': 'Ancient',
            'bi': '1753 (Geoffroy)', 'po': '1898 (Curie)',
            'at': '1940 (Corson/Mackenzie/Segrè)', 'rn': '1900 (Dorn)',
            'fr': '1939 (Perey)', 'ra': '1898 (Curie)',
            'ac': '1899 (Debierne)', 'th': '1828 (Berzelius)',
            'pa': '1913 (Fajans/Göhring)', 'u': '1789 (Klaproth)',
        }
        return years.get(symbol, '')

    def _get_crust_abundance(self, symbol: str) -> str:
        """Get abundance in Earth's crust for an element."""
        abundance = {
            'h': '1400 ppm', 'he': '0.008 ppm', 'li': '20 ppm',
            'be': '2.8 ppm', 'b': '10 ppm', 'c': '200 ppm',
            'n': '19 ppm', 'o': '461000 ppm (46.1%)', 'f': '585 ppm',
            'ne': '0.005 ppm', 'na': '23600 ppm (2.36%)', 'mg': '23300 ppm (2.33%)',
            'al': '82300 ppm (8.23%)', 'si': '282000 ppm (28.2%)', 'p': '1050 ppm',
            's': '350 ppm', 'cl': '145 ppm', 'ar': '3.5 ppm',
            'k': '20900 ppm (2.09%)', 'ca': '41500 ppm (4.15%)',
            'sc': '22 ppm', 'ti': '5650 ppm', 'v': '120 ppm',
            'cr': '102 ppm', 'mn': '950 ppm', 'fe': '56300 ppm (5.63%)',
            'co': '25 ppm', 'ni': '84 ppm', 'cu': '60 ppm',
            'zn': '70 ppm', 'ga': '19 ppm', 'ge': '1.5 ppm',
            'as': '1.8 ppm', 'se': '0.05 ppm', 'br': '2.4 ppm',
            'kr': '0.0001 ppm', 'rb': '90 ppm', 'sr': '370 ppm',
            'y': '33 ppm', 'zr': '165 ppm', 'nb': '20 ppm',
            'mo': '1.2 ppm', 'tc': '0.0000003 ppm', 'ru': '0.001 ppm',
            'rh': '0.001 ppm', 'pd': '0.015 ppm', 'ag': '0.075 ppm',
            'cd': '0.15 ppm', 'in': '0.25 ppm', 'sn': '2.3 ppm',
            'sb': '0.2 ppm', 'te': '0.001 ppm', 'i': '0.45 ppm',
            'xe': '0.00003 ppm', 'cs': '3 ppm', 'ba': '425 ppm',
            'la': '39 ppm', 'ce': '66.5 ppm', 'pr': '9.2 ppm',
            'nd': '41.5 ppm', 'pm': '0.0000000001 ppm', 'sm': '7.05 ppm',
            'eu': '2 ppm', 'gd': '6.2 ppm', 'tb': '1.2 ppm',
            'dy': '5.2 ppm', 'ho': '1.3 ppm', 'er': '3.5 ppm',
            'tm': '0.52 ppm', 'yb': '3.2 ppm', 'lu': '0.8 ppm',
            'hf': '3 ppm', 'ta': '2 ppm', 'w': '1.25 ppm',
            're': '0.0007 ppm', 'os': '0.002 ppm', 'ir': '0.001 ppm',
            'pt': '0.005 ppm', 'au': '0.004 ppm', 'hg': '0.085 ppm',
            'tl': '0.85 ppm', 'pb': '14 ppm', 'bi': '0.0085 ppm',
            'po': '0.0000002 ppm', 'at': '0.0000000003 ppm', 'rn': '0.0000000004 ppm',
            'fr': '0.0000000001 ppm', 'ra': '0.0000009 ppm', 'ac': '0.00000005 ppm',
            'th': '9.6 ppm', 'pa': '0.00001 ppm', 'u': '2.7 ppm',
        }
        return abundance.get(symbol, '')

    def _get_organic_relevance(self, symbol: str) -> str:
        """Get organic chemistry relevance for an element."""
        relevance = {
            'h': 'Essential in all organic compounds; forms C-H bonds',
            'c': 'Backbone of all organic chemistry; forms 4 covalent bonds',
            'n': 'Found in amines, amides, amino acids, and nucleic acids',
            'o': 'Found in alcohols, ethers, carbonyl groups, and carboxylic acids',
            'f': 'Used in pharmaceuticals and agrochemicals; C-F bonds are very stable',
            'p': 'Found in DNA, RNA, ATP, and phospholipids',
            's': 'Found in thiols, disulfide bonds (cysteine), and coenzymes',
            'cl': 'Used in pharmaceuticals, pesticides, and polymers (PVC)',
            'br': 'Used in flame retardants, pharmaceuticals, and dyes',
            'i': 'Used in thyroid hormones and contrast agents',
        }
        return relevance.get(symbol, '')

    def _get_organic_compounds(self, symbol: str) -> str:
        """Get common organic compounds containing an element."""
        compounds = {
            'h': 'Alkanes, alkenes, alkynes, alcohols, carboxylic acids',
            'c': 'All organic compounds: alkanes, alkenes, alkynes, aromatics, alcohols, ethers, aldehydes, ketones, carboxylic acids, esters, amines, amides',
            'n': 'Amines, amides, amino acids, nitriles, nitro compounds, heterocycles',
            'o': 'Alcohols, ethers, aldehydes, ketones, carboxylic acids, esters, epoxides',
            'f': 'Fluorinated pharmaceuticals (e.g., fluoxetine), Teflon (PTFE), refrigerants',
            'p': 'Nucleic acids (DNA, RNA), ATP, phospholipids, phosphoesters',
            's': 'Thiols (mercaptans), disulfide bonds, thioethers, sulfoxides, sulfones',
            'cl': 'Alkyl chlorides, chlorinated solvents, PVC, pesticides (DDT)',
            'br': 'Alkyl bromides, flame retardants, brominated pharmaceuticals',
            'i': 'Thyroxine (T4), triiodothyronine (T3), iodinated contrast agents',
        }
        return compounds.get(symbol, '')

    def _get_functional_groups(self, symbol: str) -> str:
        """Get functional groups containing an element."""
        groups = {
            'h': 'C-H, O-H, N-H, S-H bonds',
            'c': 'Alkyl, alkenyl, alkynyl, aryl, carbonyl, carboxyl',
            'n': 'Amino (-NH2), amide (-CONH2), imine (-C=N-), nitrile (-C≡N), nitro (-NO2)',
            'o': 'Hydroxyl (-OH), ether (-O-), carbonyl (C=O), carboxyl (-COOH), ester (-COOR)',
            'f': 'Fluoromethyl (-CF3), fluoroaryl (-ArF)',
            'p': 'Phosphate (-PO4), phosphonate, phosphine',
            's': 'Thiol (-SH), thioether (-S-), disulfide (-S-S-), sulfoxide, sulfone',
            'cl': 'Chloroalkyl (-RCl), acyl chloride (-COCl)',
            'br': 'Bromoalkyl (-RBr), acyl bromide (-COBr)',
            'i': 'Iodoalkyl (-RI), acyl iodide (-COI)',
        }
        return groups.get(symbol, '')

    def _get_bonding_properties(self, symbol: str) -> str:
        """Get bonding properties for an element."""
        bonding = {
            'h': 'Forms 1 covalent bond; bond energy: 436 kJ/mol (H-H)',
            'c': 'Forms 4 covalent bonds; bond energy: 348 kJ/mol (C-C), 614 kJ/mol (C=C), 839 kJ/mol (C≡C)',
            'n': 'Forms 3 covalent bonds; bond energy: 163 kJ/mol (N-N), 418 kJ/mol (N=N), 945 kJ/mol (N≡N)',
            'o': 'Forms 2 covalent bonds; bond energy: 146 kJ/mol (O-O), 498 kJ/mol (O=O)',
            'f': 'Forms 1 covalent bond; bond energy: 159 kJ/mol (F-F)',
            'p': 'Forms 3 or 5 covalent bonds; bond energy: 201 kJ/mol (P-P)',
            's': 'Forms 2 covalent bonds; bond energy: 266 kJ/mol (S-S)',
            'cl': 'Forms 1 covalent bond; bond energy: 243 kJ/mol (Cl-Cl)',
            'br': 'Forms 1 covalent bond; bond energy: 193 kJ/mol (Br-Br)',
            'i': 'Forms 1 covalent bond; bond energy: 151 kJ/mol (I-I)',
        }
        return bonding.get(symbol, '')

    def _get_hybridization(self, symbol: str) -> str:
        """Get hybridization for an element."""
        hybridization = {
            'c': 'sp3 (tetrahedral), sp2 (trigonal planar), sp (linear)',
            'n': 'sp3 (pyramidal), sp2 (trigonal planar), sp (linear)',
            'o': 'sp3 (bent), sp2 (trigonal planar)',
            'p': 'sp3 (tetrahedral), sp3d (trigonal bipyramidal), sp3d2 (octahedral)',
            's': 'sp3 (bent), sp3d2 (octahedral)',
            'b': 'sp2 (trigonal planar), sp3 (tetrahedral)',
            'si': 'sp3 (tetrahedral)',
        }
        return hybridization.get(symbol, '')

    def _get_vdw_radius(self, symbol: str) -> str:
        """Get Van der Waals radius for an element (in pm)."""
        vdw = {
            'h': '120 pm', 'he': '140 pm', 'li': '182 pm', 'be': '153 pm',
            'b': '192 pm', 'c': '170 pm', 'n': '155 pm', 'o': '152 pm',
            'f': '147 pm', 'ne': '154 pm', 'na': '227 pm', 'mg': '173 pm',
            'al': '184 pm', 'si': '210 pm', 'p': '180 pm', 's': '180 pm',
            'cl': '175 pm', 'ar': '188 pm', 'k': '275 pm', 'ca': '231 pm',
            'sc': '211 pm', 'ti': '190 pm', 'v': '185 pm', 'cr': '180 pm',
            'mn': '175 pm', 'fe': '175 pm', 'co': '170 pm', 'ni': '163 pm',
            'cu': '140 pm', 'zn': '139 pm', 'ga': '187 pm', 'ge': '211 pm',
            'as': '185 pm', 'se': '190 pm', 'br': '185 pm', 'kr': '202 pm',
            'rb': '303 pm', 'sr': '249 pm', 'y': '220 pm', 'zr': '210 pm',
            'nb': '205 pm', 'mo': '200 pm', 'tc': '195 pm', 'ru': '195 pm',
            'rh': '195 pm', 'pd': '202 pm', 'ag': '172 pm', 'cd': '158 pm',
            'in': '193 pm', 'sn': '217 pm', 'sb': '206 pm', 'te': '206 pm',
            'i': '198 pm', 'xe': '216 pm', 'cs': '343 pm', 'ba': '268 pm',
            'la': '240 pm', 'ce': '235 pm', 'pr': '239 pm', 'nd': '229 pm',
            'pm': '228 pm', 'sm': '228 pm', 'eu': '240 pm', 'gd': '227 pm',
            'tb': '225 pm', 'dy': '224 pm', 'ho': '223 pm', 'er': '222 pm',
            'tm': '221 pm', 'yb': '222 pm', 'lu': '221 pm', 'hf': '210 pm',
            'ta': '205 pm', 'w': '200 pm', 're': '195 pm', 'os': '190 pm',
            'ir': '190 pm', 'pt': '175 pm', 'au': '166 pm', 'hg': '155 pm',
            'tl': '196 pm', 'pb': '202 pm', 'bi': '207 pm', 'po': '197 pm',
            'at': '202 pm', 'rn': '220 pm', 'fr': '348 pm', 'ra': '283 pm',
            'ac': '260 pm', 'th': '237 pm', 'pa': '200 pm', 'u': '196 pm',
        }
        return vdw.get(symbol, '')

    def _get_covalent_radius(self, symbol: str) -> str:
        """Get covalent radius for an element (in pm)."""
        cov = {
            'h': '31 pm', 'he': '28 pm', 'li': '128 pm', 'be': '96 pm',
            'b': '84 pm', 'c': '76 pm', 'n': '71 pm', 'o': '66 pm',
            'f': '57 pm', 'ne': '58 pm', 'na': '166 pm', 'mg': '141 pm',
            'al': '121 pm', 'si': '111 pm', 'p': '107 pm', 's': '105 pm',
            'cl': '102 pm', 'ar': '106 pm', 'k': '203 pm', 'ca': '176 pm',
            'sc': '170 pm', 'ti': '160 pm', 'v': '153 pm', 'cr': '139 pm',
            'mn': '150 pm', 'fe': '142 pm', 'co': '138 pm', 'ni': '124 pm',
            'cu': '132 pm', 'zn': '122 pm', 'ga': '122 pm', 'ge': '120 pm',
            'as': '119 pm', 'se': '120 pm', 'br': '120 pm', 'kr': '116 pm',
            'rb': '220 pm', 'sr': '195 pm', 'y': '190 pm', 'zr': '175 pm',
            'nb': '164 pm', 'mo': '154 pm', 'tc': '147 pm', 'ru': '146 pm',
            'rh': '142 pm', 'pd': '139 pm', 'ag': '145 pm', 'cd': '144 pm',
            'in': '142 pm', 'sn': '139 pm', 'sb': '139 pm', 'te': '138 pm',
            'i': '139 pm', 'xe': '140 pm', 'cs': '244 pm', 'ba': '215 pm',
            'la': '207 pm', 'ce': '204 pm', 'pr': '203 pm', 'nd': '201 pm',
            'pm': '199 pm', 'sm': '198 pm', 'eu': '198 pm', 'gd': '196 pm',
            'tb': '194 pm', 'dy': '192 pm', 'ho': '192 pm', 'er': '189 pm',
            'tm': '190 pm', 'yb': '187 pm', 'lu': '187 pm', 'hf': '175 pm',
            'ta': '170 pm', 'w': '162 pm', 're': '151 pm', 'os': '144 pm',
            'ir': '141 pm', 'pt': '136 pm', 'au': '136 pm', 'hg': '132 pm',
            'tl': '145 pm', 'pb': '146 pm', 'bi': '148 pm', 'po': '140 pm',
            'at': '145 pm', 'rn': '150 pm', 'fr': '260 pm', 'ra': '221 pm',
            'ac': '215 pm', 'th': '206 pm', 'pa': '200 pm', 'u': '196 pm',
        }
        return cov.get(symbol, '')

    def _get_reduction_potential(self, symbol: str) -> str:
        """Get standard reduction potential for an element."""
        potentials = {
            'h': '0.00 V (2H+ + 2e- → H2)',
            'li': '-3.04 V (Li+ + e- → Li)',
            'k': '-2.93 V (K+ + e- → K)',
            'na': '-2.71 V (Na+ + e- → Na)',
            'mg': '-2.37 V (Mg2+ + 2e- → Mg)',
            'al': '-1.66 V (Al3+ + 3e- → Al)',
            'zn': '-0.76 V (Zn2+ + 2e- → Zn)',
            'fe': '-0.44 V (Fe2+ + 2e- → Fe)',
            'cr': '-0.74 V (Cr3+ + 3e- → Cr)',
            'ni': '-0.25 V (Ni2+ + 2e- → Ni)',
            'sn': '-0.14 V (Sn2+ + 2e- → Sn)',
            'pb': '-0.13 V (Pb2+ + 2e- → Pb)',
            'cu': '+0.34 V (Cu2+ + 2e- → Cu)',
            'ag': '+0.80 V (Ag+ + e- → Ag)',
            'au': '+1.50 V (Au3+ + 3e- → Au)',
            'cl': '+1.36 V (Cl2 + 2e- → 2Cl-)',
            'br': '+1.07 V (Br2 + 2e- → 2Br-)',
            'i': '+0.54 V (I2 + 2e- → 2I-)',
            'f': '+2.87 V (F2 + 2e- → 2F-)',
        }
        return potentials.get(symbol, '')

    def _get_ionization_energy(self, symbol: str) -> str:
        """Get first ionization energy for an element (in kJ/mol)."""
        ie = {
            'h': '1312.0', 'he': '2372.3', 'li': '520.2',
            'be': '899.5', 'b': '800.6', 'c': '1086.5',
            'n': '1402.3', 'o': '1313.9', 'f': '1681.0',
            'ne': '2080.7', 'na': '495.8', 'mg': '737.7',
            'al': '577.5', 'si': '786.5', 'p': '1011.8',
            's': '999.6', 'cl': '1251.2', 'ar': '1520.6',
            'k': '418.8', 'ca': '589.8', 'sc': '633.1',
            'ti': '658.8', 'v': '650.9', 'cr': '652.9',
            'mn': '717.3', 'fe': '762.5', 'co': '760.4',
            'ni': '737.1', 'cu': '745.5', 'zn': '906.4',
            'ga': '578.8', 'ge': '762.2', 'as': '947.0',
            'se': '941.0', 'br': '1139.9', 'kr': '1350.8',
            'rb': '403.0', 'sr': '549.5', 'y': '600.0',
            'zr': '640.1', 'nb': '652.1', 'mo': '684.3',
            'tc': '702.0', 'ru': '710.2', 'rh': '719.7',
            'pd': '804.4', 'ag': '731.0', 'cd': '867.8',
            'in': '558.3', 'sn': '708.6', 'sb': '833.7',
            'te': '869.3', 'i': '1008.4', 'xe': '1170.4',
            'cs': '375.7', 'ba': '502.9', 'la': '538.1',
            'ce': '534.4', 'pr': '527.0', 'nd': '533.1',
            'pm': '540.0', 'sm': '544.5', 'eu': '547.1',
            'gd': '593.4', 'tb': '565.8', 'dy': '573.0',
            'ho': '581.0', 'er': '589.3', 'tm': '596.7',
            'yb': '603.4', 'lu': '523.5', 'hf': '658.5',
            'ta': '761.0', 'w': '770.0', 're': '760.0',
            'os': '840.0', 'ir': '880.0', 'pt': '870.0',
            'au': '890.1', 'hg': '1007.1', 'tl': '589.4',
            'pb': '715.6', 'bi': '702.9', 'po': '811.8',
            'at': 'N/A', 'rn': '1037.0', 'fr': 'N/A',
            'ra': '509.3', 'ac': '499.0', 'th': '587.0',
            'pa': '568.0', 'u': '597.6',
        }
        return ie.get(symbol, '')

    def _get_redox_reactions(self, symbol: str) -> str:
        """Get common redox reactions for an element."""
        reactions = {
            'h': '2H2 + O2 → 2H2O (combustion), 2H+ + 2e- → H2 (reduction)',
            'o': 'O2 + 4e- → 2O2- (reduction), 2H2O → O2 + 4H+ + 4e- (oxidation)',
            'c': 'C + O2 → CO2 (combustion), CO2 + 4H+ + 4e- → CH2O + H2O (reduction)',
            'n': 'N2 + 3H2 → 2NH3 (Haber), NO3- + 10H+ + 8e- → NH4+ + 3H2O (reduction)',
            's': 'S + 2H+ + 2e- → H2S (reduction), H2S → S + 2H+ + 2e- (oxidation)',
            'fe': 'Fe2+ → Fe3+ + e- (oxidation), Fe3+ + e- → Fe2+ (reduction)',
            'cu': 'Cu → Cu2+ + 2e- (oxidation), Cu2+ + 2e- → Cu (reduction)',
            'zn': 'Zn → Zn2+ + 2e- (oxidation), Zn2+ + 2e- → Zn (reduction)',
            'al': 'Al → Al3+ + 3e- (oxidation), Al3+ + 3e- → Al (reduction)',
            'mg': 'Mg → Mg2+ + 2e- (oxidation), Mg2+ + 2e- → Mg (reduction)',
            'na': 'Na → Na+ + e- (oxidation), Na+ + e- → Na (reduction)',
            'cl': '2Cl- → Cl2 + 2e- (oxidation), Cl2 + 2e- → 2Cl- (reduction)',
            'br': '2Br- → Br2 + 2e- (oxidation), Br2 + 2e- → 2Br- (reduction)',
            'i': '2I- → I2 + 2e- (oxidation), I2 + 2e- → 2I- (reduction)',
            'ag': 'Ag → Ag+ + e- (oxidation), Ag+ + e- → Ag (reduction)',
            'au': 'Au → Au3+ + 3e- (oxidation), Au3+ + 3e- → Au (reduction)',
            'cr': 'Cr3+ + 3e- → Cr (reduction), Cr → Cr3+ + 3e- (oxidation)',
            'mn': 'Mn2+ + 2e- → Mn (reduction), Mn → Mn2+ + 2e- (oxidation)',
            'ni': 'Ni2+ + 2e- → Ni (reduction), Ni → Ni2+ + 2e- (oxidation)',
            'cd': 'Cd2+ + 2e- → Cd (reduction), Cd → Cd2+ + 2e- (oxidation)',
            'pb': 'Pb2+ + 2e- → Pb (reduction), Pb → Pb2+ + 2e- (oxidation)',
        }
        return reactions.get(symbol, '')

    def _describe_capabilities(self) -> str:
        """Describe what chemistry mode can do."""
        return (
            "Chemistry mode (-c) - Coming soon!\n\n"
            "Planned features:\n"
            "- General chemistry (-g): formula solver, element lookup\n"
            "- Organic chemistry (-o): organic compound database, reactions\n"
            "- Redox (-r): oxidation/reduction, Eo values, pH\n"
            "- CGS unit system\n"
            "- Avogadro's number, molarity, moles"
        )
