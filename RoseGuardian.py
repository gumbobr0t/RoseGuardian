
import base64

import sys

import math

import os

import marshal

import zlib

import random

import time

import ast

import astor

import ctypes

import string

from colorama import Fore, init

init()

__version__ = '1.0.2'

key = random.randint(0, 255)

class clear():
    system = os.name
    if system == 'nt':
        os.system('cls')
    elif system == 'posix':
        os.system('clear')
    else:
        print('\n'*120)

def set_window_title(title):
    ctypes.windll.kernel32.SetConsoleTitleW(title)

system = os.name
if system == 'nt':
    set_window_title(f'RoseGuardian | {__version__}')
    os.system('mode CON: COLS=140 LINES=25')

def slow_print(text, delay=0.1):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def random_name(length=10):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

def remove_comments(node):
    for child in ast.walk(node):
        if isinstance(child, ast.FunctionDef):
            child.body = [n for n in child.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Str)]
        elif isinstance(child, ast.ClassDef):
            child.body = [n for n in child.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Str)]
        elif isinstance(child, ast.AsyncFunctionDef):
            child.body = [n for n in child.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Str)]
        elif isinstance(child, ast.Module):
            child.body = [n for n in child.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Str)]

def encode_strings(node):
    for child in ast.walk(node):
        if isinstance(child, ast.Str):
            obfuscated = ''.join(f'{ord(c) ^ key:02x}' for c in child.s)
            child.s = obfuscated

def replace_strings_with_decoding(node):
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name) and isinstance(child.value, ast.Str):
                    obfuscated = child.value.s
                    new_value = ast.Call(
                        func=ast.Name(id='decode_obfuscated_string', ctx=ast.Load()),
                        args=[ast.Str(s=obfuscated)],
                        keywords=[]
                    )
                    child.value = new_value

def remove_blank_lines(node):
    for child in ast.walk(node):
        if hasattr(child, 'body'):
            child.body = [n for n in child.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Str)]

def rename_identifiers(node, mapping):
    if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
        node.name = mapping.get(node.name, node.name)

    if isinstance(node, ast.Name):
        node.id = mapping.get(node.id, node.id)

    if isinstance(node, ast.ClassDef):
        node.name = mapping.get(node.name, node.name)

    for child in ast.iter_child_nodes(node):
        rename_identifiers(child, mapping)

def obfuscator1():
    with open(os.path.abspath(sys.argv[1]), 'r') as f:
        slow_print(Fore.RED + 'Reading source code...\n' + Fore.RESET, delay=0.005)
        source_code = f.read()
    
    def random_name():
        return ''.join(random.choices(string.ascii_letters, k=10))

    slow_print(Fore.RED + 'Parsing tree...\n' + Fore.RESET, delay=0.005)

    tree = ast.parse(source_code)

    slow_print(Fore.RED + 'Removing comments...\n' + Fore.RESET, delay=0.005)
    
    remove_comments(tree)
                    
    slow_print(Fore.RED + 'Removing blank lines...\n' + Fore.RESET, delay=0.005)
    
    remove_blank_lines(tree)

    slow_print(Fore.RED + 'Changing class and function names...\n' + Fore.RESET, delay=0.005)
    functions = set()
    classes = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            functions.add(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.add(node.name)

    mapping = {name: random_name() for name in functions | classes}

    rename_identifiers(tree, mapping)

    slow_print(Fore.RED + 'Compressing and base64 encoding the marshalized objects...\n' + Fore.RESET, delay=0.005)
    obfuscated_code = base64.b64encode(zlib.compress(marshal.dumps(astor.to_source(tree).encode())))

    return obfuscated_code

def obfuscator0(input_file, output_file):
    with open(input_file, 'r') as file:
        source_code = file.read()

    tree = ast.parse(source_code)

    remove_comments(tree)

    remove_blank_lines(tree)

    encode_strings(tree)

    variables = set()
    functions = set()
    classes = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            functions.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            variables.add(node.id)
        elif isinstance(node, ast.ClassDef):
            classes.add(node.name)

    mapping = {name: random_name() for name in variables | functions | classes}

    rename_identifiers(tree, mapping)
    
    with open(output_file, 'w') as file:
        file.write(f"key = {key}\n\n")
        file.write("def decode_obfuscated_string(obfuscated_string):\n")
        file.write("    encoded = bytes.fromhex(obfuscated_string)\n")
        file.write("    decoded = ''.join(chr(b ^ key) for b in encoded)\n")
        file.write("    return decoded\n\n")
        replace_strings_with_decoding(tree)
        file.write(ast.unparse(tree))

def get_junk():

    _1 = '''def morphogenic_adapter():
    pass

class quantum_oscillator:
    def __init__(self):
        self._ = None
        self.__ = None

    def hyperbolic_momentum(self, _):
        return self.hyperbolic_momentum(_)

def tachyon_cascade():
    _ = quantum_oscillator()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def stellar_phase_converter():
    _ = morphogenic_adapter()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def temporal_disruptor():
    pass'''

    _2 = '''def spectral_synthesizer():
    pass

class warp_singularity:
    def __init__(self):
        self._ = None
        self.__ = None

    def temporal_siphon(self, _):
        return self.temporal_siphon(_)

def entropic_fluctuation():
    _ = warp_singularity()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def quantum_reactor():
    _ = spectral_synthesizer()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def graviton_matrix():
    pass
'''

    _3 = '''def quantum_resonator():
    pass

class singularity_nexus:
    def __init__(self):
        self._ = None
        self.__ = None

    def hyperspatial_field(self, _):
        return self.hyperspatial_field(_)

def chronometric_analyzer():
    _ = singularity_nexus()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def gravimetric_capacitor():
    _ = quantum_resonator()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def tachyon_diffractor():
    pass
'''

    _4 = '''def nebula_processor():
    pass

class interstellar_grid:
    def __init__(self):
        self._ = None
        self.__ = None

    def quantum_encoder(self, _):
        return self.quantum_encoder(_)

def cosmic_displacement():
    _ = interstellar_grid()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def subspace_modulator():
    _ = nebula_processor()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def dark_matter_calibrator():
    pass
'''

    _5 = '''def gravimetric_flux():
    pass

class warp_inverter:
    def __init__(self):
        self._ = None
        self.__ = None

    def subspace_transducer(self, _):
        return self.subspace_transducer(_)

def chronal_conduit():
    _ = warp_inverter()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def singularity_stabilizer():
    _ = gravimetric_flux()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def entropic_reactor():
    pass
'''

    _6 = '''def quantum_transmitter():
    pass

class flux_capacitor:
    def __init__(self):
        self._ = None
        self.__ = None

    def tachyon_oscillator(self, _):
        return self.tachyon_oscillator(_)

def spatial_phase():
    _ = flux_capacitor()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def temporal_analyzer():
    _ = quantum_transmitter()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def stellar_modulator():
    pass
'''

    _7 = '''def nebular_computron():
    pass

class cosmic_relay:
    def __init__(self):
        self._ = None
        self.__ = None

    def subspace_coil(self, _):
        return self.subspace_coil(_)

def graviton_amplifier():
    _ = cosmic_relay()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def singularity_catalyst():
    _ = nebular_computron()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def temporal_enigma():
    pass
'''

    _8 = '''def hyperion_mechanism():
    pass

class gravimetric_tensor:
    def __init__(self):
        self._ = None
        self.__ = None

    def warp_crypt(self, _):
        return self.warp_crypt(_)

def quantum_effluvium():
    _ = gravimetric_tensor()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def subspace_resonator():
    _ = hyperion_mechanism()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def temporal_reverberator():
    pass
'''

    _9 = '''def exo_quantum():
    pass

class astro_waveform:
    def __init__(self):
        self._ = None
        self.__ = None

    def flux_capacitance(self, _):
        return self.flux_capacitance(_)

def chrono_stasis():
    _ = astro_waveform()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def gravitic_magnetron():
    _ = exo_quantum()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def subspace_scrambler():
    pass
'''

    _10 = '''def tachyon_entangler():
    pass

class quantum_fusion:
    def __init__(self):
        self._ = None
        self.__ = None

    def warp_induction(self, _):
        return self.warp_induction(_)

def chrono_spectral():
    _ = quantum_fusion()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def gravimetric_anomaly():
    _ = tachyon_entangler()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def subspace_phase_lock():
    pass
'''

    _11 = '''def astral_inverter():
    pass

class stellar_singularity:
    def __init__(self):
        self._ = None
        self.__ = None

    def quantum_nucleus(self, _):
        return self.quantum_nucleus(_)

def hyper_gravimetric():
    _ = stellar_singularity()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def temporal_deflector():
    _ = astral_inverter()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def subspace_injector():
    pass
'''

    _12 = '''def chrono_tesseract():
    pass

class gravimetric_reactor:
    def __init__(self):
        self._ = None
        self.__ = None

    def warp_manifold(self, _):
        return self.warp_manifold(_)

def quantum_vortex():
    _ = gravimetric_reactor()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def astral_siphon():
    _ = chrono_tesseract()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def hyper_flux_analyzer():
    pass
'''

    _13 = '''def warp_oscillator():
    pass

class gravimetric_disruptor:
    def __init__(self):
        self._ = None
        self.__ = None

    def subspace_transmitter(self, _):
        return self.subspace_transmitter(_)

def quantum_chronometer():
    _ = gravimetric_disruptor()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def astral_magnetar():
    _ = warp_oscillator()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def hyperbolic_reactor():
    pass
'''

    _14 = '''def temporal_nexus():
    pass

class quantum_flux:
    def __init__(self):
        self._ = None
        self.__ = None

    def warp_field(self, _):
        return self.warp_field(_)

def stellar_maneuver():
    _ = quantum_flux()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def subspace_chronicle():
    _ = temporal_nexus()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def astral_deflector():
    pass
'''

    _15 = '''def warp_modulator():
    pass

class gravimetric_inverter:
    def __init__(self):
        self._ = None
        self.__ = None

    def quantum_matrix(self, _):
        return self.quantum_matrix(_)

def stellar_entropy():
    _ = gravimetric_inverter()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def temporal_spectral():
    _ = warp_modulator()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def subspace_transmogrifier():
    pass
'''

    _16 = '''def quantum_fissure():
    pass

class stellar_flux:
    def __init__(self):
        self._ = None
        self.__ = None

    def warp_cloak(self, _):
        return self.warp_cloak(_)

def chrono_singularity():
    _ = stellar_flux()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def gravimetric_oscillation():
    _ = quantum_fissure()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def astral_paradigm():
    pass
'''

    _17 = '''def genetic_analyzer():
    pass

class dna_sequencer:
    def __init__(self):
        self._ = None
        self.__ = None

    def genetic_encoder(self, _):
        return self.genetic_encoder(_)

def mutation_inducer():
    _ = dna_sequencer()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_))
'''

    _18 = '''def neural_processor():
    pass

class synaptic_matrix:
    def __init__(self):
        self._ = None
        self.__ = None

    def dendritic_cascade(self, _):
        return self.dendritic_cascade(_)

def cognitive_amplifier():
    _ = synaptic_matrix()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def thought_inhibitor():
    _ = neural_processor()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def consciousness_monitor():
    pass
'''

    _19 = '''def quantum_analyzer():
    pass

class wave_function:
    def __init__(self):
        self._ = None
        self.__ = None

    def entanglement_resolver(self, _):
        return self.entanglement_resolver(_)

def probability_calculator():
    _ = wave_function()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def uncertainty_optimizer():
    _ = quantum_analyzer()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def wave_collapser():
    pass
'''

    _20 = '''def algorithm_optimizer():
    pass

class computational_neuron:
    def __init__(self):
        self._ = None
        self.__ = None

    def synaptic_connection(self, _):
        return self.synaptic_connection(_)

def heuristic_generator():
    _ = computational_neuron()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def pattern_analyzer():
    _ = algorithm_optimizer()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def data_compiler():
    pass
'''

    _21 = '''def audio_processor():
    pass

class waveform_converter:
    def __init__(self):
        self._ = None
        self.__ = None

    def frequency_analyzer(self, _):
        return self.frequency_analyzer(_)

def spectral_filter():
    _ = waveform_converter()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def audio_resampler():
    pass
'''

    _22 = '''def data_encoder():
    pass

class compression_algorithm:
    def __init__(self):
        self._ = None
        self.__ = None

    def data_compressor(self, _):
        return self.data_compressor(_)

def encryption_scheme():
    _ = compression_algorithm()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def data_decoder():
    pass
'''

    _23 = '''def image_processor():
    pass

class color_correction:
    def __init__(self):
        self._ = None
        self.__ = None

    def hue_adjustment(self, _):
        return self.hue_adjustment(_)

def image_filter():
    _ = color_correction()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def image_resizer():
    pass
'''

    _24 = '''def text_analyzer():
    pass

class sentiment_classifier:
    def __init__(self):
        self._ = None
        self.__ = None

    def emotion_analysis(self, _):
        return self.emotion_analysis(_)

def text_summarizer():
    _ = sentiment_classifier()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def text_translator():
    pass
'''

    _25 = '''def financial_analyzer():
    pass

class portfolio_optimizer:
    def __init__(self):
        self._ = None
        self.__ = None

    def risk_assessment(self, _):
        return self.risk_assessment(_)

def investment_strategy():
    _ = portfolio_optimizer()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def asset_alocator():
    pass
'''

    _26 = '''def network_monitor():
    pass

class packet_sniffer:
    def __init__(self):
        self._ = None
        self.__ = None

    def traffic_analyzer(self, _):
        return self.traffic_analyzer(_)

def security_audit():
    _ = packet_sniffer()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def intrusion_detection():
    pass
'''

    _27 = '''def medical_diagnostic():
    pass

class patient_monitor:
    def __init__(self):
        self._ = None
        self.__ = None

    def vital_signs(self, _):
        return self.vital_signs(_)

def disease_classifier():
    _ = patient_monitor()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def treatment_planner():
    pass
'''

    _28 = '''def climate_sensor():
    pass

class temperature_controller:
    def __init__(self):
        self._ = None
        self.__ = None

    def climate_regulator(self, _):
        return self.climate_regulator(_)

def humidity_adjuster():
    _ = temperature_controller()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def air_quality_monitor():
    pass
'''

    _29 = '''def audio_processor():
    return None

class WaveformConverter:
    def __init__(self):
        self.data = None
        self.metadata = None

    def frequency_analysis(self, data):
        return self.frequency_analysis(data)

def SpectralFilter():
    converter = WaveformConverter()
    converter.data = lambda x, y: x(y) + x(x(x(x(y))))

def AudioResampler():
    pass
'''

    _30 = '''def DataEncoder():
    return None

class CompressionAlgorithm:
    def __init__(self):
        self.buffer = None
        self.settings = None

    def CompressData(self, buffer):
        return self.CompressData(buffer)

def EncryptionScheme():
    algorithm = CompressionAlgorithm()
    algorithm.buffer = lambda x, y: x(y) + x(x(x(x(y))))

def DataDecoder():
    pass
'''

    _31 = '''def ImageProcessor():
    return None

class ColorCorrection:
    def __init__(self):
        self.settings = None
        self.mask = None

    def AdjustHue(self, image):
        return self.AdjustHue(image)

def ImageFilter():
    correction = ColorCorrection()
    correction.mask = lambda x, y: x(y) + x(x(x(x(y))))

def ImageResizer():
    pass
'''

    _32 = '''def TextAnalyzer():
    return None

class SentimentClassifier:
    def __init__(self):
        self.model = None
        self.vocab = None

    def AnalyzeEmotion(self, text):
        return self.AnalyzeEmotion(text)

def TextSummarizer():
    classifier = SentimentClassifier()
    classifier.model = lambda x, y: x(y) + x(x(x(x(y))))

def TextTranslator():
    pass
'''

    _33 = '''def FinancialAnalyzer():
    return None

class PortfolioOptimizer:
    def __init__(self):
        self.portfolio = None
        self.strategy = None

    def AssessRisk(self, portfolio):
        return self.AssessRisk(portfolio)

def InvestmentStrategy():
    optimizer = PortfolioOptimizer()
    optimizer.strategy = lambda x, y: x(y) + x(x(x(x(y))))

def AssetAllocator():
    pass
'''

    _34 = '''def complex_processor():
    result = 0
    for i in range(10):
        result += i * (i + 1) / (i + 2)
    return result

class ComplexProcessor:
    def __init__(self):
        self._ = None
        self.__ = None

    def process_data(self, data):
        return self.process_data(data)

def perform_calculation():
    _ = ComplexProcessor()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def simulate_system():
    value = 1
    for _ in range(5):
        value *= value + 1
'''

    _35 = '''def genetic_optimizer():
    result = 1
    for i in range(1, 11):
        result *= i
    return result

class GeneticAlgorithm:
    def __init__(self):
        self._ = None
        self.__ = None

    def optimize(self):
        pass

def calculate_fitness():
    _ = GeneticAlgorithm()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def evaluate_population():
    total = 0
    for i in range(1, 6):
        total += i**2
    return total
'''
   
    _36 = '''def particle_simulator():
    result = 0
    for i in range(1, 21):
        result += i**3 - i**2 + i
    return result

class ParticleSimulator:
    def __init__(self):
        self._ = None
        self.__ = None

    def simulate(self, iterations):
        return self.simulate(iterations)

def simulate_particles():
    _ = ParticleSimulator()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def analyze_data():
    data = [1, 2, 3, 4, 5]
    result = sum(data) / len(data)
'''

    _37 = '''def matrix_multiplier():
    matrix_a = [[1, 2], [3, 4]]
    matrix_b = [[5, 6], [7, 8]]
    result = [[0, 0], [0, 0]]

    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result'''

    _38 = '''def newton_raphson():
    def f(x):
        return x**3 - 6 * x**2 + 11 * x - 6

    def df(x):
        return 3 * x**2 - 12 * x + 11

    x0 = 1.0
    for _ in range(5):
        x0 = x0 - f(x0) / df(x0)

    return x0
'''

    _39 = '''def complex_integration():
    def f(x):
        return x**2 + 2 * x + 1

    a = 0
    b = 2
    n = 1000
    h = (b - a) / n

    result = 0
    for i in range(n):
        result += h * (f(a + i*h) + f(a + (i+1)*h)) / 2

    return result
'''

    _40 = '''def random_walk():
    import random

    position = 0
    steps = 1000

    for _ in range(steps):
        if random.random() < 0.5:
            position += 1
        else:
            position -= 1

    return position'''

    _41 = '''def monte_carlo_simulation():
    import random

    inside_circle = 0
    total_points = 100000

    for _ in range(total_points):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        if x**2 + y**2 <= 1:
            inside_circle += 1

    return 4 * inside_circle / total_points
'''
    
    _42 = '''def complicated_function():
    result = 0
    for i in range(1, 11):
        result += i**3 - i**2 + i
    return result

class ComplicatedAlgorithm:
    def __init__(self):
        self._ = None
        self.__ = None

    def execute(self):
        pass

def execute_complicated_algorithm():
    _ = ComplicatedAlgorithm()
    _._ = lambda _, __: _.___(__) + _.___(_.___(_.__(_.___(_.__))))

def analyze_data():
    import random
    
    data = [random.randint(1, 100) for _ in range(10)]
    result = sum(data) / len(data)
'''
    
    _43 = '''def hard_equation_solver():
    result = 0
    for i in range(1, 11):
        result += i**5 - i**4 + i**3 - i**2 + i - 1
    return result

class EquationSolver:
    def __init__(self):
        self._ = None
        self.__ = None

    def solve(self):
        pass

    def quadratic_equation(self, a, b, c):
        discriminant = b**2 - 4*a*c
        if discriminant > 0:
            x1 = (-b + discriminant**0.5) / (2*a)
            x2 = (-b - discriminant**0.5) / (2*a)
            return x1, x2
        elif discriminant == 0:
            x = -b / (2*a)
            return x
        else:
            real_part = -b / (2*a)
            imaginary_part = (abs(discriminant)**0.5) / (2*a)
            return (real_part + imaginary_part * 1j, real_part - imaginary_part * 1j)

    def integrate_polynomial(self, coefficients):
        result = [coeff / (i+1) for i, coeff in enumerate(coefficients)]
        result.append(0)
        return result

def solve_equation():
    _ = EquationSolver()
    _.quadratic_equation(1, -3, 2)

def compute_fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return compute_fibonacci(n-1) + compute_fibonacci(n-2)
'''

    _44 = '''def junk_function_1():
    result = 0
    for i in range(1, 6):
        result += i**3 - i**2 + i - 1
    return result

class RandomEquationSolver:
    def __init__(self):
        self.a = 5
        self.b = 3

    def solve(self):
        return self.a * self.b + self.a - self.b

    def random_operation(self):
        return self.a**2 + self.b**3 - self.a * self.b

def solve_random_equation():
    solver = RandomEquationSolver()
    return solver.solve()

def compute_random_sequence(length):
    return [i**2 - i + 1 for i in range(length)]

def find_random_factors(n):
    factors = []
    while n % 2 == 0:
        factors.append(2)
        n = n // 2
    for i in range(3, int(n**0.5) + 1, 2):
        while n % i == 0:
            factors.append(i)
            n = n // i
    if n > 2:
        factors.append(n)
    return factors

def calculate_random_gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def generate_random_points(num_points):
    points = []
    for _ in range(num_points):
        x = 2 * _ + 1
        y = 3 * _ + 2
        points.append((x, y))
    return points
'''

    _45 = '''def multiply_matrices(mat1, mat2):
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            total = 0
            for k in range(len(mat2)):
                total += mat1[i][k] * mat2[k][j]
            row.append(total)
        result.append(row)
    return result
'''

    return random.choice([_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45])

def write_code():
    obfuscated_file = os.path.join(os.getcwd(),  'obfuscated_' + f"{os.path.basename(os.path.abspath(sys.argv[1]))}")

    with open(os.path.join(obfuscated_file), 'w') as f:

        if (len(sys.argv) > 2 and sys.argv[2].isdigit()):

            f.write("""__obfuscator__ = 'RoseGuardian'
__author__ = 'gumbobr0t'
__github__ = 'https://github.com/DamagingRose/RoseGuardian'
__license__ = 'EPL-2.0'\n\n""")
            
            slow_print(Fore.RED + 'Junk method detected...\n' + Fore.RESET, delay=0.005)

            slow_print(Fore.RED + 'Pumping first junk layer into code...\n' + Fore.RESET, delay=0.005)
            
            for i in range(math.ceil(int(sys.argv[2])/2)):

                f.write(f'\n{get_junk()}')

        if (len(sys.argv) > 3 and sys.argv[3].isdigit()):
            
            if (sys.argv[3] == '0'):

                slow_print(Fore.RED + 'Obfuscation method 0 detected...\n' + Fore.RESET, delay=0.005)
        
                with open(os.path.abspath(sys.argv[1]), 'r') as file:
                    slow_print(Fore.RED + 'Reading source code...\n' + Fore.RESET, delay=0.005)
                    source_code = file.read()

                slow_print(Fore.RED + 'Parsing tree...\n' + Fore.RESET, delay=0.005)
                tree = ast.parse(source_code)

                slow_print(Fore.RED + 'Removing comments...\n' + Fore.RESET, delay=0.005)
                remove_comments(tree)
                    
                slow_print(Fore.RED + 'Removing blank lines...\n' + Fore.RESET, delay=0.005)
                remove_blank_lines(tree)
    
                slow_print(Fore.RED + 'Encoding strings...\n' + Fore.RESET, delay=0.005)
                encode_strings(tree)

                variables = set()
                functions = set()
                classes = set()

                slow_print(Fore.RED + 'Changing class, function and variable names...\n' + Fore.RESET, delay=0.005)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                        functions.add(node.name)
                    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        variables.add(node.id)
                    elif isinstance(node, ast.ClassDef):
                        classes.add(node.name)

                mapping = {name: random_name() for name in variables | functions | classes}

                rename_identifiers(tree, mapping)

                slow_print(Fore.RED + 'Adding self-decode to strings...\n' + Fore.RESET, delay=0.005)
                f.write(f"\n\nkey = {key}\n\n")
                f.write("def decode_obfuscated_string(obfuscated_string):\n")
                f.write("    encoded = bytes.fromhex(obfuscated_string)\n")
                f.write("    decoded = ''.join(chr(b ^ key) for b in encoded)\n")
                f.write("    return decoded\n\n")
                replace_strings_with_decoding(tree)
                f.write(ast.unparse(tree))

            if (sys.argv[3] == '1'):

                slow_print(Fore.RED + 'Obfuscation method 1 detected...\n' + Fore.RESET, delay=0.005)

                slow_print(Fore.RED + 'Writing obfuscated code to file...\n' + Fore.RESET, delay=0.005)
                
                f.write('\n\nimport marshal, base64, zlib; exec(marshal.loads(zlib.decompress(base64.b64decode(' + repr(obfuscator1()) + '))))')

        if (len(sys.argv) > 2 and sys.argv[2].isdigit()):

            slow_print(Fore.RED + 'Pumping second junk layer into code...\n' + Fore.RESET, delay=0.005)
            
            for i in range(math.ceil(int(sys.argv[2])/2)):

                f.write(f'\n{get_junk()}')

if (len(sys.argv) != 4):
    
    slow_print(Fore.RED + 'Usage: python RoseGuardian.py <filename> <junk layers> <obfuscation method>\n' + Fore.RESET, delay=0.005)
    slow_print(Fore.RED + 'Example: python RoseGuardian.py <your file> 10 1\n' + Fore.RESET, delay=0.005)
    
    sys.exit(0)

if (len(sys.argv) == 4):

    if not os.path.exists(os.path.abspath(sys.argv[1])):
            
        slow_print(Fore.RED + f'File {sys.argv[1]} was not found.\n' + Fore.RESET, delay=0.005)
            
        sys.exit(0)

    if os.path.splitext(os.path.abspath(sys.argv[1]))[1] not in ('.py', '.pyc'):

        slow_print(Fore.RED + f'File {sys.argv[1]} is not a valid python file (.py or .pyc).\n' + Fore.RESET, delay=0.005)
            
        sys.exit(0)

    try:

        write_code()

        slow_print(Fore.RED + f'{sys.argv[1]} -> {"obfuscated_" + f"{sys.argv[1]}"}\n' + Fore.RESET, delay=0.005)

    except Exception as e:
    
        slow_print(Fore.RED + f'Failed to obfuscate {os.path.abspath(sys.argv[1])}.\n\nError: "{e}"\n' + Fore.RESET, delay=0.005)
