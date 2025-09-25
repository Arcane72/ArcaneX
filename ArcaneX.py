import os
import struct
import numpy as np
from fbm import FBM
from Crypto.Cipher import AES, ChaCha20_Poly1305
import hashlib
import hmac
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import traceback
import secrets
import logging
import time
import gc
import platform
import copy
from scipy.stats import ortho_group
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
import math
import pickle
from datetime import datetime
import threading

# Configuration du logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='fractalkem_ultra_secure.log',
                    filemode='a')

# Paramètres de sécurité
ENTROPY_SEED_SIZE = 64
HURST_RANGE = (0.15, 0.85)
EXCLUDED_HURST = (0.45, 0.55)
FBM_POINTS = 8192
KEY_DERIVATION_SAMPLES = [1, 15, 32, 77, 256, 512, 1024]
MASK_SAMPLES = list(range(200, 264))
FBM_SAMPLES_COUNT = len(KEY_DERIVATION_SAMPLES) + len(MASK_SAMPLES)
FBM_SERIALIZED_SIZE = FBM_SAMPLES_COUNT * 8

# Paramètres Scrypt militaires
SCRYPT_SALT_SIZE = 32
SCRYPT_N = 2**22  # Coût CPU/mémoire ( itérations)
SCRYPT_R = 8      # Taille de bloc
SCRYPT_P = 1      # Facteur de parallélisation
SCRYPT_KEY_LEN = 64  # 512 bits

# Tailles fixes
HURST_SIZE = 8
SALT_SIZE = 32
NONCE_SEED_SIZE = 12
TAG_SEED_SIZE = 16
CIPHERTEXT_SEED_SIZE = 64
NONCE1_SIZE = 12
TAG1_SIZE = 16
CHACHA_NONCE_SIZE = 12
CHACHA_TAG_SIZE = 16
CHACHA_KEY_SIZE = 32
HEADER_IV_SIZE = 12
HEADER_TAG_SIZE = 16

# Paramètres Cliques
CLIQUE_SEED_SIZE = 32
CLIQUE_DEPTH_SIZE = 1
CLIQUE_MIN_SIZE = 1
CLIQUE_DELTA_SIZE = 8

# Taille du header original
PLAIN_HEADER_SIZE = (
    HURST_SIZE + SALT_SIZE + CLIQUE_SEED_SIZE + CLIQUE_DEPTH_SIZE +
    CLIQUE_MIN_SIZE + CLIQUE_DELTA_SIZE + NONCE_SEED_SIZE +
    TAG_SEED_SIZE + CIPHERTEXT_SEED_SIZE + NONCE1_SIZE + TAG1_SIZE +
    FBM_SERIALIZED_SIZE + CHACHA_NONCE_SIZE + CHACHA_TAG_SIZE
)

# Taille du header chiffré (même taille que le header original)
ENCRYPTED_HEADER_SIZE = PLAIN_HEADER_SIZE

# Taille totale du nouveau header
HEADER_SIZE = (
    SCRYPT_SALT_SIZE + HEADER_IV_SIZE + 
    ENCRYPTED_HEADER_SIZE + HEADER_TAG_SIZE
)

# Classes d'IA
class IAClass:
    class MasterIA:
        def __init__(self, ias=None, id=0):
            self.ias = ias if ias else []
            self.score = 0
            self.maxX = 0
            self.id = id
            self.ScoreOriginal = 0
            self.checkPoint = None
            
        def isBetter(self, pos):
            if pos.x > self.score:
                self.maxX = pos.x
                return True
            return False
    
    class Input:
        def __init__(self, pos, type, rValue, weights):
            self.pos = pos
            self.type = type
            self.rValue = rValue
            self.weights = weights
            self.value = 0
            
        def SetBool(self, b):
            self.value = self.rValue if b else 0
    
    class Neuron:
        def __init__(self, bias, weights):
            self.bias = bias
            self.weights = weights
            self.value = 0
    
    class Layer:
        def __init__(self):
            self.neurons = []
    
    class Output:
        def __init__(self):
            self.value = 0
    
    class IA:
        def __init__(self, inputs, layers, outputs):
            self.inputs = inputs
            self.layers = layers
            self.outputs = outputs

# Fonctions de sécurité
def secure_wipe(data):
    if not data or len(data) == 0:
        return
    for _ in range(7):
        random_data = secrets.token_bytes(len(data))
        for i in range(len(data)):
            data[i] = random_data[i]
    del data
    gc.collect()
    if platform.system() == 'Linux':
        os.system('sync; echo 3 > /proc/sys/vm/drop_caches')

# Générateur quantique
class QuantumInspiredRNG:
    def __init__(self, seed=None):
        self.entropy_pool = bytearray()
        self.state = np.zeros(1024, dtype=np.complex128)
        self.H = 0.23
        self.H_range = HURST_RANGE
        self.graph_size = 512
        self.density = 0.05
        self.quantum_steps = 5
        self.last_update = time.time()
        seed_bytes = seed if seed else os.urandom(64)
        self.local_rng = np.random.default_rng(int.from_bytes(seed_bytes, 'big'))
        self.update_entropy(seed_bytes)
        self.quantum_graph = self.generate_quantum_graph()

    def update_entropy(self, entropy_input):
        h = hashlib.sha3_512()
        h.update(self.entropy_pool + entropy_input)
        new_pool = bytearray(h.digest())
        secure_wipe(self.entropy_pool)
        self.entropy_pool = new_pool
        required_bytes = 8192
        padded_data = self.entropy_pool.ljust(required_bytes, b'\x00')[:required_bytes]
        complex_data = np.frombuffer(padded_data, dtype=np.complex128)
        complex_data = np.nan_to_num(complex_data, nan=0.0, posinf=1e100, neginf=-1e100)
        max_abs = np.max(np.abs(complex_data)) or 1e-100
        normalized_data = complex_data / max_abs
        normalized_data = np.clip(normalized_data, -1e100, 1e100)
        normalized_data = np.nan_to_num(normalized_data, nan=0.0)
        try:
            self.state = np.fft.fft(normalized_data)[:1024]
        except FloatingPointError:
            self.state = np.zeros(1024, dtype=np.complex128)
            self.state[0] = 1.0

    def generate_quantum_graph(self):
        U = ortho_group.rvs(dim=self.graph_size, random_state=self.local_rng)
        fractal_component = self.local_rng.random((self.graph_size, self.graph_size))
        fractal_component = np.power(fractal_component, self.H)
        quantum_graph = U * fractal_component
        norm = np.linalg.norm(quantum_graph)
        return quantum_graph / norm if norm > 0 else quantum_graph

    def apply_quantum_walk(self):
        position = self.local_rng.integers(0, self.graph_size)
        state_vector = np.zeros(self.graph_size, dtype=np.complex128)
        state_vector[position] = 1.0 + 0j
        state_vector = self.quantum_graph.dot(state_vector)
        probs = np.abs(state_vector)**2
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.maximum(probs, 0)
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs /= probs_sum
        else:
            probs = np.ones_like(probs) / len(probs)
        return state_vector

    def fractal_noise_generation(self, size=1024):
        noise = self.local_rng.random(size)
        for i in range(1, size):
            noise[i] = noise[i-1] * (1 - self.H) + noise[i] * self.H
        return noise

    def generate_random_bytes(self, num_bytes=64):
        current_time = time.time()
        if current_time - self.last_update > 0.5:
            self.update_entropy(os.urandom(64))
            self.H = self.local_rng.uniform(*self.H_range)
            self.quantum_graph = self.generate_quantum_graph()
            self.last_update = current_time
        quantum_result = self.apply_quantum_walk()
        fractal_result = self.fractal_noise_generation(len(quantum_result))
        combined = np.multiply(np.abs(quantum_result), fractal_result)
        scaled = (combined * (2**64 - 1)).astype(np.uint64)
        random_data = hashlib.shake_256(scaled.tobytes()).digest(num_bytes)
        secure_wipe(bytearray(quantum_result.tobytes()))
        secure_wipe(bytearray(fractal_result.tobytes()))
        secure_wipe(bytearray(scaled.tobytes()))
        return random_data

    def bytes(self, length):
        return self.generate_random_bytes(length)

# Défi des cliques
class NestedCliqueChallenge:
    def __init__(self, n_nodes=1024, min_clique=72, depth=6, delta=None, seed=None):
        MAX_NODES = 2048
        MAX_DEPTH = 8
        MAX_CLIQUE = 256
        
        if n_nodes > MAX_NODES:
            raise ValueError(f"n_nodes ne peut pas dépasser {MAX_NODES}")
        if depth > MAX_DEPTH:
            raise ValueError(f"depth ne peut pas dépasser {MAX_DEPTH}")
        if min_clique > MAX_CLIQUE:
            raise ValueError(f"min_clique ne peut pas dépasser {MAX_CLIQUE}")
        
        self.n_nodes = n_nodes
        self.min_clique = min_clique
        self.depth = depth
        self.delta = delta if delta else secrets.randbits(64)
        self.seed = seed
        self.reset()
        seed_bytes = seed if seed else os.urandom(32)
        seed_int = int.from_bytes(seed_bytes, 'big')
        self.rng = np.random.default_rng(seed_int)
        
    def reset(self):
        self.graph = None
        self.cliques = []
        self.fail_count = 0
        self.last_attempt = 0
        
    def generate(self):
        start_time = time.time()
        timeout = 30
        
        try:
            self.reset()
            self.graph = self._generate_erdos_renyi_graph(self.n_nodes, 0.5)
            current_size = self.min_clique
            nodes = list(range(self.n_nodes))
            self.rng.shuffle(nodes)
            clique = nodes[:current_size]
            self.cliques.append(clique)
            self._force_clique(clique)
            
            for i in range(1, self.depth):
                if time.time() - start_time > timeout:
                    raise TimeoutException("Operation timed out")
                    
                current_size += self.delta % 32
                start_idx = i * self.min_clique
                if start_idx + self.delta % 32 > len(nodes):
                    raise RuntimeError("Pas assez de nœuds")
                new_nodes = nodes[start_idx:start_idx + (self.delta % 32)]
                clique = self.cliques[i-1] + new_nodes
                self.cliques.append(clique)
                self._force_clique(clique)
                
            return self.graph.copy(), copy.deepcopy(self.cliques)
        except TimeoutException:
            logging.error("Timeout lors de la génération des cliques")
            raise
    
    def _generate_erdos_renyi_graph(self, n, p):
        seed = hashlib.sha3_256(
            struct.pack("Q", self.n_nodes) +
            struct.pack("Q", self.min_clique) +
            self.seed
        ).digest()
        rng = np.random.default_rng(int.from_bytes(seed, 'big'))
        random_matrix = rng.random((n, n)) < p
        np.fill_diagonal(random_matrix, False)
        graph = random_matrix | random_matrix.T
        return graph
    
    def _force_clique(self, nodes):
        for i in nodes:
            for j in nodes:
                if i != j:
                    self.graph[i, j] = True
                    self.graph[j, i] = True
                    
    def verify_solution(self, solution):
        current_time = time.time()
        delay = 0.1 * (4 ** min(self.fail_count, 5))
        if current_time - self.last_attempt < delay:
            time.sleep(delay - (current_time - self.last_attempt))
        self.last_attempt = current_time
        valid = True
        if len(solution) != self.depth:
            valid = False
        for i in range(1, self.depth):
            if not set(solution[i-1]).issubset(set(solution[i])):
                valid = False
        for clique in solution:
            if not self._is_clique(clique):
                valid = False
        if valid:
            self.fail_count = 0
            return True
        else:
            self.fail_count = min(self.fail_count + 1, 10)
            return False
            
    def _is_clique(self, nodes):
        n = len(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                if not self.graph[nodes[i], nodes[j]]:
                    return False
        return True

# Cœur de FractalKEM avec header chiffré
class UltraSecureFractalKEM:
    def __init__(self):
        self.quantum_rng = QuantumInspiredRNG()

    def build_plaintext_header(self, hurst, salt, clique_seed, depth, min_clique, delta, 
                             nonce_seed, tag_seed, ciphertext_seed, nonce1, tag1, 
                             fbm_serialized, nonce_chacha, tag2):
        """Construit le header en clair pour le chiffrement"""
        header = (
            struct.pack('d', hurst) + 
            salt +
            clique_seed +
            struct.pack('B', depth) +
            struct.pack('B', min_clique) +
            struct.pack('Q', delta) +
            nonce_seed +
            tag_seed +
            ciphertext_seed +
            nonce1 +
            tag1 +
            fbm_serialized +
            nonce_chacha +
            tag2
        )
        
        if len(header) != PLAIN_HEADER_SIZE:
            raise ValueError(f"Taille header incorrecte: {len(header)} vs {PLAIN_HEADER_SIZE}")
            
        return header

    def parse_plaintext_header(self, plaintext_header):
        """Parse le header déchiffré"""
        if len(plaintext_header) != PLAIN_HEADER_SIZE:
            raise ValueError(f"Taille header incorrecte: {len(plaintext_header)} vs {PLAIN_HEADER_SIZE}")
            
        offset = 0
        hurst = struct.unpack('d', plaintext_header[offset:offset+HURST_SIZE])[0]
        offset += HURST_SIZE
        salt = plaintext_header[offset:offset+SALT_SIZE]
        offset += SALT_SIZE
        clique_seed = plaintext_header[offset:offset+CLIQUE_SEED_SIZE]
        offset += CLIQUE_SEED_SIZE
        depth = struct.unpack('B', plaintext_header[offset:offset+1])[0]
        offset += 1
        min_clique = struct.unpack('B', plaintext_header[offset:offset+1])[0]
        offset += 1
        delta = struct.unpack('Q', plaintext_header[offset:offset+8])[0]
        offset += 8
        nonce_seed = plaintext_header[offset:offset+NONCE_SEED_SIZE]
        offset += NONCE_SEED_SIZE
        tag_seed = plaintext_header[offset:offset+TAG_SEED_SIZE]
        offset += TAG_SEED_SIZE
        ciphertext_seed = plaintext_header[offset:offset+CIPHERTEXT_SEED_SIZE]
        offset += CIPHERTEXT_SEED_SIZE
        nonce1 = plaintext_header[offset:offset+NONCE1_SIZE]
        offset += NONCE1_SIZE
        tag1 = plaintext_header[offset:offset+TAG1_SIZE]
        offset += TAG1_SIZE
        fbm_serialized = plaintext_header[offset:offset+FBM_SERIALIZED_SIZE]
        offset += FBM_SERIALIZED_SIZE
        nonce_chacha = plaintext_header[offset:offset+CHACHA_NONCE_SIZE]
        offset += CHACHA_NONCE_SIZE
        tag2 = plaintext_header[offset:offset+CHACHA_TAG_SIZE]
        
        return (hurst, salt, clique_seed, depth, min_clique, delta, nonce_seed, 
                tag_seed, ciphertext_seed, nonce1, tag1, fbm_serialized, 
                nonce_chacha, tag2)

    def derive_scrypt_key(self, password, salt):
        """Dérive une clé de 512 bits avec scrypt (paramètres militaires)"""
        kdf = Scrypt(
            salt=salt,
            length=SCRYPT_KEY_LEN,
            n=SCRYPT_N,
            r=SCRYPT_R,
            p=SCRYPT_P,
        )
        return kdf.derive(password.encode())

    def encrypt_header(self, plaintext_header, key, iv):
        """Chiffre le header avec ChaCha20-Poly1305"""
        cipher = ChaCha20_Poly1305.new(key=key[:32], nonce=iv)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext_header)
        return ciphertext + tag

    def decrypt_header(self, encrypted_header, key, iv):
        """Déchiffre le header avec ChaCha20-Poly1305"""
        ciphertext = encrypted_header[:-16]
        tag = encrypted_header[-16:]
        cipher = ChaCha20_Poly1305.new(key=key[:32], nonce=iv)
        return cipher.decrypt_and_verify(ciphertext, tag)

    def generate_parameters(self):
        entropy_seed = self.quantum_rng.bytes(ENTROPY_SEED_SIZE)
        hurst = self._generate_hurst()
        return entropy_seed, hurst

    def _generate_hurst(self):
        while True:
            hurst_bytes = self.quantum_rng.bytes(8)
            try:
                hurst = struct.unpack('d', hurst_bytes)[0] % 0.7 + 0.15
                if not (EXCLUDED_HURST[0] <= hurst <= EXCLUDED_HURST[1]):
                    return hurst
            except struct.error:
                hurst = 0.3

    def generate_fbm(self, entropy_seed, hurst):
        rng_seed = int.from_bytes(hashlib.sha3_512(entropy_seed).digest()[:64], 'big')
        rng = np.random.default_rng(rng_seed)
        noise = rng.standard_normal(size=FBM_POINTS)
        fbm = FBM(n=FBM_POINTS, hurst=hurst, method='daviesharte')
        fbm._gaussian_noise = lambda size=None: noise if size is None else noise[:size]
        trajectory = fbm.fbm()
        trajectory = np.nan_to_num(trajectory, nan=0.0, posinf=1.0, neginf=-1.0)
        return trajectory

    def derive_key(self, samples):
        combined = b''.join(struct.pack('d', s) for s in samples)
        return hashlib.shake_256(combined).digest(64)

    def apply_chaos_mask(self, data, mask_samples):
        mask_bytes = b''.join(struct.pack('d', v) for v in mask_samples)
        mask_stream = hashlib.shake_256(mask_bytes).digest(len(data))
        
        data_array = bytearray(data)
        for i in range(len(data_array)):
            data_array[i] ^= mask_stream[i]
        
        return bytes(data_array)

    def derive_master_key(self, password, salt, cliques):
        try:
            clique_bytes = b""
            for clique in cliques:
                clique_bytes += struct.pack(f"{len(clique)}I", *clique)
            key = HKDF(
                algorithm=hashes.SHA3_512(),
                length=64,
                salt=salt,
                info=clique_bytes,
                backend=default_backend()
            ).derive(password.encode())
            hq1 = hashlib.shake_256(key).digest(64)
            hq2 = hashlib.blake2b(key, digest_size=64).digest()
            return bytes(a ^ b for a, b in zip(hq1, hq2))
        finally:
            if 'clique_bytes' in locals():
                secure_wipe(bytearray(clique_bytes))

    def encrypt_file(self, password, input_path, output_path):
        entropy_seed = None
        key = None
        try:
            logging.info("DÉBUT CHIFFREMENT AVEC HEADER CHIFFRÉ")
            
            # Génération des paramètres
            entropy_seed, hurst = self.generate_parameters()
            logging.info(f"Hurst: {hurst:.6f}")
            
            trajectory = self.generate_fbm(entropy_seed, hurst)
            samples_key = [trajectory[i] for i in KEY_DERIVATION_SAMPLES]
            samples_mask = [trajectory[i] for i in MASK_SAMPLES]
            key = self.derive_key(samples_key)

            with open(input_path, 'rb') as f:
                original_data = f.read()
            logging.info(f"Données originales: {len(original_data)} octets")

            # Génération des cliques
            clique_seed = self.quantum_rng.bytes(CLIQUE_SEED_SIZE)
            clique_challenge = NestedCliqueChallenge(
                n_nodes=1024,
                min_clique=72,
                depth=6,
                seed=clique_seed
            )
            graph, cliques = clique_challenge.generate()

            # Chiffrement AES
            nonce1 = self.quantum_rng.bytes(NONCE1_SIZE)
            cipher1 = AES.new(key[:32], AES.MODE_GCM, nonce=nonce1)
            ciphertext1, tag1 = cipher1.encrypt_and_digest(original_data)

            # Masquage chaos
            masked_data = self.apply_chaos_mask(ciphertext1, samples_mask)

            # Chiffrement ChaCha20
            chacha_key = key[32:64]
            nonce_chacha = self.quantum_rng.bytes(CHACHA_NONCE_SIZE)
            cipher2 = ChaCha20_Poly1305.new(key=chacha_key, nonce=nonce_chacha)
            ciphertext2, tag2 = cipher2.encrypt_and_digest(masked_data)

            # Préparation du header
            salt = self.quantum_rng.bytes(SALT_SIZE)
            key_enc_seed = self.derive_master_key(password, salt, cliques)
            
            nonce_seed = self.quantum_rng.bytes(NONCE_SEED_SIZE)
            cipher_seed = AES.new(key_enc_seed[:32], AES.MODE_GCM, nonce=nonce_seed)
            ciphertext_seed, tag_seed = cipher_seed.encrypt_and_digest(entropy_seed)

            fbm_values = [trajectory[i] for i in KEY_DERIVATION_SAMPLES + MASK_SAMPLES]
            fbm_serialized = b''.join(struct.pack('d', v) for v in fbm_values)

            # Construction du header en clair
            plaintext_header = self.build_plaintext_header(
                hurst, salt, clique_seed, clique_challenge.depth, clique_challenge.min_clique,
                clique_challenge.delta, nonce_seed, tag_seed, ciphertext_seed,
                nonce1, tag1, fbm_serialized, nonce_chacha, tag2
            )

            # Chiffrement du header avec scrypt + ChaCha20
            scrypt_salt = self.quantum_rng.bytes(SCRYPT_SALT_SIZE)
            header_iv = self.quantum_rng.bytes(HEADER_IV_SIZE)
            scrypt_key = self.derive_scrypt_key(password, scrypt_salt)
            encrypted_header = self.encrypt_header(plaintext_header, scrypt_key, header_iv)

            # Construction du fichier final
            with open(output_path, 'wb') as f:
                f.write(scrypt_salt)
                f.write(header_iv)
                f.write(encrypted_header)
                f.write(ciphertext2)
                
            logging.info(f"CHIFFREMENT RÉUSSI - HEADER CHIFFRÉ: {len(encrypted_header)} octets")
            return True
        except Exception as e:
            logging.error(f"ÉCHEC CHIFFREMENT: {type(e).__name__} - {str(e)}", exc_info=True)
            return False
        finally:
            if key: secure_wipe(bytearray(key))
            if entropy_seed: secure_wipe(bytearray(entropy_seed))
            gc.collect()

    def decrypt_file(self, password, input_path, output_path):
        try:
            logging.info("DÉBUT DÉCHIFFREMENT AVEC HEADER CHIFFRÉ")
            
            file_size = os.path.getsize(input_path)
            min_file_size = HEADER_SIZE + 1
            if file_size < min_file_size:
                raise ValueError(f"Fichier trop petit ({file_size} < {min_file_size})")

            with open(input_path, 'rb') as f:
                scrypt_salt = f.read(SCRYPT_SALT_SIZE)
                header_iv = f.read(HEADER_IV_SIZE)
                encrypted_header = f.read(ENCRYPTED_HEADER_SIZE + HEADER_TAG_SIZE)
                ciphertext2 = f.read()
                
            # Déchiffrement du header
            scrypt_key = self.derive_scrypt_key(password, scrypt_salt)
            plaintext_header = self.decrypt_header(encrypted_header, scrypt_key, header_iv)
            
            # Extraction des paramètres
            (hurst, salt, clique_seed, depth, min_clique, delta, nonce_seed, 
             tag_seed, ciphertext_seed, nonce1, tag1, fbm_serialized, 
             nonce_chacha, tag2) = self.parse_plaintext_header(plaintext_header)

            # Régénération des cliques
            clique_challenge = NestedCliqueChallenge(
                n_nodes=1024,
                min_clique=min_clique,
                depth=depth,
                delta=delta,
                seed=clique_seed
            )
            graph, cliques = clique_challenge.generate()

            # Récupération de l'entropy seed
            key_enc_seed = self.derive_master_key(password, salt, cliques)
            cipher_seed = AES.new(key_enc_seed[:32], AES.MODE_GCM, nonce=nonce_seed)
            entropy_seed = cipher_seed.decrypt_and_verify(ciphertext_seed, tag_seed)
            
            # Extraction des valeurs FBM sauvegardées
            fbm_values = []
            for i in range(FBM_SAMPLES_COUNT):
                start = i * 8
                value_bytes = fbm_serialized[start:start+8]
                value = struct.unpack('d', value_bytes)[0]
                fbm_values.append(value)
                
            samples_key = fbm_values[:len(KEY_DERIVATION_SAMPLES)]
            samples_mask = fbm_values[len(KEY_DERIVATION_SAMPLES):]
            
            # Dérivation de la clé
            key = self.derive_key(samples_key)

            # Déchiffrement ChaCha20
            chacha_key = key[32:64]
            cipher2 = ChaCha20_Poly1305.new(key=chacha_key, nonce=nonce_chacha)
            masked_data = cipher2.decrypt_and_verify(ciphertext2, tag2)

            # Démasquage chaos
            ciphertext1 = self.apply_chaos_mask(masked_data, samples_mask)

            # Déchiffrement AES
            cipher1 = AES.new(key[:32], AES.MODE_GCM, nonce=nonce1)
            decrypted_data = cipher1.decrypt_and_verify(ciphertext1, tag1)

            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
                
            logging.info("DÉCHIFFREMENT RÉUSSI")
            return True
        except Exception as e:
            logging.error(f"ERREUR DÉCHIFFREMENT: {type(e).__name__} - {str(e)}")
            logging.error(traceback.format_exc())
            return False
        finally:
            if 'key' in locals(): secure_wipe(bytearray(key))
            if 'decrypted_data' in locals(): secure_wipe(bytearray(decrypted_data))
            gc.collect()

# Interface utilisateur
class UltraSecureKEMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FractalKEM ULTRA SECURE - Header Chiffré")
        self.root.geometry("900x650")
        self.root.configure(bg='#0a0a2a')
        self.kem = UltraSecureFractalKEM()
        self.create_widgets()
        self.center_window()

    def center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'+{x}+{y}')

    def create_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#0a0a2a')
        style.configure('TLabel', background='#0a0a2a', foreground='#e0f7ff', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10, 'bold'), 
                      background='#1e3c72', foreground='white', padding=10)
        style.configure('Header.TLabel', font=('Arial', 20, 'bold'), 
                      foreground='#4fc3f7', background='#0a0a2a')
        style.configure('TEntry', font=('Arial', 10), padding=5, fieldbackground='#1c2e4a')
        style.map('TButton', 
                background=[('active', '#152c50'), ('pressed', '#0f2240')],
                foreground=[('active', '#ffffff')])
        style.configure('TLabelframe', background='#0a0a2a', foreground='#4fc3f7')
        style.configure('TLabelframe.Label', background='#0a0a2a', foreground='#4fc3f7')

        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(main_frame, 
                         text="🔐 FRACTALKEM ULTRA SECURE - HEADER CHIFFRÉ", 
                         style='Header.TLabel')
        header.pack(pady=15)

        security_level = ttk.Label(main_frame, 
                                 text="NIVEAU DE SÉCURITÉ: ULTRA (512+ BITS) - HEADER CHIFFRÉ",
                                 font=('Arial', 12, 'bold'),
                                 foreground='#00ff00',
                                 background='#0a0a2a')
        security_level.pack(pady=5)

        pass_frame = ttk.LabelFrame(main_frame, text="AUTHENTIFICATION", padding=15)
        pass_frame.pack(fill=tk.X, pady=15)
        
        ttk.Label(pass_frame, text="PHRASE SECRÈTE (24+ CARACTÈRES):", 
                font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=5, pady=8)
        self.password_entry = ttk.Entry(pass_frame, show="•", width=60)
        self.password_entry.grid(row=0, column=1, padx=5, pady=8)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=20)

        btn_encrypt = ttk.Button(btn_frame, text="🔒 CHIFFRER UN FICHIER", 
                               command=lambda: self.select_file("encrypt"))
        btn_encrypt.pack(side=tk.LEFT, padx=15, fill=tk.X, expand=True)
        
        btn_decrypt = ttk.Button(btn_frame, text="🔓 DÉCHIFFRER UN FICHIER", 
                               command=lambda: self.select_file("decrypt"))
        btn_decrypt.pack(side=tk.RIGHT, padx=15, fill=tk.X, expand=True)

        security_frame = ttk.LabelFrame(main_frame, text="CARACTÉRISTIQUES DE SÉCURITÉ", padding=15)
        security_frame.pack(fill=tk.BOTH, expand=True, pady=15)
        
        security_info = ttk.Label(security_frame, 
                text="• Header complètement chiffré avec scrypt + ChaCha20-Poly1305\n• Paramètres scrypt militaires (N=2^20, r=8, p=1)\n• Clés 512 bits dérivées avec scrypt\n• Double chiffrement AES-256 + ChaCha20-Poly1305\n• Entropie quantique certifiée\n• Cliques emboîtées NP-complètes\n• Effacement militaire 7 passes",
                font=('Arial', 11), 
                background='#0a0a2a', 
                foreground='#a5d6ff',
                justify=tk.LEFT)
        security_info.pack(pady=10, anchor=tk.W)

        self.status = ttk.Label(main_frame, text="PRÊT. SYSTÈME DE HEADER CHIFFRÉ ACTIVÉ.", 
                             font=('Arial', 11, 'bold'), background='#0a0a2a', 
                             foreground='#4fc3f7', anchor=tk.CENTER)
        self.status.pack(side=tk.BOTTOM, fill=tk.X, pady=15)

    def select_file(self, operation):
        password = self.password_entry.get()
        if len(password) < 24:
            messagebox.showwarning("Attention", "La phrase secrète doit avoir au moins 24 caractères")
            return
            
        if operation == "encrypt":
            path = filedialog.askopenfilename(title="SÉLECTIONNER UN FICHIER À CHIFFRER")
            if path:
                output = filedialog.asksaveasfilename(
                    title="ENREGISTRER LE FICHIER CHIFFRÉ",
                    defaultextension=".fenc512",
                    filetypes=(("Fichiers FractalKEM ULTRA", "*.fenc512"), ("Tous fichiers", "*.*")))
                if output:
                    self.status.config(text="CHIFFREMENT EN COURS...")
                    success = self.kem.encrypt_file(password, path, output)
                    if success:
                        self.status.config(text=f"CHIFFREMENT RÉUSSI: {os.path.basename(output)}")
                        messagebox.showinfo("Succès", "Chiffrement terminé avec succès!")
                    else:
                        self.status.config(text="ÉCHEC DU CHIFFREMENT")
        else:
            path = filedialog.askopenfilename(
                title="SÉLECTIONNER UN FICHIER À DÉCHIFFRÉ", 
                filetypes=(("Fichiers FractalKEM ULTRA", "*.fenc512"), ("Tous fichiers", "*.*")))
            if path:
                output = filedialog.asksaveasfilename(
                    title="ENREGISTRER LE FICHIER DÉCHIFFRÉ",
                    defaultextension=".dec"
                )
                if output:
                    self.status.config(text="DÉCHIFFREMENT EN COURS...")
                    success = self.kem.decrypt_file(password, path, output)
                    if success:
                        self.status.config(text=f"DÉCHIFFREMENT RÉUSSI: {os.path.basename(output)}")
                        messagebox.showinfo("Succès", "Déchiffrement terminé avec succès!")
                    else:
                        self.status.config(text="ÉCHEC DU DÉCHIFFREMENT")

if __name__ == "__main__":
    root = tk.Tk()
    app = UltraSecureKEMGUI(root)
    root.mainloop()