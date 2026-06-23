import bchlib
import hashlib
import os
import numpy as np

# Initialize BCH
m = 9       # GF(2^9)
t = 30      # correct up to 30 bit errors
bch = bchlib.BCH(t=t, m=m)

def fuzzy_generate_reusable(input_bytes: bytearray):
    """Generate reusable secret R, helper, and salt from input bytes."""
    # 1. Generate reusable random secret R
    R = os.urandom(len(input_bytes))

    # 2. Encode R with BCH
    ecc = bch.encode(bytearray(R))
    codeword = bytearray(R) + ecc

    # Pad input_bytes to match codeword length
    padded_input = input_bytes + bytes(len(ecc))

    # Helper is full-length XOR
    helper = bytearray(a ^ b for a, b in zip(padded_input, codeword))


    # 4. Pick a random salt for key derivation
    salt = os.urandom(16)

    # 5. Derive application key
    application_key = hashlib.sha256(R + salt).digest()

    return application_key, helper, salt


def fuzzy_reproduce_reusable(noisy_bytes: bytearray, helper: bytearray, salt: bytes):
    """Reproduce application key from noisy bytes, helper, and salt."""
    codeword_len = len(helper)
    padded_noisy = noisy_bytes + bytes(codeword_len - len(noisy_bytes))

    # 2. Reconstruct noisy codeword
    noisy_codeword = bytearray(a ^ b for a, b in zip(padded_noisy, helper))

    # 3. Separate data and ecc
    data = noisy_codeword[:-bch.ecc_bytes]
    ecc = noisy_codeword[-bch.ecc_bytes:]

    # 3. Decode
    nerr = bch.decode(data, ecc)
    if nerr >= 0:
        bch.correct(data, ecc)
        R = bytes(data)

        # 4. Derive key with same salt
        return hashlib.sha256(R + salt).digest()
    return None

def fuzzy_revocation(input_bytes: bytearray):
    # 1. Generate reusable random secret R
    R = os.urandom(len(input_bytes))

    # 2. Encode R with BCH
    ecc = bch.encode(bytearray(R))
    codeword = bytearray(R) + ecc

    # Pad input_bytes to match codeword length
    padded_input = input_bytes + bytes(len(ecc))

    # Helper is full-length XOR
    helper = bytearray(a ^ b for a, b in zip(padded_input, codeword))


    # 4. Pick a random salt for key derivation
    salt = os.urandom(16)

    # 5. Derive application key
    application_key = hashlib.sha256(R + salt).digest()

    return application_key, helper, salt
