import numpy as np


def create_final_permutation():
    initial_permutation = [
        [152, 142, 132, 122, 112, 102, 92, 82, 72, 62, 52, 42, 32, 22, 12, 2],
        [154, 144, 134, 124, 114, 104, 94, 84, 74, 64, 54, 44, 34, 24, 14, 4],
        [156, 146, 136, 126, 116, 106, 96, 86, 76, 66, 56, 46, 36, 26, 16, 6],
        [158, 148, 138, 128, 118, 108, 98, 88, 78, 68, 58, 48, 38, 28, 18, 8],
        [160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
        [151, 141, 131, 121, 111, 101, 91, 81, 71, 61, 51, 41, 31, 21, 11, 1],
        [153, 143, 133, 123, 113, 103, 93, 83, 73, 63, 53, 43, 33, 23, 13, 3],
        [155, 145, 135, 125, 115, 105, 95, 85, 75, 65, 55, 45, 35, 25, 15, 5],
        [157, 147, 137, 127, 117, 107, 97, 87, 77, 67, 57, 47, 37, 27, 17, 7],
        [159, 149, 139, 129, 119, 109, 99, 89, 79, 69, 59, 49, 39, 29, 19, 9]
    ]
    final_permutation = np.zeros(shape=(10, 16)).astype(int)
    for i in range(1, 161):
        for j in range(10):
            for k in range(16):
                if initial_permutation[j][k] == i:
                    number = j * 16 + (k + 1)
                    row = int((i - 1) / 16)
                    if i % 16 == 0:
                        column = 15
                    else:
                        column = (i % 16) - 1
                    final_permutation[row][column] = number
    return final_permutation


class SerpentPlus:
    def __init__(self):
        self.IPTable = [
            152, 142, 132, 122, 112, 102, 92, 82, 72, 62, 52, 42, 32, 22, 12, 2,
            154, 144, 134, 124, 114, 104, 94, 84, 74, 64, 54, 44, 34, 24, 14, 4,
            156, 146, 136, 126, 116, 106, 96, 86, 76, 66, 56, 46, 36, 26, 16, 6,
            158, 148, 138, 128, 118, 108, 98, 88, 78, 68, 58, 48, 38, 28, 18, 8,
            160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10,
            151, 141, 131, 121, 111, 101, 91, 81, 71, 61, 51, 41, 31, 21, 11, 1,
            153, 143, 133, 123, 113, 103, 93, 83, 73, 63, 53, 43, 33, 23, 13, 3,
            155, 145, 135, 125, 115, 105, 95, 85, 75, 65, 55, 45, 35, 25, 15, 5,
            157, 147, 137, 127, 117, 107, 97, 87, 77, 67, 57, 47, 37, 27, 17, 7,
            159, 149, 139, 129, 119, 109, 99, 89, 79, 69, 59, 49, 39, 29, 19, 9
        ]
        self.FPTable = [
            96, 16, 112, 32, 128, 48, 144, 64, 160, 80, 95, 15, 111, 31, 127, 47,
            143, 63, 159, 79, 94, 14, 110, 30, 126, 46, 142, 62, 158, 78, 93, 13,
            109, 29, 125, 45, 141, 61, 157, 77, 92, 12, 108, 28, 124, 44, 140, 60,
            156, 76, 91, 11, 107, 27, 123, 43, 139, 59, 155, 75, 90, 10, 106, 26,
            122, 42, 138, 58, 154, 74, 89, 9, 105, 25, 121, 41, 137, 57, 153, 73,
            88, 8, 104, 24, 120, 40, 136, 56, 152, 72, 87, 7, 103, 23, 119, 39,
            135, 55, 151, 71, 86, 6, 102, 22, 118, 38, 134, 54, 150, 70, 85, 5,
            101, 21, 117, 37, 133, 53, 149, 69, 84, 4, 100, 20, 116, 36, 132, 52,
            148, 68, 83, 3, 99, 19, 115, 35, 131, 51, 147, 67, 82, 2, 98, 18,
            114, 34, 130, 50, 146, 66, 81, 1, 97, 17, 113, 33, 129, 49, 145, 65
        ]
        self.SBoxDecimalTable = [
            # Hummingbird2
            [7, 12, 14, 9, 2, 1, 5, 15, 11, 6, 13, 0, 4, 8, 10, 3],  # S0
            [4, 10, 1, 6, 8, 15, 7, 12, 3, 0, 14, 13, 5, 9, 11, 2],  # S1
            [2, 15, 12, 1, 5, 6, 10, 13, 14, 8, 3, 4, 0, 11, 9, 7],  # S2
            [15, 4, 5, 8, 9, 7, 2, 1, 10, 3, 0, 14, 6, 12, 13, 11],  # S3
            # Hummingbird1
            [8, 6, 5, 15, 1, 12, 10, 9, 14, 11, 2, 4, 7, 0, 13, 3],  # S4
            [0, 7, 14, 1, 5, 11, 8, 2, 3, 10, 13, 6, 15, 12, 4, 9],  # S5
            [2, 14, 15, 5, 12, 1, 9, 10, 11, 4, 6, 8, 0, 7, 3, 13],  # S6
            [0, 7, 3, 4, 12, 1, 10, 15, 13, 14, 6, 11, 2, 8, 9, 5],  # S7
        ]
        self.keyNumber = 33
        self.roundNumber = 32
        self.r_con = ['00000001', '00000010', '00000100', '00001000', '00010000', '00100000', '01000000', '10000000',
                      '00011011', '00110110', '01101100', '11011000', '10101011', '01001101', '10011010', '00101111',
                      '01011110', '10111100', '01100011', '11000110', '10010111', '00110101', '01101010', '11010100',
                      '10110011', '01111101', '11111010', '11101111', '11000101', '10010001', '00111001']
        pi = "243F6A8885"
        square2 = "6A09E667F3"
        e = "93C467E37D"
        golden = "9E3779B97F"
        self.pi_binary = bin(int(pi, 16))[2:].zfill(40)
        self.square2_binary = bin(int(square2, 16))[2:].zfill(40)
        self.e_binary = bin(int(e, 16))[2:].zfill(40)
        self.golden_binary = bin(int(golden, 16))[2:].zfill(40)

        self.SBoxBitstring = []
        self.SBoxBitstringInverse = []
        for line in self.SBoxDecimalTable:
            dict = {}
            inverseDict = {}
            for i in range(len(line)):
                index = self.bitstring(i, 4)
                value = self.bitstring(line[i], 4)
                dict[index] = value
                inverseDict[value] = index
            self.SBoxBitstring.append(dict)
            self.SBoxBitstringInverse.append(inverseDict)

    def bitstring(self, n, minlen=1):
        """Translate n from integer to bitstring, padding it with 0s as
        necessary to reach the minimum length 'minlen'. 'n' must be >= 0 since
        the bitstring format is undefined for negative integers.  Note that,
        while the bitstring format can represent arbitrarily large numbers,
        this is not so for Python's normal integer type: on a 32-bit machine,
        values of n >= 2^31 need to be expressed as python long integers or
        they will "look" negative and won't work. E.g. 0x80000000 needs to be
        passed in as 0x80000000L, or it will be taken as -2147483648 instead of
        +2147483648L.

        EXAMPLE: bitstring(10, 8) -> "01010000"
        """

        if minlen < 1:
            raise ValueError("a bitstring must have at least 1 char")
        if n < 0:
            raise ValueError("bitstring representation undefined for neg numbers")

        result = ""
        while n > 0:
            if n & 1:
                result = result + "1"
            else:
                result = result + "0"
            n = n >> 1
        if len(result) < minlen:
            result = result + "0" * (minlen - len(result))
        return result

    def change_rcon(self):
        r_con = [
            "0x01", "0x02", "0x04", "0x08", "0x10", "0x20", "0x40",
            "0x80", "0x1B", "0x36", "0x6C", "0xD8", "0xAB", "0x4D", "0x9A",
            "0x2F", "0x5E", "0xBC", "0x63", "0xC6", "0x97", "0x35", "0x6A",
            "0xD4", "0xB3", "0x7D", "0xFA", "0xEF", "0xC5", "0x91", "0x39",
        ]
        new_r_con = []
        for i in range(len(r_con)):
            new_r_con.append(bin(int(str(r_con[i]), 16))[2:].zfill(8))
        return new_r_con

    def applyPermutation(self, permutationTable, input):
        """Apply the permutation specified by the 160-element list
        'permutationTable' to the 160-bit bitstring 'input' and return a
        160-bit bitstring as the result."""

        if len(input) != len(permutationTable):
            raise ValueError("input size (%d) doesn't match perm table size (%d)" \
                             % (len(input), len(permutationTable)))

        result = ""
        for i in range(len(permutationTable)):
            result = result + input[permutationTable[i] - 1]
        return result

    def IP(self, input):
        """Apply the Initial Permutation to the 160-bit bitstring 'input'
        and return a 160-bit bitstring as the result."""

        return self.applyPermutation(self.IPTable, input)

    def FP(self, input):
        """Apply the Final Permutation to the 128-bit bitstring 'input'
        and return a 128-bit bitstring as the result."""

        return self.applyPermutation(self.FPTable, input)

    def IPInverse(self, output):
        """Apply the Initial Permutation in reverse."""

        return self.FP(output)

    def FPInverse(self, output):
        """Apply the Final Permutation in reverse."""

        return self.IP(output)

    def S(self, box, input):
        """Apply S-box number 'box' to 4-bit bitstring 'input' and return a
        4-bit bitstring as the result."""

        return self.SBoxBitstring[box % 8][input]
        # There used to be 32 different S-boxes in serpent-0. Now there are
        # only 8, each of which is used 4 times (Sboxes 8, 16, 24 are all
        # identical to Sbox 0, etc). Hence the %8.

    def SInverse(self, box, output):
        """Apply S-box number 'box' in reverse to 4-bit bitstring 'output' and
        return a 4-bit bitstring (the input) as the result."""

        return self.SBoxBitstringInverse[box % 8][output]

    def SHat(self, box, input):
        """Apply a parallel array of 40 copies of S-box number 'box' to the
        160-bit bitstring 'input' and return a 160-bit bitstring as the
        result."""

        result = ""
        for i in range(40):
            result = result + self.S(box, input[4 * i:4 * (i + 1)])
        return result

    def SHatInverse(self, box, output):
        """Apply, in reverse, a parallel array of 40 copies of S-box number
        'box' to the 160-bit bitstring 'output' and return a 160-bit bitstring
        (the input) as the result."""

        result = ""
        for i in range(40):
            result = result + self.SInverse(box, output[4 * i:4 * (i + 1)])
        return result

    def binaryXor(self, n1, n2):
        """Return the xor of two bitstrings of equal length as another
        bitstring of the same length.

        EXAMPLE: binaryXor("10010", "00011") -> "10001"
        """

        if len(n1) != len(n2):
            raise ValueError("can't xor bitstrings of different " + \
                             "lengths (%d and %d)" % (len(n1), len(n2)))
        # We assume that they are genuine bitstrings instead of just random
        # character strings.

        result = ""
        for i in range(len(n1)):
            if n1[i] == n2[i]:
                result = result + "0"
            else:
                result = result + "1"
        return result

    def bitstring2matrix(self, text):
        """ Converts a 160 bitstring into a 5x4 matrix.  """
        one_dimension_list = [text[i:i + 8] for i in range(0, len(text), 8)]
        return [list(one_dimension_list[i:i + 4]) for i in range(0, len(one_dimension_list), 4)]

    def get_n_column(self, matrix, number):
        return [matrix[i][number] for i in range(len(matrix))][:]

    def put_column_to_matrix(self, matrix, column):
        for i in range(len(matrix)):
            matrix[i].append(column[i])
        return matrix

    def sbox_key_scheduling(self, input, box):
        return self.SBoxBitstring[box % 8][input[:4]] + self.SBoxBitstring[box % 8][input[4:]]

    def get_key_from_key_columns(self, matrix, number):
        return_key = ""
        for i in range(len(matrix)):
            for bitstring in matrix[i][number * 4: number * 4 + 4]:
                return_key += bitstring
        return return_key

    def key_scheduling(self, master_key):
        """
        Expands and returns a list of key bitstring for the given master_key.
        """
        key_columns = self.bitstring2matrix(master_key)
        counter = 0
        while counter < self.keyNumber:
            last_column = self.get_n_column(key_columns, -1)
            if len(key_columns[0]) % 4 == 0:
                last_column.append(last_column.pop(0))
                last_column = [self.sbox_key_scheduling(binary, counter) for binary in last_column]
                last_column[0] = self.binaryXor(last_column[0], self.r_con[counter % len(self.r_con)])
                before_column = self.get_n_column(key_columns, -4)
                for i in range(len(last_column)):
                    last_column[i] = self.binaryXor(last_column[i], before_column[i])
                key_columns = self.put_column_to_matrix(key_columns, last_column)
                counter += 1
            else:
                before_column = self.get_n_column(key_columns, -4)
                for i in range(len(last_column)):
                    last_column[i] = self.binaryXor(last_column[i], before_column[i])
                key_columns = self.put_column_to_matrix(key_columns, last_column)
        return [self.get_key_from_key_columns(key_columns, i) for i in range(self.keyNumber)]

    def leftRotate(self, n, d):
        if d == 0:
            return n
        return n[d:] + n[:d]

    def rightRotate(self, n, d):
        if d == 0:
            return n
        return n[-d:] + n[:len(n) - d]

    def znotfill(self, input):
        if len(input) != 40:
            excessive_bits = len(input) - 40
            return input[excessive_bits:]
        return input

    def linear_transformation(self, input):
        A = input[:40]
        B = input[40:80]
        C = input[80:120]
        D = input[120:160]

        modulo = 2 ** 40

        B = (int(B, 2) + int(self.pi_binary, 2)) % modulo
        D = (int(D, 2) + int(self.square2_binary, 2)) % modulo

        t = self.leftRotate(self.znotfill(bin((B * (2 * B + 1)) % modulo)[2:].zfill(40)), 6)
        u = self.leftRotate(self.znotfill(bin((D * (2 * D + 1)) % modulo)[2:].zfill(40)), 6)

        tmod = int(t, 2) % 40
        umod = int(u, 2) % 40

        A = self.znotfill(
            bin((int(self.leftRotate(self.binaryXor(A, t), umod), 2) + int(self.e_binary, 2)) % modulo)[2:].zfill(40))
        C = self.znotfill(
            bin((int(self.leftRotate(self.binaryXor(C, u), tmod), 2) + int(self.golden_binary, 2)) % modulo)[2:].zfill(
                40))
        B = self.znotfill(bin(B)[2:].zfill(40))
        D = self.znotfill(bin(D)[2:].zfill(40))
        A, B, C, D = B, C, D, A

        A = self.znotfill(bin((int(A, 2) + int(self.pi_binary, 2)) % modulo)[2:].zfill(40))
        C = self.znotfill(bin((int(C, 2) + int(self.square2_binary, 2)) % modulo)[2:].zfill(40))

        return A + B + C + D

    def linear_transformation_inverse(self, input):
        A = input[:40]
        B = input[40:80]
        C = input[80:120]
        D = input[120:160]

        modulo = 2 ** 40

        A = self.znotfill(bin((int(A, 2) - int(self.pi_binary, 2)) % modulo)[2:].zfill(40))
        C = self.znotfill(bin((int(C, 2) - int(self.square2_binary, 2)) % modulo)[2:].zfill(40))

        A, B, C, D = D, A, B, C

        u = self.leftRotate(self.znotfill(bin((int(D, 2) * (2 * int(D, 2) + 1)) % modulo)[2:].zfill(40)), 6)
        t = self.leftRotate(self.znotfill(bin((int(B, 2) * (2 * int(B, 2) + 1)) % modulo)[2:].zfill(40)), 6)

        tmod = int(t, 2) % 40
        umod = int(u, 2) % 40

        C = self.binaryXor(
            self.rightRotate(self.znotfill(bin((int(C, 2) - int(self.golden_binary, 2)) % modulo)[2:].zfill(40)), tmod),
            u)
        A = self.binaryXor(
            self.rightRotate(self.znotfill(bin((int(A, 2) - int(self.e_binary, 2)) % modulo)[2:].zfill(40)), umod), t)

        B = self.znotfill(bin((int(B, 2) - int(self.pi_binary, 2)) % modulo)[2:].zfill(40))
        D = self.znotfill(bin((int(D, 2) - int(self.square2_binary, 2)) % modulo)[2:].zfill(40))

        return A + B + C + D

    def R(self, i, BHati, KHat):
        """Apply round 'i' to the 160-bit bitstring 'BHati', returning another
        160-bit bitstring (conceptually BHatiPlus1). Do this using the
        appropriately numbered subkey(s) from the 'KHat' list of 33 160-bit
        bitstrings."""

        xored = self.binaryXor(BHati, KHat[i])

        SHati = self.SHat(i, xored)

        if 0 <= i <= self.roundNumber - 2:
            BHatiPlus1 = self.linear_transformation(SHati)
        elif i == self.roundNumber - 1:
            BHatiPlus1 = self.binaryXor(SHati, KHat[self.roundNumber])
        else:
            raise ValueError("round %d is out of 0..%d range" % (i, self.roundNumber - 1))

        return BHatiPlus1

    def RInverse(self, i, BHatiPlus1, KHat):
        """Apply round 'i' in reverse to the 160-bit bitstring 'BHatiPlus1',
        returning another 160-bit bitstring (conceptually BHati). Do this using
        the appropriately numbered subkey(s) from the 'KHat' list of 33 160-bit
        bitstrings."""

        if 0 <= i <= self.roundNumber - 2:
            SHati = self.linear_transformation_inverse(BHatiPlus1)
        elif i == self.roundNumber - 1:
            SHati = self.binaryXor(BHatiPlus1, KHat[self.roundNumber])
        else:
            raise ValueError("round %d is out of 0..%d range" % (i, self.roundNumber - 1))

        xored = self.SHatInverse(i, SHati)

        BHati = self.binaryXor(xored, KHat[i])

        return BHati

    def encrypt(self, plainText, userKey):
        """Encrypt the 160-bit bitstring 'plainText' with the 160-bit bitstring
        'userKey', using the normal algorithm, and return a 160-bit ciphertext
        bitstring."""

        KHat = self.key_scheduling(userKey)
        BHat = self.IP(plainText)  # BHat_0 at this stage
        for i in range(self.roundNumber):
            BHat = self.R(i, BHat, KHat)  # Produce BHat_i+1 from BHat_i
        # BHat is now _32 i.e. _r
        C = self.FP(BHat)
        return C

    def encrypt_full_round(self, plainText, userKey):
        """Encrypt the 160-bit bitstring 'plainText' with the 160-bit bitstring
        'userKey', using the normal algorithm, and return a 160-bit ciphertext
        bitstring."""

        KHat = self.key_scheduling(userKey)
        BHat = self.IP(plainText)  # BHat_0 at this stage
        round_output_list = []
        for i in range(self.roundNumber):
            BHat = self.R(i, BHat, KHat)  # Produce BHat_i+1 from BHat_i
            round_output_list.append(BHat)
        # BHat is now _32 i.e. _r
        C = self.FP(BHat)
        round_output_list.append(C)
        return round_output_list

    def decrypt(self, cipherText, userKey):
        """Decrypt the 160-bit bitstring 'cipherText' with the 160-bit
        bitstring 'userKey', using the normal algorithm, and return a 160-bit
        plaintext bitstring."""

        KHat = self.key_scheduling(userKey)
        BHat = self.FPInverse(cipherText)  # BHat_r at this stage
        for i in range(self.roundNumber - 1, -1, -1):  # from r-1 down to 0 included
            BHat = self.RInverse(i, BHat, KHat)  # Produce BHat_i from BHat_i+1
        # BHat is now _0
        plainText = self.IPInverse(BHat)
        return plainText
