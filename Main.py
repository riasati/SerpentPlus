import argparse

from Avalanche import create_round_results
from SerpentPlus import SerpentPlus
from pprint import pprint


def convert_string_to_bitstream(string):
    str_encoded = bytes(string, 'UTF-8')
    return_bitstream = ""
    for one_byte in str_encoded:
        return_bitstream += bin(one_byte)[2:].zfill(8)
    if len(return_bitstream) % 160 != 0:
        return_bitstream += "01111000" * ((160 - (len(return_bitstream) % 160)) // 8)
        if (160 - (len(return_bitstream) % 160)) != 0:
            if (160 - (len(return_bitstream) % 160)) == 1:
                return_bitstream += "0"
            elif (160 - (len(return_bitstream) % 160)) == 2:
                return_bitstream += "01"
            elif (160 - (len(return_bitstream) % 160)) == 3:
                return_bitstream += "011"
            elif (160 - (len(return_bitstream) % 160)) == 4:
                return_bitstream += "0111"
            elif (160 - (len(return_bitstream) % 160)) == 5:
                return_bitstream += "01111"
            elif (160 - (len(return_bitstream) % 160)) == 6:
                return_bitstream += "011110"
            elif (160 - (len(return_bitstream) % 160)) == 7:
                return_bitstream += "0111100"

    return [return_bitstream[i: i + 160] for i in range(0, len(return_bitstream), 160)]


def convert_bitstream_to_string(bitstream_list):
    complete_bitstream = ""
    for bitstream in bitstream_list:
        complete_bitstream += bitstream
    ascii_numbers = [int(complete_bitstream[i: i + 8], 2) for i in range(0, len(complete_bitstream), 8)]
    return bytes(ascii_numbers).decode("UTF-8", "ignore")


def encrypt_plain_text(plain_text, key):
    serpent = SerpentPlus()
    bitstream_list = convert_string_to_bitstream(plain_text)
    encrypted = [serpent.encrypt(bitstream_list[i], key) for i in range(len(bitstream_list))]
    cipher_text = convert_bitstream_to_string(encrypted)
    return cipher_text, encrypted


def decrypt_cipher_text(cipher_text, encrypted, key):
    serpent = SerpentPlus()
    decrypted = [serpent.decrypt(encrypted[i], key) for i in range(len(encrypted))]
    plaintext = convert_bitstream_to_string(decrypted)
    return plaintext


def encrypt_plain_text_binary(plain_text, key):
    serpent = SerpentPlus()
    cipher_text = serpent.encrypt(plain_text, key)
    return cipher_text


def decrypt_cipher_text_binary(cipher_text, key):
    serpent = SerpentPlus()
    plain_text = serpent.decrypt(cipher_text, key)
    return plain_text


def encrypt_input_file(input_file, output_file, key):

    def bin_to_hexa(binary):
        num = int(binary, 2)
        hex_num = hex(num)
        return hex_num[2:]

    def hexa_to_bin(hexa, size_number=160):
        return bin(int(hexa, 16))[2:].zfill(size_number)

    serpent = SerpentPlus()
    with open(input_file, "r") as read_file:
        lines = read_file.readlines()
        encrypted = [serpent.encrypt(hexa_to_bin(lines[i]), key) for i in range(len(lines))]
        with open(output_file, "w") as write_file:
            [print(f'{bin_to_hexa(encrypt)}', file=write_file) for encrypt in encrypted]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Serpent Plus")

    parser.add_argument("--mode", "-m",
                        help=" number of mode that code will run: \r\n1- encrypt plain text data \r\n2- encrypt plain text binary data\r\n3- encrypt input file data \r\n4- create avalanche for each round")
    parser.add_argument("--key", "-k",
                        help="binary key of serpent plus",
                        default="1011001110011000011001010011011101111110001111001110101101011111000101111010000100111111000000011000010101111110000111111011111100000000001101000000101101101001")
    parser.add_argument("--plaintext", "-p",
                        help="plaintext of serpent plus (required for mode 1)")
    parser.add_argument("--plaintext_binary", "-b",
                        help="plaintext in binary format (required for mode 2)")
    parser.add_argument("--input_plaintext_file", "-i",
                        help="address of input file", default="input.txt")
    parser.add_argument("--output_ciphertext_file", "-o",
                        help="address of output file", default="output.txt")

    args = parser.parse_args()

    if int(args.mode) == 1:
        cipher_text, encrypted_bitstream = encrypt_plain_text(args.plaintext, args.key)
        plain_text = decrypt_cipher_text(cipher_text, encrypted_bitstream, args.key)
        pprint({
            "input plain text": args.plaintext,
            "input key": args.key,
            "output cipher text": cipher_text,
            "output plain text": plain_text
        })
    elif int(args.mode) == 2:
        cipher_text = encrypt_plain_text_binary(args.plaintext_binary, args.key)
        plain_text = decrypt_cipher_text_binary(cipher_text, args.key)
        pprint({
            "input plain text binary": args.plaintext_binary,
            "input key": args.key,
            "output cipher text binary": cipher_text,
            "output plain text binary": plain_text
        })
    elif int(args.mode) == 3:
        encrypt_input_file(args.input_plaintext_file, args.output_ciphertext_file, args.key)
    elif int(args.mode) == 4:
        create_round_results(args.key, args.input_plaintext_file)
