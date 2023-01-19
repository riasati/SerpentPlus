import numpy as np
from SerpentPlus import SerpentPlus
from tqdm import tqdm


def bin_to_hexa(binary):
    num = int(binary, 2)
    hex_num = hex(num)
    return hex_num


def hexa_to_bin(hexa, size_number=160):
    return bin(int(hexa, 16))[2:].zfill(size_number)


def create_random_number(size_number):
    random_list = np.random.randint(2, size=(size_number,))
    string_random = "".join(random_list.astype(str))
    hexa_random = bin_to_hexa(string_random).upper()[2:]
    return hexa_random


def create_hex_random_file(filename, random_size):
    with open(filename, 'w') as f:
        for i in range(500):
            hex_random = create_random_number(random_size)
            print(f'{hex_random}', file=f)


def create_binary_random_file(filename, filename2):
    with open(filename, 'r') as f:
        lines = f.readlines()
        binary_list = [hexa_to_bin(lines[i]) for i in range(len(lines))]
    with open(filename2, 'w') as f:
        [print(f'{binary}', file=f) for binary in binary_list]


def change_one_bit(input_binary, index):
    return input_binary[0: index] + str(int(input_binary[index]) ^ 1) + input_binary[index + 1:]


def get_collection_of_inputs(input_binary):
    return [change_one_bit(input_binary, i) for i in range(len(input_binary))]


def calculate_list_of_distinct(first_output, second_output):
    result_list = []
    for i in range(len(first_output)):
        if int(first_output[i]) ^ int(second_output[i]) == 1:
            result_list.append(i)
    return result_list


def calculate_result_output(input_binary_list, input_binary, serpent, index, key):
    return [serpent.encrypt_full_round(input_binary_list[i], key)[index] for i in range(len(input_binary_list))], \
           serpent.encrypt_full_round(input_binary, key)[index]


def calculate_matrix_one_row(result_matrix, first_output, second_output, index):
    change_list = calculate_list_of_distinct(first_output, second_output)
    for var in change_list:
        result_matrix[index][var] += 1


def calculate_matrix_all_rows(matrix, input_binary, serpent, index, key):
    input_binary_list = get_collection_of_inputs(input_binary)
    outputs, first_output = calculate_result_output(input_binary_list, input_binary, serpent, index, key)

    for i in range(len(outputs)):
        calculate_matrix_one_row(matrix, first_output, outputs[i], i)


def calculate_inputs(input_address):
    with open(input_address, 'r') as f:
        lines = f.readlines()
        binary_list = [hexa_to_bin(lines[i]) for i in range(len(lines))]
    return binary_list


def create_round_results(key, input_address):
    serpent = SerpentPlus()
    input_binarys = calculate_inputs(input_address)
    # key = "1011001110011000011001010011011101111110001111001110101101011111000101111010000100111111000000011000010101111110000111111011111100000000001101000000101101101001"
    for i in tqdm(range(serpent.roundNumber + 1)):
        matrix = np.zeros(shape=(160, 160)).astype(int)
        for j in tqdm(range(len(input_binarys))):
            calculate_matrix_all_rows(matrix, input_binarys[j], serpent, i, key)
        np.savetxt(f"Round_{i + 1}.txt", matrix, fmt='%.0f')
