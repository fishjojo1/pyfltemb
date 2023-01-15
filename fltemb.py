import numpy as np


def _byte_emb(arr, byte_seq, mask):
    """
    Embeds a byte sequence into a 2D float32 numpy array. Byte sequence will be embedded into the least
    significant byte of the first N floats, where N is the length of the byte sequence.

    :param arr: 2D float32 numpy array.
    :param byte_seq: python byte sequence to embed into 2D array.
    :param mask: np bool float of size 4 that determines bytes to change
    :return: Embedded 2D float32 numpy array.
    """
    num_zeros = np.count_nonzero(mask)
    section_length = np.ceil(len(byte_seq)/num_zeros).astype(np.uint32)
    byte_seq = byte_seq + b'\x00' * (section_length*num_zeros - len(byte_seq))
    byte_arr = np.frombuffer(arr, dtype=np.uint8)
    byte_seq = np.frombuffer(byte_seq, dtype=np.uint8)
    if mask[3]:
        byte_arr[::4][:section_length] = byte_seq[:section_length]
        byte_seq = byte_seq[section_length:]
    if mask[2]:
        byte_arr[1::4][:section_length] = byte_seq[:section_length]
        byte_seq = byte_seq[section_length:]
    if mask[1]:
        byte_arr[2::4][:section_length] = byte_seq[:section_length]
        byte_seq = byte_seq[section_length:]
    if mask[0]:
        byte_arr[3::4][:section_length] = byte_seq[:section_length]

    return arr


def _byte_ext(emb_arr, seq_len, mask):
    """
    Extracts byte sequence from embedded 2D float32 numpy array. Byte sequence will be extracted from provided mask, up
    till seq_len bytes.
    :param emb_arr: Embedded 2D float32 numpy array.
    :param seq_len: Number of bytes in byte sequence to extract.
    :param mask: numpy bool array of size 4 that determines which bytes of floats to extract from
    :return: Extracted python byte sequence.
    """
    num_zeros = np.count_nonzero(mask)
    section_length = np.ceil(seq_len/num_zeros).astype(np.uint32)
    byte_arr = np.frombuffer(emb_arr, dtype=np.uint8)
    byte_seq = np.array([], np.uint8)
    if mask[3]:
        byte_seq = np.concatenate((byte_seq, byte_arr[::4][:section_length]))
    if mask[2]:
        byte_seq = np.concatenate((byte_seq, byte_arr[1::4][:section_length]))
    if mask[1]:
        byte_seq = np.concatenate((byte_seq, byte_arr[2::4][:section_length]))
    if mask[0]:
        byte_seq = np.concatenate((byte_seq, byte_arr[3::4][:section_length]))
    return byte_seq[:seq_len].tobytes()


def _nib_emb(arr, byte_seq):
    """
    Embeds a byte sequence into a 2D float32 numpy array. Byte sequence will be embedded into the least
    significant nibble of the first 2N floats, where N is the length of the byte sequence.

    :param arr: 2D float32 numpy array.
    :param byte_seq: Byte sequence to embed into 2D array.
    :return: Embedded 2D float32 numpy array.
    """
    byte_arr = np.frombuffer(arr, dtype=np.uint32)
    byte_seq = np.frombuffer(byte_seq, dtype=np.uint8)

    # Splitting bytes into 4 bits each
    temp_seq1 = byte_seq.copy()
    temp_seq2 = byte_seq.copy()
    temp_seq1 >>= 4
    temp_seq2 &= 15

    final_seq = np.vstack((temp_seq1, temp_seq2))
    final_seq = final_seq.T.flatten()

    byte_arr[:final_seq.size] = (byte_arr[:final_seq.size] & (~15)) | final_seq

    return arr


def _nib_ext(emb_arr, seq_len):
    """
    Extracts byte sequence from embedded 2D float32 numpy array. Byte sequence will be extracted from the least
    significant nibble of the first 2N floats, where N is the length of the byte sequence.

    :param emb_arr: Embedded 2D float32 numpy array.
    :param seq_len: Number of bytes in byte sequence to extract.
    :return: Extracted python byte sequence.
    """
    byte_arr = np.frombuffer(emb_arr, dtype=np.uint32)
    process_bytes = byte_arr[:seq_len * 2] & 15
    result = process_bytes[::2] << 4 | process_bytes[1::2]
    return result.astype(np.uint8).tobytes()


def _transpose_bytes(byte_seq, chunk_size, mask):
    """
    Converts byte sequence into chunk_size bit integers given a mask. E.g. Given chunk_size = 4, byte_seq = b"\xff"
    (1111 1111) and mask = np.asarray([0 0 0 ... 1 0 1 0 1 0 1], dtype=bool), where mask.size = 32 and number of 1s in
     mask = chunk_size, output will return [85 85]. 85(base 10) = 1010101(base 2).

    :param byte_seq: Byte sequence to transpose.
    :param chunk_size: Number of bits to transpose byte_seq into.
    :param mask: Numpy bool array of size 32 that masks position to embed in a float.
    :return: uint8 numpy array.
    """
    byte_seq = np.unpackbits(np.frombuffer(byte_seq, dtype=np.uint8))
    byte_seq = np.pad(byte_seq, ((chunk_size - byte_seq.size % chunk_size) % chunk_size, 0))
    byte_seq = np.flip(byte_seq)
    size = byte_seq.size // chunk_size
    byte_seq = np.reshape(byte_seq, (size, chunk_size))

    if mask is not None:
        flip_mask = np.flip(mask)
        result = np.uint8(np.zeros((size, 32)))
        result[:, flip_mask] = byte_seq[:, :]
    else:
        result = np.pad(byte_seq, [(0, 0), (0, 32 - byte_seq[0].size)])

    result = np.packbits(result, bitorder="little", axis=-1).view(np.uint32)
    result = np.flip(result)
    return result.flatten()


def _bit_emb(arr, byte_seq, chunk_size, mask):
    """
    Embeds a byte sequence into a 2D float32 numpy array. Byte sequence will be embedded according to the mask.

    :param arr: 2D float32 numpy array.
    :param byte_seq: Byte sequence to embed into 2D array.
    :param chunk_size: Number of bits to embed at the end of each float.
    :param mask: Numpy bool array of size 32 that masks position to embed in a float.
    :return: Embedded 2D float32 numpy array.
    """
    byte_arr = np.frombuffer(arr, dtype=np.uint32)
    byte_seq = _transpose_bytes(byte_seq, chunk_size, mask)

    if mask is not None:
        flip_mask = np.flip(mask)
        num = np.packbits(flip_mask, bitorder="little").view(np.uint32)
    else:
        num = int('1' * chunk_size, 2)

    byte_arr[:byte_seq.size] = (byte_arr[:byte_seq.size] & (~num)) | byte_seq
    return arr


def _bit_ext(emb_arr, seq_len, chunk_size, mask):
    """
    Extracts a byte sequence from a 2D float32 numpy array. Byte sequence will be extracted according to the mask.

    :param emb_arr: 2D float32 numpy array.
    :param seq_len: Number of bytes in byte sequence to extract.
    :param chunk_size: Number of bits to extract at the end of each float.
    :param mask: Numpy bool array of size 32 that masks position to embed in a float.
    :return: Extracted python byte sequence.
    """
    byte_arr = np.frombuffer(emb_arr, dtype=np.uint32)

    if mask is not None:
        size = int(np.ceil(seq_len * 8 / chunk_size))
        process_bytes = byte_arr[:size]
        process_bytes = np.unpackbits(np.flip(np.frombuffer(process_bytes, dtype=np.uint8)))
        process_bytes = np.reshape(process_bytes, (size, 32))
        result = process_bytes[:, mask]
        result = np.flip(result)[::-1]
        result = np.packbits(result.flatten(), bitorder="little")[::-1]
        return result.astype(np.uint8)[-seq_len:].tobytes()
    else:
        num = int('1' * chunk_size, 2)
        process_bytes = byte_arr[:int(np.ceil(seq_len * 8 / chunk_size))] & num
        process_bytes = np.frombuffer(process_bytes, dtype=np.uint8)
        process_bytes = np.flip(process_bytes)
        process_bytes = np.unpackbits(process_bytes)
        process_bytes = np.reshape(process_bytes, (process_bytes.size // 32, 32))[::-1]
        process_bytes = process_bytes[:, -chunk_size:].flatten()
        process_bytes = np.pad(process_bytes, ((8 - process_bytes.size % 8) % 8, 0))
        process_bytes = np.packbits(process_bytes)
        return process_bytes[-seq_len:].tobytes()


# Main Driver Functions here

class NoParamsProvided(Exception):
    "Raised when either resolution or mask parameters are lacking"
    pass


def embed(arr, byte_seq, mask=None, resolution=None):
    """
    Embeds a python byte sequence into a 2D Numpy array of 32bit floats, given a mask and custom resolution
    :param arr: 2D array of 32bit numpy floats(np.float32) in which to perform embedding of <byte_seq> into
    :param byte_seq: python byte sequence to embed into <arr>
    :param mask: 1D array of 32 np.bool to specify locations of bits to perform embedding
    :param resolution: Allows user to toggle between "bit" and "byte" mode,
    which embeds with a 1 bit and 8 bit resolutions respectively
    :return: Changes <arr> in place and also returns the edited array
    """
    try:
        if resolution is None or mask is None:
            raise NoParamsProvided
        if resolution == "bit":
            if mask is None:
                raise NoParamsProvided
            return _bit_emb(arr, byte_seq, np.count_nonzero(mask), mask)
        elif resolution == "byte":
            if mask is None:
                raise NoParamsProvided
            return _byte_emb(arr, byte_seq, mask[::8])
        else:
            return _byte_emb(arr, byte_seq, [0, 0, 0, 1])
    except NoParamsProvided:
        print("Mask or resolution parameter is None")


def extract(emb_arr, seq_len, mask=None, resolution=None):
    """
    Extracts byte sequence from 2D numpy array of <np.float32>, given a mask. Accepts resolution command of
    "bit" or "byte" to adjust 1 bit or 8 bit level resolution
    :param emb_arr: 2D numpy float32 array of which to extract byte sequence of
    :param seq_len: Number of bytes in byte sequence to extract.
    :param mask: Numpy bool array of size 32 that masks position to embed in a float.
    :param resolution: "Allows user to toggle between "bit" and "byte" mode,
    which embeds with a 1 bit and 8 bit resolutions respectively
    :return: Extracted python byte sequence.
    """
    try:
        if resolution is None and mask is None:
            raise NoParamsProvided
        if resolution == "bit":
            if mask is None:
                raise NoParamsProvided
            return _bit_ext(emb_arr, seq_len, np.count_nonzero(mask), mask)
        elif resolution == "byte":
            if mask is None:
                raise NoParamsProvided
            return _byte_ext(emb_arr, seq_len, mask[::8])
    except NoParamsProvided:
        print("Mask or resolution parameter is None")


def gen_mask(chunk_size):
    """
    Generate a size 32 numpy bool array to represent the mask. 1s will be appended to the right of the array.
    :param chunk_size: Number of bits to embed
    :return: Size 32 numpy book array
    """
    mask = np.asarray(np.concatenate((np.zeros(32 - chunk_size), np.ones(chunk_size))), dtype=bool)
    return mask

