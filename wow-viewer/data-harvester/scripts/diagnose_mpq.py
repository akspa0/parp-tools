import struct, sys

path = r'I:\parp\parp-tools\output\tmp\wowarchive-clients\0_5_3_3368\World of Warcraft\Data\World\Maps\Kalimdor\Kalimdor.wdt.MPQ'
with open(path, 'rb') as f:
    f.seek(0)
    h = f.read(4)
    magic = struct.unpack("<I", h)[0]
    print(f'Offset 0: magic = 0x{magic:08X} = {h}')

    # Search for MPQ header signature across the file
    f.seek(0, 2)
    total_size = f.tell()
    print(f'File size: {total_size:,} bytes')

    f.seek(0)
    found = False
    for offset in range(0, min(total_size - 4, 0x100000), 0x200):
        f.seek(offset)
        if f.read(4) == b'MPQ\x1a':
            f.seek(offset)
            header_data = f.read(32)
            if len(header_data) < 32:
                break
            hs = struct.unpack_from('<I', header_data, 4)[0]
            as_ = struct.unpack_from('<I', header_data, 8)[0]
            fv = struct.unpack_from('<H', header_data, 12)[0]
            ss = struct.unpack_from('<H', header_data, 14)[0]
            ho = struct.unpack_from('<I', header_data, 16)[0]
            ht = struct.unpack_from('<I', header_data, 20)[0]
            bt = struct.unpack_from('<I', header_data, 24)[0]
            print(f'\nMPQ header at offset 0x{offset:X}:')
            print(f'  headerSize={hs}, archiveSize={as_:,}, formatVersion={fv}')
            print(f'  sectorSize={ss}, hashTableOffset={ho}, hashTableEntries={ht}, blockTableOffset={bt}')

            # Read first few blocks
            f.seek(offset + bt)
            print(f'  First 20 block entries (offset, blockSize, fileSize, flags):')
            for j in range(min(ht, 20)):
                be = f.read(16)
                if len(be) < 16:
                    break
                bo, bs, fs, fl = struct.unpack('<IIIi', be)
                label = ""
                if bs > 0 and fs > 0:
                    f_pos = f.tell()
                    f.seek(offset + bo)
                    block_start = f.read(8)
                    f.seek(f_pos)
                    if len(block_start) >= 4:
                        if block_start[:4] in (b'REVM', b'MVER', b'ERAM', b'MARE', b'FOAM', b'MAOF'):
                            label = f' <- {block_start[:4].decode("ascii", errors="replace")}'
                        elif bs > 1000000 and fs > 1000000:
                            label = ' <- LARGE (WDT?)'
                print(f'    [{j}] offset=0x{bo:08X} blockSize={bs:,} fileSize={fs:,} flags=0x{fl:08X}{label}')
            found = True
            break

    if not found:
        print('No MPQ header found in first 1MB')
