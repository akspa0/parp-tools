import sys
import os
import struct

def read_chunk_header(f):
    try:
        ident_bytes = f.read(4)
        if len(ident_bytes) < 4:
            return None, 0
        ident = ident_bytes[::-1].decode('ascii', errors='replace')
        size_bytes = f.read(4)
        if len(size_bytes) < 4:
            return None, 0
        size = struct.unpack('<I', size_bytes)[0]
        return ident, size
    except Exception as e:
        print(f"Error reading header: {e}")
        return None, 0

def parse_wmo(file_path):
    print(f"Parsing: {file_path}")
    with open(file_path, 'rb') as f:
        file_size = os.path.getsize(file_path)
        while f.tell() < file_size:
            pos = f.tell()
            ident, size = read_chunk_header(f)
            if not ident:
                break
            
            print(f"  [{pos:08X}] {ident} (Size: {size})")
            
            if ident == 'MOMO':
                print(f"    -> Entering MOMO container...")
                end_pos = f.tell() + size
                while f.tell() < end_pos:
                    sub_pos = f.tell()
                    sub_ident, sub_size = read_chunk_header(f)
                    if not sub_ident:
                        break
                    print(f"    [{sub_pos:08X}]   {sub_ident} (Size: {sub_size})")
                    if sub_ident == 'MOGP':
                        print(f"    -> Scanning MOGP (Size: {sub_size})...")
                        curr_pos = f.tell()
                        end_mogp = curr_pos + sub_size
                        
                        # Read full MOGP body
                        data = f.read(sub_size)
                        
                        # Scan for known chunk signatures
                        # MOPY, MOVT, MONR, MOTV, MOIN, MOBA, MOBN, MOBR, MOLV, MOCV, MLIQ
                        signatures = [b'MOPY', b'MOVT', b'MONR', b'MOTV', b'MOIN', b'MOBA', b'MOBN', b'MOBR', b'MOLV', b'MOCV', b'MLIQ']
                        
                        for i in range(len(data) - 4):
                            seg = data[i:i+4]
                            seg_rev = seg[::-1] # WMO chunks often reversed in file
                            
                            found = None
                            if seg_rev in signatures:
                                found = seg_rev.decode('ascii')
                            elif seg in signatures:
                                found = seg.decode('ascii')
                                
                            if found:
                                # Verify size
                                if i + 8 <= len(data):
                                    sz = struct.unpack('<I', data[i+4:i+8])[0]
                                    if sz < 10000000: # Sanity check size
                                        print(f"      [OFFSET {i:06X}] {found} (Size: {sz})")
                        
                        # Restoration
                        f.seek(end_mogp)
                    elif sub_ident == 'MCVP':
                        print(f"    !!!! FOUND MCVP (Convex Volume Planes) !!!! Size: {sub_size}")
                        f.seek(sub_size, 1)
            else:
                f.seek(size, 1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: dump_wmo_chunks.py <wmo_file>")
        sys.exit(1)
    
    parse_wmo(sys.argv[1])
