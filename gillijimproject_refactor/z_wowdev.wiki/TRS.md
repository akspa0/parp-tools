To avoid having duplicates of all-water minimaps and thus wasting space, minimap textures have been stored using an md5 hash of their content as filename. To map from map tile to minimap file, textures/minimap/md5translate.trs exists. The file contains blocks of tile → filename mappings per map in text form.

block_header := "dir: " map_basename "\n"
block_entry := map_basename "\map" x "_" y ".blp\t" actual_filename "\n"
block := block_header block_entry+
file := block+

    map_basename as found in MapRec
    x and y as in ADT tile coordinates. Note that x is not zero-padded but y is zero-padded to two digits. ("map_%d_%02d.blp")
    actual_filename is given relative to textures/minimap/
    note: an earlier version of this page has block_entry := actual_filename "\t" map_basename "\map" x "_" y ".blp\n"